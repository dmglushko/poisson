"""Command-line entry point for experiment scripts, shared by laptop runs and SLURM jobs.

An experiment script ends with::

    if __name__ == "__main__":
        raise SystemExit(experiment_main(pipeline))

and then accepts:

    python3 sweep.py                       # run every branch here (one SLURM array task: its share)
    python3 sweep.py --count               # dry run: how many branches, how many stage runs needed
    python3 sweep.py --shard 3/16          # run piece 3 of 16
    python3 sweep.py --warmup              # run only the first branch (fills shared early steps)
    python3 sweep.py --collect --out DIR   # gather every piece's cached results into one saved run

DAG mode (each stage call its own Slurm array task, see `dag.py`):

    python3 sweep.py --dag-plan PLAN.json [--resources cpus=1,time=01:00:00,mem-per-cpu=3900M]
                     [--stage-resources solve:cpus=8,time=02:00:00] [--split-by mesh]
    python3 sweep.py --dag-plan PLAN.json --dag-task 2/5   # run task 5 of array 2

Every mode also takes `--select NAME=LABEL` (repeatable; only branches whose choice NAME has one
of the given labels, several names must all match) and `--shard-order stride|block`.

`scripts/delftblue/submit.sh` drives these options on the cluster.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .cache import Cache
from .driver import CacheMissError, run_experiment
from .report import ExperimentReport

__all__ = ["experiment_main", "parse_select", "parse_shard"]

EXIT_OK = 0
EXIT_BRANCH_FAILED = 1
EXIT_COLLECT_REFUSED = 2
EXIT_USAGE = 2

_SHARD_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_shard(text: str) -> tuple[int, int]:
    """Parse `"I/N"` into `(I, N)` with `0 <= I < N`."""
    m = _SHARD_RE.match(text)
    if m is None:
        raise argparse.ArgumentTypeError(f"expected INDEX/TOTAL such as 3/16, got {text!r}")
    index, total = int(m.group(1)), int(m.group(2))
    if total < 1 or not (0 <= index < total):
        raise argparse.ArgumentTypeError(f"shard {text!r} needs 0 <= INDEX < TOTAL")
    return index, total


def parse_dag_task(text: str) -> tuple[int, int]:
    """Parse `"G/I"` (array G, task I of it) into `(G, I)`."""
    m = _SHARD_RE.match(text)
    if m is None:
        raise argparse.ArgumentTypeError(f"expected ARRAY/TASK such as 2/5, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_select(text: str) -> tuple[str, str]:
    """Parse `"NAME=LABEL"` into `(NAME, LABEL)`."""
    name, sep, label = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=LABEL such as mesh=fine, got {text!r}")
    return name.strip(), label


def _selection(pairs: Sequence[tuple[str, str]] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, label in pairs or ():
        out.setdefault(name, []).append(label)
    return out


def _build_parser(prog: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run this experiment (see poisson_runner.experiment_cli).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", action="store_true", help="dry run: report branch count and stage runs needed")
    mode.add_argument("--shard", type=parse_shard, metavar="I/N", help="run only piece I of N (0-based)")
    mode.add_argument("--warmup", action="store_true", help="run only the first branch, to fill shared early steps")
    mode.add_argument("--collect", action="store_true", help="gather cached results of all pieces and save them")
    mode.add_argument("--dag-task", type=parse_dag_task, metavar="G/I", help="with --dag-plan: run task I of array G")
    parser.add_argument(
        "--select",
        type=parse_select,
        action="append",
        metavar="NAME=LABEL",
        help="only branches whose compare(NAME) chose LABEL (repeat: same NAME = any of, different = all of)",
    )
    parser.add_argument(
        "--shard-order",
        choices=("stride", "block"),
        default="stride",
        help="with --shard: every N-th branch (stride, default) or one contiguous run of branches (block)",
    )
    dag = parser.add_argument_group("DAG mode")
    dag.add_argument("--dag-plan", type=Path, metavar="FILE", help="write the DAG plan here (or, with --dag-task, read it)")
    dag.add_argument("--resources", metavar="K=V,...", help="default task resources: cpus, time, mem-per-cpu")
    dag.add_argument(
        "--stage-resources",
        action="append",
        metavar="STAGE:K=V,...",
        help="resources of one stage's tasks, e.g. solve:cpus=8,time=02:00:00 (repeatable)",
    )
    dag.add_argument("--split-by", metavar="NAME", help="separate arrays per option of compare(NAME)")
    dag.add_argument("--max-array-size", type=int, default=1000, metavar="N", help="split larger arrays [1000]")
    dag.add_argument(
        "--no-fuse",
        action="store_true",
        help="one task per stage call, even for chains that could run back to back in one job",
    )
    dag.add_argument(
        "--max-fused-time",
        metavar="T",
        help="don't fuse a chain past this summed time limit [24:00:00, DelftBlue's education limit]",
    )
    parser.add_argument("--out", type=Path, help="save the run here (required with --collect)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing saved run in --out")
    parser.add_argument(
        "--allow-recompute",
        action="store_true",
        help="with --collect: run stages that are missing from the cache instead of refusing",
    )
    return parser


def _print_summary(report: ExperimentReport, label: str) -> None:
    failed = [b for b in report.branches if not b.ok]
    print(
        f"[{label}] branches: {len(report.branches)}, failed: {len(failed)}, "
        f"stage runs: {report.cache_misses} executed, {report.cache_hits} cached, {report.reused} reused, "
        f"{report.total_wall_time_s:.1f} s"
    )
    for b in failed:
        print(f"[{label}] branch {b.index} {b.choice_labels} failed: {b.error}", file=sys.stderr)


def experiment_main(
    pipeline_fn: Callable[..., Any],
    *,
    argv: Sequence[str] | None = None,
    cache: Cache | None = None,
    prog: str | None = None,
    **run_kwargs: Any,
) -> int:
    """Parse the command line, run `pipeline_fn` accordingly, and return a process exit code.

    `run_kwargs` go to `run_experiment` (e.g. `repeats=`, `stage_cache_modes=`); the mode flags
    override `shard`, `dry_run`, `max_branches` and, for `--collect`, the cache modes.

    Exit codes: 0 success; 1 some branch failed (so a SLURM `afterok` dependency doesn't start
    the collector on incomplete results); 2 `--collect` refused because stages are missing.
    """
    args = _build_parser(prog).parse_args(argv)
    for key in ("shard", "shard_order", "select", "dry_run", "max_branches", "require_cached", "force_cache_mode"):
        if key in run_kwargs:
            raise TypeError(f"experiment_main sets {key!r} from the command line; don't pass it")
    if args.collect and args.out is None:
        print("--collect needs --out DIR", file=sys.stderr)
        return EXIT_COLLECT_REFUSED
    cache = cache if cache is not None else Cache()
    selection = _selection(args.select)
    if selection:
        run_kwargs["select"] = selection

    dag_only = (args.resources, args.stage_resources, args.split_by, True if args.no_fuse else None,
                args.max_fused_time)
    if args.dag_plan is None:
        if args.dag_task is not None or any(v is not None for v in dag_only):
            print("--dag-task, --resources, --stage-resources, --split-by, --no-fuse and --max-fused-time "
                  "need --dag-plan", file=sys.stderr)
            return EXIT_USAGE
    elif args.dag_task is not None:
        if any(v is not None for v in dag_only) or selection or args.out is not None:
            print("--dag-task takes only --dag-plan (the plan holds everything else)", file=sys.stderr)
            return EXIT_USAGE
        return _run_dag_task(pipeline_fn, args, cache, run_kwargs)
    elif args.count or args.shard or args.warmup or args.collect or args.out is not None:
        print("--dag-plan writes a plan; it takes no other mode or --out", file=sys.stderr)
        return EXIT_USAGE
    else:
        return _plan_dag(pipeline_fn, args, cache, run_kwargs)

    if args.count:
        report = run_experiment(pipeline_fn, cache=cache, dry_run=True, **run_kwargs)
        print(
            f"{report.cache_misses} stage runs needed, {report.cache_hits} already cached, "
            f"{report.reused} shared between branches"
        )
        print(f"branches: {len(report.branches)}")  # machine-readable; submit.sh reads this line
        return EXIT_OK

    if args.collect:
        collect_kwargs = dict(run_kwargs, cache_mode="read_only", stage_cache_modes={}, force_cache_mode="read_only")
        if not args.allow_recompute:
            # Enforce cache hits at the point of use (under one lease) so a concurrent eviction
            # between a dry-run check and the real pass can never execute a stage.
            collect_kwargs["continue_on_error"] = False
            try:
                report = run_experiment(pipeline_fn, cache=cache, require_cached=True, **collect_kwargs)
            except CacheMissError as miss:
                check = run_experiment(pipeline_fn, cache=cache, dry_run=True, **collect_kwargs)
                missing = [
                    (b.index, s.stage) for b in check.branches for s in b.stages if s.executed
                ]
                if missing:
                    shown = ", ".join(f"branch {i}: {stage}" for i, stage in missing[:10])
                    more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
                    detail = f"{len(missing)} stage runs are not in the cache ({shown}{more})"
                else:
                    detail = f"stage '{miss.stage_name}' missed the cache during collection"
                print(
                    f"Refusing to collect: {detail}. "
                    "Did a piece fail or run with a different build? Rerun those pieces, or pass "
                    "--allow-recompute to compute them here.",
                    file=sys.stderr,
                )
                return EXIT_COLLECT_REFUSED
        else:
            report = run_experiment(pipeline_fn, cache=cache, **collect_kwargs)
        label = "collect"
    elif args.warmup:
        report = run_experiment(pipeline_fn, cache=cache, max_branches=1, **run_kwargs)
        label = "warmup"
    else:
        shard: tuple[int, int] | str | None = args.shard
        if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
            shard = "slurm"
        if shard is not None:
            run_kwargs.setdefault("continue_on_error", True)
        report = run_experiment(
            pipeline_fn, cache=cache, shard=shard, shard_order=args.shard_order, **run_kwargs  # type: ignore[arg-type]
        )
        label = f"piece {report.shard[0]}/{report.shard[1]}" if report.shard is not None else "run"

    print(report.format_table())
    _print_summary(report, label)
    if args.out is not None:
        partial = report.shard is not None or bool(selection)
        saved = report.save(args.out, overwrite=args.overwrite, allow_partial=partial)
        print(f"[{label}] saved to {saved}")
    return EXIT_BRANCH_FAILED if any(not b.ok for b in report.branches) else EXIT_OK


def _plan_dag(pipeline_fn: Callable[..., Any], args: argparse.Namespace, cache: Cache, run_kwargs: dict[str, Any]) -> int:
    from .dag import DEFAULT_MAX_FUSED_TIME, parse_resources, parse_stage_resources, plan_dag, time_seconds

    try:
        defaults = parse_resources(args.resources, where="--resources") if args.resources else {}
        per_stage: dict[str, dict[str, Any]] = {}
        for text in args.stage_resources or ():
            stage, res = parse_stage_resources(text)
            per_stage.setdefault(stage, {}).update(res)
        max_fused = args.max_fused_time if args.max_fused_time is not None else DEFAULT_MAX_FUSED_TIME
        if time_seconds(max_fused) <= 0:
            raise ValueError(f"--max-fused-time must be a positive time limit, got {max_fused!r}")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    plan = plan_dag(
        pipeline_fn,
        cache=cache,
        resources=defaults,
        stage_resources=per_stage,
        split_by=args.split_by,
        max_array_size=args.max_array_size,
        fuse=not args.no_fuse,
        max_fused_time=max_fused,
        **run_kwargs,
    )
    saved = plan.save(args.dag_plan)
    fused = len(plan.tasks) - len(plan.units)
    print(
        f"[dag] {len(plan.tasks)} stage runs needed in {len(plan.groups)} arrays, "
        f"{plan.cached_calls} already cached, {plan.shared_calls} shared between branches"
        f"{f', {fused} fused into the job before them' if fused else ''}; plan: {saved}"
    )
    for line in plan.group_lines():
        print(f"dag-group {line}")  # machine-readable; submit.sh reads these lines
    return EXIT_OK


def _run_dag_task(pipeline_fn: Callable[..., Any], args: argparse.Namespace, cache: Cache, run_kwargs: dict[str, Any]) -> int:
    from .dag import run_dag_task

    group, element = args.dag_task
    result = run_dag_task(pipeline_fn, args.dag_plan, group, element, cache=cache, **run_kwargs)
    for t in result.traces:
        timing = f", {t.time_s:.2f} s" if t.time_s is not None else ""
        print(
            f"[dag task {group}/{element}] branch {result.branch}, stage '{t.stage}': "
            f"{'already cached' if t.cached else 'executed'}{timing} [{t.status}]"
        )
        for w in t.warnings:
            print(f"[dag task {group}/{element}] warning: {w}", file=sys.stderr)
    return EXIT_OK
