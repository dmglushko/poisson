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

The script can set the defaults of these options in Python, e.g.
`experiment_main(pipeline, mode="dag", resources={"time": "00:30:00"}, split_by="mesh")`;
a flag given on the command line wins. `--show-settings [--resources ...] [--shard-order ...]`
prints the resolved mode, resources and shard order without running anything.

`scripts/delftblue/submit.sh` drives these options on the cluster.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import BUNDLE_INCLUDES, format_bytes
from .cache import Cache
from .driver import (
    CacheMissError,
    _discover_branch_paths,
    _make_config,
    _normalize_select,
    _select_paths,
    run_experiment,
)
from .report import ExperimentReport
from .python_stages import CHILD_ENV as PYTHON_STAGE_CHILD_ENV

__all__ = ["MODES", "experiment_main", "parse_select", "parse_shard"]

MODES = ("shard", "dag")
SHARD_ORDERS = ("stride", "block")

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
    mode.add_argument(
        "--show-settings",
        action="store_true",
        help="print the resolved mode, resources and shard order (`setting KEY=VALUE` lines); runs nothing",
    )
    parser.add_argument(
        "--select",
        type=parse_select,
        action="append",
        metavar="NAME=LABEL",
        help="only branches whose compare(NAME) chose LABEL (repeat: same NAME = any of, different = all of)",
    )
    parser.add_argument(
        "--shard-order",
        choices=SHARD_ORDERS,
        help="with --shard: every N-th branch (stride) or one contiguous run of branches (block) "
        "[the script's shard_order, else stride]",
    )
    dag = parser.add_argument_group("DAG mode")
    dag.add_argument("--dag-plan", type=Path, metavar="FILE", help="write the DAG plan here (or, with --dag-task, read it)")
    dag.add_argument(
        "--resources",
        metavar="K=V,...",
        help="default task resources: cpus, time, mem-per-cpu (over the script's resources=, key by key)",
    )
    dag.add_argument(
        "--stage-resources",
        action="append",
        metavar="STAGE:K=V,...",
        help="resources of one stage's tasks, e.g. solve:cpus=8,time=02:00:00 (repeatable); over the "
        "script's stage_resources= and its run_stage(..., resources=), key by key",
    )
    split = dag.add_mutually_exclusive_group()
    split.add_argument(
        "--split-by",
        action="append",
        metavar="NAME",
        help="separate arrays per option of compare(NAME) (repeatable); replaces the script's split_by= "
        "and compare(..., split=True)",
    )
    split.add_argument("--no-split", action="store_true", help="ignore the script's split_by= and split=True")
    dag.add_argument(
        "--max-array-size", type=int, metavar="N", help="split larger arrays [the script's, else 1000]"
    )
    fuse = dag.add_mutually_exclusive_group()
    fuse.add_argument(
        "--no-fuse",
        action="store_true",
        help="one task per stage call, even for chains that could run back to back in one job",
    )
    fuse.add_argument("--fuse", action="store_true", help="fuse chains even if the script sets fuse=False")
    dag.add_argument(
        "--max-fused-time",
        metavar="T",
        help="don't fuse a chain past this summed time limit [the script's, else 24:00:00, DelftBlue's "
        "education limit]",
    )
    parser.add_argument("--out", type=Path, help="save the run here (required with --collect)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing saved run in --out")
    parser.add_argument(
        "--bundle",
        nargs="?",
        const="outputs",
        choices=BUNDLE_INCLUDES,
        default=None,
        help="with --out: link or copy artifacts into <out>/objects/ (outputs [default] or stages)",
    )
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


@dataclass(frozen=True)
class _Defaults:
    """The execution defaults an experiment script passes to `experiment_main`, validated."""

    mode: str
    resources: dict[str, Any]
    stage_resources: dict[str, dict[str, Any]]
    split_by: list[str]
    fuse: bool
    max_fused_time: str
    max_array_size: int
    shard_order: str


def _validate_defaults(
    *,
    mode: str,
    resources: Mapping[str, Any] | None,
    stage_resources: Mapping[str, Mapping[str, Any]] | None,
    split_by: str | Sequence[str] | None,
    fuse: bool,
    max_fused_time: str | None,
    max_array_size: int,
    shard_order: str,
) -> _Defaults:
    from .dag import DEFAULT_MAX_FUSED_TIME, normalize_resources, normalize_split_by, time_seconds

    if mode not in MODES:
        raise ValueError(f"experiment_main: mode must be one of {', '.join(MODES)}, got {mode!r}")
    if shard_order not in SHARD_ORDERS:
        raise ValueError(f"experiment_main: shard_order must be one of {', '.join(SHARD_ORDERS)}, got {shard_order!r}")
    if not isinstance(fuse, bool):
        raise ValueError(f"experiment_main: fuse must be True or False, got {fuse!r}")
    if isinstance(max_array_size, bool) or not isinstance(max_array_size, int) or max_array_size < 1:
        raise ValueError(f"experiment_main: max_array_size must be a positive integer, got {max_array_size!r}")
    fused = DEFAULT_MAX_FUSED_TIME if max_fused_time is None else max_fused_time
    try:
        ok = isinstance(fused, str) and time_seconds(fused) > 0
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"experiment_main: max_fused_time must be a positive time limit, got {fused!r}")
    if resources is not None and not isinstance(resources, Mapping):
        raise ValueError(f"experiment_main: resources must be a dict, got {resources!r}")
    if stage_resources is not None and not isinstance(stage_resources, Mapping):
        raise ValueError(f"experiment_main: stage_resources must be a dict of dicts, got {stage_resources!r}")
    per_stage: dict[str, dict[str, Any]] = {}
    for name, res in (stage_resources or {}).items():
        if not isinstance(res, Mapping):
            raise ValueError(f"experiment_main: stage_resources[{name!r}] must be a dict, got {res!r}")
        per_stage[str(name)] = normalize_resources(res, where=f"experiment_main stage_resources[{name!r}]")
    return _Defaults(
        mode=mode,
        resources=normalize_resources(resources or {}, where="experiment_main resources"),
        stage_resources=per_stage,
        split_by=normalize_split_by(split_by),
        fuse=fuse,
        max_fused_time=fused.strip(),
        max_array_size=max_array_size,
        shard_order=shard_order,
    )


def experiment_main(
    pipeline_fn: Callable[..., Any],
    *,
    argv: Sequence[str] | None = None,
    cache: Cache | None = None,
    prog: str | None = None,
    mode: str = "shard",
    resources: Mapping[str, Any] | None = None,
    stage_resources: Mapping[str, Mapping[str, Any]] | None = None,
    split_by: str | Sequence[str] | None = None,
    fuse: bool = True,
    max_fused_time: str | None = None,
    max_array_size: int = 1000,
    shard_order: str = "stride",
    **run_kwargs: Any,
) -> int:
    """Parse the command line, run `pipeline_fn` accordingly, and return a process exit code.

    The keyword arguments after `prog` are the script's execution defaults; the matching command
    line flag wins over each (`--resources` and `--stage-resources` key by key):

    - `mode`: `"shard"` or `"dag"`, which `submit.sh` uses unless given `--dag` or `--shards`.
      A plain `python3 sweep.py` always runs every branch here.
    - `resources`: cpus, time and mem_per_cpu of every DAG task and of every shard piece.
    - `stage_resources`: per stage name, the resources of its DAG tasks (below a call's own
      `run_stage(..., resources=)`, which the command line's `--stage-resources` overrides).
    - `split_by`: `compare()` names whose options get separate DAG arrays, as
      `compare(..., split=True)` does (`--split-by` replaces both, `--no-split` clears both).
    - `fuse`, `max_fused_time`, `max_array_size`: DAG chain fusion and array size limits.
    - `shard_order`: `"stride"` or `"block"`, how branches are dealt to pieces.

    The other `run_kwargs` go to `run_experiment` (e.g. `repeats=`, `stage_cache_modes=`); the
    mode flags override `shard`, `dry_run`, `max_branches` and, for `--collect`, the cache modes.

    Exit codes: 0 success; 1 some branch failed (so a SLURM `afterok` dependency doesn't start
    the collector on incomplete results); 2 `--collect` refused because stages are missing.

    Inside the process of a `@python_stage` function defined in this script it returns 0 at
    once, so loading the script to call that function doesn't start the experiment again.
    """
    if os.environ.get(PYTHON_STAGE_CHILD_ENV):
        return 0
    defaults = _validate_defaults(
        mode=mode,
        resources=resources,
        stage_resources=stage_resources,
        split_by=split_by,
        fuse=fuse,
        max_fused_time=max_fused_time,
        max_array_size=max_array_size,
        shard_order=shard_order,
    )
    for key in ("shard", "select", "dry_run", "max_branches", "require_cached", "force_cache_mode"):
        if key in run_kwargs:
            raise TypeError(f"experiment_main sets {key!r} from the command line; don't pass it")
    args = _build_parser(prog).parse_args(argv)
    shard_order = args.shard_order or defaults.shard_order
    if args.show_settings:
        return _show_settings(args, defaults, shard_order)
    if args.collect and args.out is None:
        print("--collect needs --out DIR", file=sys.stderr)
        return EXIT_COLLECT_REFUSED
    selection = _selection(args.select)
    if selection:
        run_kwargs["select"] = selection

    dag_only = (args.resources, args.stage_resources, args.split_by, True if args.no_split else None,
                True if args.no_fuse else None, True if args.fuse else None, args.max_fused_time,
                args.max_array_size)
    if args.dag_plan is None:
        if args.dag_task is not None or any(v is not None for v in dag_only):
            print("--dag-task, --resources, --stage-resources, --split-by, --no-split, --fuse, --no-fuse, "
                  "--max-fused-time and --max-array-size need --dag-plan (--resources also goes with "
                  "--show-settings)", file=sys.stderr)
            return EXIT_USAGE
    elif args.dag_task is not None:
        if any(v is not None for v in dag_only) or selection or args.out is not None:
            print("--dag-task takes only --dag-plan (the plan holds everything else)", file=sys.stderr)
            return EXIT_USAGE
        return _run_dag_task(pipeline_fn, args, _cache(cache), run_kwargs)
    elif args.count or args.shard or args.warmup or args.collect or args.out is not None or args.bundle is not None:
        print("--dag-plan writes a plan; it takes no other mode, --out, or --bundle", file=sys.stderr)
        return EXIT_USAGE
    else:
        return _plan_dag(pipeline_fn, args, defaults, _cache(cache), run_kwargs)
    cache = _cache(cache)

    if args.count:
        report = run_experiment(pipeline_fn, cache=cache, dry_run=True, **run_kwargs)
        print(
            f"{report.cache_misses} stage runs needed, {report.cache_hits} already cached, "
            f"{report.reused} shared between branches"
        )
        stopped = [b for b in report.branches if b.dry_run_stopped is not None]
        if stopped:
            print(
                f"note: {len(stopped)} branch(es) stopped during dry run at a stage-output read "
                f"({stopped[0].dry_run_stopped}); later stage calls and compare()s are not counted"
            )
        # Cross-check against cache-free discovery (what sharded pieces actually use) so a warm
        # cache never hides a compare() that follows a stage-output read.
        cfg_kwargs = {k: v for k, v in run_kwargs.items() if k in {
            "executor", "build_dir", "cache_mode", "stage_cache_modes",
            "repeats", "warmup", "stage_repeats", "stage_warmup",
        }}
        disc_cfg = _make_config(cache=cache, dry_run=True, **cfg_kwargs)
        takes_ctx = len(inspect.signature(pipeline_fn).parameters) > 0
        disc_all = _discover_branch_paths(pipeline_fn, takes_ctx=takes_ctx, config=disc_cfg)
        mismatch_reason: str | None = None
        try:
            disc_selected = _select_paths(list(enumerate(disc_all)), _normalize_select(selection))
        except ValueError as exc:
            mismatch_reason = str(exc)
        else:
            disc_labels = [{s.name: s.label for s in b.path} for _, b in disc_selected]
            count_labels = [b.choice_labels for b in report.branches]
            if disc_labels != count_labels:
                mismatch_reason = (
                    f"{len(count_labels)} branch(es) seen with cache vs {len(disc_labels)} without cache; "
                    "some compare() follows a stage-output read and pieces (shards) cannot plan it"
                )
        if mismatch_reason is not None:
            print(f"warning: {mismatch_reason}", file=sys.stderr)
            print("shard-incompatible: compare-after-read")  # machine-readable; submit.sh checks this
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
            pipeline_fn, cache=cache, shard=shard, shard_order=shard_order, **run_kwargs  # type: ignore[arg-type]
        )
        label = f"piece {report.shard[0]}/{report.shard[1]}" if report.shard is not None else "run"

    print(report.format_table())
    _print_summary(report, label)
    bundle_failed = False
    if args.out is not None:
        partial = report.shard is not None or bool(selection)
        bundle_mode = "link_or_copy" if args.bundle is not None else "link"
        bundle_include = args.bundle if args.bundle is not None else "outputs"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            saved = report.save(
                args.out,
                overwrite=args.overwrite,
                allow_partial=partial,
                bundle=bundle_mode,
                include=bundle_include,
            )
        print(f"[{label}] saved to {saved}")
        summary = report._last_bundle_summary
        if summary is not None:
            if summary.copied:
                print(
                    f"[{label}] bundled {summary.copied} copied ({format_bytes(summary.copied_bytes)})"
                    f"{f', {summary.linked} linked' if summary.linked else ''}"
                )
            elif summary.left_in_cache > 0:
                where = str(report.cache_root) if report.cache_root is not None else "the cache"
                print(
                    f"[{label}] {summary.left_in_cache} result file(s) ({format_bytes(summary.left_in_cache_bytes)}) "
                    f"are only in {where}; keep them with: python3 -m poisson_runner run bundle {saved}"
                )
            if args.bundle is not None and summary.missing > 0:
                print(
                    f"[{label}] error: {summary.missing} artifact(s) missing while bundling: "
                    f"{', '.join(summary.missing_objects[:10])}",
                    file=sys.stderr,
                )
                bundle_failed = True
    return EXIT_BRANCH_FAILED if (bundle_failed or any(not b.ok for b in report.branches)) else EXIT_OK


def _cache(cache: Cache | None) -> Cache:
    return cache if cache is not None else Cache()


def _cli_resources(args: argparse.Namespace, defaults: _Defaults) -> dict[str, Any]:
    """The script's `resources` with `--resources` over them, key by key."""
    from .dag import parse_resources

    return {**defaults.resources, **(parse_resources(args.resources, where="--resources") if args.resources else {})}


def _show_settings(args: argparse.Namespace, defaults: _Defaults, shard_order: str) -> int:
    """Print the resolved execution settings for `submit.sh`, without creating a cache or
    running the pipeline. Only `--resources` and `--shard-order` may be combined with it."""
    from .dag import DEFAULT_RESOURCES

    extra = [
        flag for flag, value in (
            ("--select", args.select), ("--dag-plan", args.dag_plan), ("--stage-resources", args.stage_resources),
            ("--split-by", args.split_by), ("--no-split", args.no_split or None), ("--fuse", args.fuse or None),
            ("--no-fuse", args.no_fuse or None), ("--max-fused-time", args.max_fused_time),
            ("--max-array-size", args.max_array_size), ("--out", args.out), ("--overwrite", args.overwrite or None),
            ("--bundle", args.bundle), ("--allow-recompute", args.allow_recompute or None),
        ) if value is not None
    ]
    if extra:
        print(f"--show-settings takes only --resources and --shard-order, not {', '.join(extra)}", file=sys.stderr)
        return EXIT_USAGE
    try:
        res = {**DEFAULT_RESOURCES, **_cli_resources(args, defaults)}
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    # Machine-readable; submit.sh reads these lines.
    print(f"setting mode={defaults.mode}")
    print(f"setting cpus={res['cpus']}")
    print(f"setting time={res['time']}")
    print(f"setting mem_per_cpu={res['mem_per_cpu']}")
    print(f"setting shard_order={shard_order}")
    return EXIT_OK


def _plan_dag(
    pipeline_fn: Callable[..., Any],
    args: argparse.Namespace,
    defaults: _Defaults,
    cache: Cache,
    run_kwargs: dict[str, Any],
) -> int:
    from .dag import parse_stage_resources, plan_dag, time_seconds

    try:
        resources = _cli_resources(args, defaults)
        overrides: dict[str, dict[str, Any]] = {}
        for text in args.stage_resources or ():
            stage, res = parse_stage_resources(text)
            overrides.setdefault(stage, {}).update(res)
        max_fused = args.max_fused_time if args.max_fused_time is not None else defaults.max_fused_time
        if time_seconds(max_fused) <= 0:
            raise ValueError(f"--max-fused-time must be a positive time limit, got {max_fused!r}")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    if args.split_by:
        split_by, compare_splits = args.split_by, False
    elif args.no_split:
        split_by, compare_splits = [], False
    else:
        split_by, compare_splits = defaults.split_by, True
    fuse = True if args.fuse else False if args.no_fuse else defaults.fuse
    plan = plan_dag(
        pipeline_fn,
        cache=cache,
        resources=resources,
        stage_resources=defaults.stage_resources,
        stage_resource_overrides=overrides,
        split_by=split_by,
        compare_splits=compare_splits,
        max_array_size=args.max_array_size if args.max_array_size is not None else defaults.max_array_size,
        fuse=fuse,
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
