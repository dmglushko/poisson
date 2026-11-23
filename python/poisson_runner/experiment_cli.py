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

__all__ = ["experiment_main", "parse_shard"]

EXIT_OK = 0
EXIT_BRANCH_FAILED = 1
EXIT_COLLECT_REFUSED = 2

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


def _build_parser(prog: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run this experiment (see poisson_runner.experiment_cli).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", action="store_true", help="dry run: report branch count and stage runs needed")
    mode.add_argument("--shard", type=parse_shard, metavar="I/N", help="run only piece I of N (0-based)")
    mode.add_argument("--warmup", action="store_true", help="run only the first branch, to fill shared early steps")
    mode.add_argument("--collect", action="store_true", help="gather cached results of all pieces and save them")
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
    for key in ("shard", "dry_run", "max_branches", "require_cached"):
        if key in run_kwargs:
            raise TypeError(f"experiment_main sets {key!r} from the command line; don't pass it")
    if args.collect and args.out is None:
        print("--collect needs --out DIR", file=sys.stderr)
        return EXIT_COLLECT_REFUSED
    cache = cache if cache is not None else Cache()

    if args.count:
        report = run_experiment(pipeline_fn, cache=cache, dry_run=True, **run_kwargs)
        print(
            f"{report.cache_misses} stage runs needed, {report.cache_hits} already cached, "
            f"{report.reused} shared between branches"
        )
        print(f"branches: {len(report.branches)}")  # machine-readable; submit.sh reads this line
        return EXIT_OK

    if args.collect:
        collect_kwargs = dict(run_kwargs, cache_mode="read_only", stage_cache_modes={})
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
        report = run_experiment(pipeline_fn, cache=cache, shard=shard, **run_kwargs)  # type: ignore[arg-type]
        label = f"piece {report.shard[0]}/{report.shard[1]}" if report.shard is not None else "run"

    print(report.format_table())
    _print_summary(report, label)
    if args.out is not None:
        saved = report.save(args.out, overwrite=args.overwrite, allow_partial=report.shard is not None)
        print(f"[{label}] saved to {saved}")
    return EXIT_BRANCH_FAILED if any(not b.ok for b in report.branches) else EXIT_OK
