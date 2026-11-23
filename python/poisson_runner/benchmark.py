"""Warmup/repeat execution of one stage command, per-run statistics, and determinism checks."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .executor import ExecutionRequest, ExecutionResult, Executor


def summarize_numbers(samples: Sequence[float | int]) -> dict[str, Any]:
    """Return `count`/`min`/`median`/`mean`/`max`/`std` plus the raw `samples`."""
    vals = [float(x) for x in samples]
    return {
        "count": len(vals),
        "min": min(vals),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "max": max(vals),
        "std": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
        "samples": list(samples),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


@dataclass(frozen=True)
class Measurement:
    """Outcome of `measure(...)`: last run's info plus the aggregated execution metadata."""

    info: dict[str, Any]
    execution: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def measure(
    executor: Executor,
    request: ExecutionRequest,
    *,
    repeats: int,
    warmup: int,
    outputs: Mapping[str, Path],
    info_path: Path,
    hash_file: Callable[[Path], str],
) -> Measurement:
    """Run `warmup` discarded + `repeats` measured executions and aggregate their metrics.

    Every run (warmup included) rewrites the same output paths. When more than one run happens,
    each run's outputs are hashed and compared against the first run; differing outputs are
    reported as warnings and listed in `execution["nondeterministic_outputs"]`.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    check_determinism = repeats + warmup > 1
    reference_ids: dict[str, str] | None = None
    nondeterministic: set[str] = set()

    def run_once(phase: str) -> ExecutionResult:
        nonlocal reference_ids
        info_path.unlink(missing_ok=True)
        result = executor.run(request)
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            during = " during warmup" if phase == "warmup" else ""
            raise RuntimeError(
                f"Stage '{request.stage_name}' failed{during} with exit code {result.exit_code}: {detail}"
            )
        if check_determinism:
            ids = {name: hash_file(path) for name, path in outputs.items() if path.is_file()}
            if reference_ids is None:
                reference_ids = ids
            else:
                for name, cid in ids.items():
                    if reference_ids.get(name) != cid:
                        nondeterministic.add(name)
        return result

    for _ in range(warmup):
        run_once("warmup")

    runs: list[ExecutionResult] = []
    infos: list[dict[str, Any]] = []
    for _ in range(repeats):
        runs.append(run_once("measured"))
        infos.append(json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {})

    last = runs[-1]
    wall_stats = summarize_numbers([r.wall_time_s for r in runs])
    benchmark: dict[str, Any] = {"repeats": repeats, "warmup": warmup}

    info_times = [t for t in (_number(i.get("time")) for i in infos) if t is not None]
    if info_times:
        benchmark["info_time_s"] = summarize_numbers(info_times)
    benchmark["wall_time_s"] = wall_stats

    def collect(attr: str) -> list[Any]:
        return [getattr(r, attr) for r in runs if getattr(r, attr) is not None]

    rss = collect("max_rss_kb")
    cpu_user = collect("cpu_user_s")
    cpu_sys = collect("cpu_sys_s")
    cpu_eff = collect("cpu_efficiency")
    for key, samples in (
        ("max_rss_kb", rss),
        ("cpu_user_s", cpu_user),
        ("cpu_sys_s", cpu_sys),
        ("cpu_efficiency", cpu_eff),
    ):
        if samples:
            benchmark[key] = summarize_numbers(samples)

    warnings: list[str] = []
    if nondeterministic:
        names = sorted(nondeterministic)
        warnings.append(
            f"Stage '{request.stage_name}' produced different output bytes across its {repeats + warmup} runs "
            f"(outputs: {', '.join(names)}); cached results and content deduplication are unreliable for it"
        )

    execution: dict[str, Any] = {
        "info_time_s": benchmark["info_time_s"]["median"] if info_times else None,
        "wall_time_s": wall_stats["median"],
        "max_rss_kb": int(max(rss)) if rss else None,
        "cpu_user_s": statistics.median(cpu_user) if cpu_user else None,
        "cpu_sys_s": statistics.median(cpu_sys) if cpu_sys else None,
        "cpu_efficiency": statistics.median(cpu_eff) if cpu_eff else None,
        "minor_page_faults": last.minor_page_faults,
        "major_page_faults": last.major_page_faults,
        "voluntary_ctx_switches": last.voluntary_ctx_switches,
        "involuntary_ctx_switches": last.involuntary_ctx_switches,
        "exit_code": last.exit_code,
        "hostname": last.hostname,
        "environment": last.environment,
        "command": list(request.command),
        "benchmark": benchmark,
    }
    if check_determinism:
        execution["nondeterministic_outputs"] = sorted(nondeterministic)
    if warnings:
        execution["warnings"] = list(warnings)

    return Measurement(info=infos[-1], execution=execution, warnings=warnings)
