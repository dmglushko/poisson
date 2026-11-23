"""Per-stage traces, per-branch results, and the aggregated `ExperimentReport` (table/JSON/CSV)."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class StageTrace:
    """Record of one stage invocation inside a branch.

    - `cached`: served from the persistent cache (`HIT`).
    - `reused`: served from an execution earlier in the same `run_experiment` call (`REUSED`),
      e.g. a shared prefix stage under `refresh`/`off`/`read_only` that already ran on a previous branch.
    - `time_s`: the stage's self-reported `info["time"]` (median over repeats) — the headline timing.
    - `wall_time_s`: median process wall time, including process startup and file I/O.
    """

    stage: str
    cached: bool
    request_key: str | None
    content_ids: dict[str, str]
    settings: dict[str, Any]
    wall_time_s: float
    max_rss_kb: int | None
    info: dict[str, Any]
    execution: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    reused: bool = False
    time_s: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        """True when this call launched (or, in a dry run, would launch) the stage binary."""
        return not self.cached and not self.reused

    @property
    def status(self) -> str:
        if self.cached:
            return "HIT"
        if self.reused:
            return "REUSED"
        return "WOULD_RUN" if self.dry_run else "MISS"


@dataclass(frozen=True)
class BranchResult:
    """Outcome of one complete branch through the user's pipeline."""

    index: int
    choices: dict[str, Any]
    choice_labels: dict[str, str]
    stages: list[StageTrace]
    result: Any = None
    error: str | None = None
    wall_time_s: float = 0.0


def _fmt_opt_float(val: Any) -> str:
    return f"{float(val):.6f}" if isinstance(val, (int, float)) and not isinstance(val, bool) else ""


def _fmt_opt_int(val: Any) -> str:
    return str(int(val)) if isinstance(val, (int, float)) and not isinstance(val, bool) else ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class ExperimentReport:
    """Aggregated results across all branches explored by `run_experiment`."""

    branches: list[BranchResult]
    total_wall_time_s: float
    dry_run: bool = False
    shard: tuple[int, int] | None = None

    @property
    def cache_hits(self) -> int:
        return sum(1 for b in self.branches for s in b.stages if s.cached)

    @property
    def reused(self) -> int:
        return sum(1 for b in self.branches for s in b.stages if s.reused)

    @property
    def cache_misses(self) -> int:
        """Stage calls that executed (or would execute, in a dry run) a binary."""
        return sum(1 for b in self.branches for s in b.stages if s.executed)

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        for b in self.branches:
            for s in b.stages:
                for w in s.warnings:
                    if w not in out:
                        out.append(w)
        return out

    def _choice_keys(self) -> list[str]:
        keys: list[str] = []
        for b in self.branches:
            for k in b.choice_labels:
                if k not in keys:
                    keys.append(k)
        return keys

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dry_run": self.dry_run,
            "total_wall_time_s": self.total_wall_time_s,
            "branch_count": len(self.branches),
            "cache_hits": self.cache_hits,
            "reused": self.reused,
            "cache_misses": self.cache_misses,
            "warnings": self.warnings,
        }
        if self.shard is not None:
            payload["shard"] = {"index": self.shard[0], "total": self.shard[1]}
        payload["branches"] = [
            {
                "index": b.index,
                "choices": b.choice_labels,
                "wall_time_s": b.wall_time_s,
                "error": b.error,
                "stages": [
                    {
                        "stage": s.stage,
                        "status": s.status,
                        "cached": s.cached,
                        "reused": s.reused,
                        "dry_run": s.dry_run,
                        "request_key": s.request_key,
                        "content_ids": s.content_ids,
                        "settings": s.settings,
                        "time_s": s.time_s,
                        "wall_time_s": s.wall_time_s,
                        "max_rss_kb": s.max_rss_kb,
                        "warnings": s.warnings,
                        "info": s.info,
                        "execution": s.execution,
                    }
                    for s in b.stages
                ],
            }
            for b in self.branches
        ]
        return payload

    def save_json(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def save_csv(self, path: str | os.PathLike[str], *, layout: Literal["wide", "long"] = "wide") -> None:
        """Export results as CSV.

        - `layout="wide"` (default): one row per branch (repeated stages in a loop are disambiguated
          as `stage#0`, `stage#1`, ...).
        - `layout="long"`: tidy table with one row per `(branch, step_index, stage)` including full
          repeat statistics (`min`/`median`/`mean`/`max`/`std`) and OS metrics. Ideal for pandas.

        The stage's self-reported `info["time"]` columns come first; process wall-time columns
        (which include process startup and HDF5 I/O) follow.
        """
        if layout not in ("wide", "long"):
            raise ValueError(f"Unsupported CSV layout {layout!r}; expected 'wide' or 'long'")
        if layout == "long":
            self._save_csv_long(Path(path))
        else:
            self._save_csv_wide(Path(path))

    def _save_csv_long(self, path: Path) -> None:
        choice_keys = self._choice_keys()
        fixed_tail = [
            "step_index",
            "stage",
            "status",
            "repeats",
            "warmup",
            "info_time_s",
            "info_time_min_s",
            "info_time_median_s",
            "info_time_mean_s",
            "info_time_max_s",
            "info_time_std_s",
            "wall_s",
            "wall_min_s",
            "wall_median_s",
            "wall_mean_s",
            "wall_max_s",
            "wall_std_s",
            "max_rss_kb",
            "cpu_user_s",
            "cpu_sys_s",
            "cpu_efficiency",
            "major_page_faults",
            "hostname",
            "cpu_model",
            "omp_num_threads",
            "slurm_job_id",
            "request_key",
            "warnings",
            "error",
        ]
        reserved = {"branch", *fixed_tail}
        choice_col_map: dict[str, str] = {}
        used_cols = set(reserved)
        for k in choice_keys:
            col = k
            if col in used_cols:
                col = f"choice_{k}"
                suffix = 2
                while col in used_cols:
                    col = f"choice_{k}_{suffix}"
                    suffix += 1
            choice_col_map[k] = col
            used_cols.add(col)

        fieldnames = ["branch", *(choice_col_map[k] for k in choice_keys), *fixed_tail]
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for b in self.branches:
                for step_idx, st in enumerate(b.stages):
                    execution = _as_dict(st.execution)
                    bench = _as_dict(execution.get("benchmark"))
                    w_stats = _as_dict(bench.get("wall_time_s"))
                    i_stats = _as_dict(bench.get("info_time_s"))
                    env_meta = _as_dict(execution.get("environment"))
                    threading_meta = _as_dict(env_meta.get("threading"))
                    slurm_meta = _as_dict(env_meta.get("slurm"))
                    row: dict[str, Any] = {
                        "branch": b.index,
                        **{choice_col_map[k]: b.choice_labels.get(k, "") for k in choice_keys},
                        "step_index": step_idx,
                        "stage": st.stage,
                        "status": st.status,
                        "repeats": _fmt_opt_int(bench.get("repeats", 1 if not st.dry_run else None)),
                        "warmup": _fmt_opt_int(bench.get("warmup", 0 if not st.dry_run else None)),
                        "info_time_s": _fmt_opt_float(st.time_s),
                        "info_time_min_s": _fmt_opt_float(i_stats.get("min")),
                        "info_time_median_s": _fmt_opt_float(i_stats.get("median")),
                        "info_time_mean_s": _fmt_opt_float(i_stats.get("mean")),
                        "info_time_max_s": _fmt_opt_float(i_stats.get("max")),
                        "info_time_std_s": _fmt_opt_float(i_stats.get("std")),
                        "wall_s": _fmt_opt_float(st.wall_time_s),
                        "wall_min_s": _fmt_opt_float(w_stats.get("min")),
                        "wall_median_s": _fmt_opt_float(w_stats.get("median")),
                        "wall_mean_s": _fmt_opt_float(w_stats.get("mean")),
                        "wall_max_s": _fmt_opt_float(w_stats.get("max")),
                        "wall_std_s": _fmt_opt_float(w_stats.get("std")),
                        "max_rss_kb": _fmt_opt_int(st.max_rss_kb),
                        "cpu_user_s": _fmt_opt_float(execution.get("cpu_user_s")),
                        "cpu_sys_s": _fmt_opt_float(execution.get("cpu_sys_s")),
                        "cpu_efficiency": _fmt_opt_float(execution.get("cpu_efficiency")),
                        "major_page_faults": _fmt_opt_int(execution.get("major_page_faults")),
                        "hostname": str(execution.get("hostname", "") or ""),
                        "cpu_model": str(env_meta.get("cpu_model", "") or ""),
                        "omp_num_threads": str(threading_meta.get("OMP_NUM_THREADS", "") or ""),
                        "slurm_job_id": str(slurm_meta.get("SLURM_JOB_ID", "") or ""),
                        "request_key": st.request_key or "",
                        "warnings": "; ".join(st.warnings),
                        "error": b.error or "",
                    }
                    writer.writerow(row)

    def _save_csv_wide(self, path: Path) -> None:
        choice_keys = self._choice_keys()
        # Disambiguate repeated stage names within any branch as stage#0, stage#1, ...
        max_occurrences: dict[str, int] = {}
        for b in self.branches:
            counts: dict[str, int] = {}
            for s in b.stages:
                counts[s.stage] = counts.get(s.stage, 0) + 1
            for name, c in counts.items():
                max_occurrences[name] = max(max_occurrences.get(name, 0), c)

        stage_cols: list[str] = []
        branch_stage_maps: list[dict[str, StageTrace]] = []
        for b in self.branches:
            occ: dict[str, int] = {}
            smap: dict[str, StageTrace] = {}
            for s in b.stages:
                idx = occ.get(s.stage, 0)
                occ[s.stage] = idx + 1
                col = f"{s.stage}#{idx}" if max_occurrences[s.stage] > 1 else s.stage
                smap[col] = s
                if col not in stage_cols:
                    stage_cols.append(col)
            branch_stage_maps.append(smap)

        fieldnames = ["branch", *choice_keys, "branch_wall_s", "error"]
        for sn in stage_cols:
            fieldnames.extend(
                [f"{sn}_status", f"{sn}_info_time_s", f"{sn}_wall_s", f"{sn}_rss_kb", f"{sn}_cpu_eff"]
            )

        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for b, smap in zip(self.branches, branch_stage_maps):
                row: dict[str, Any] = {
                    "branch": b.index,
                    "branch_wall_s": f"{b.wall_time_s:.6f}",
                    "error": b.error or "",
                }
                for k in choice_keys:
                    row[k] = b.choice_labels.get(k, "")
                for sn in stage_cols:
                    st = smap.get(sn)
                    if st is not None:
                        row[f"{sn}_status"] = st.status
                        row[f"{sn}_info_time_s"] = _fmt_opt_float(st.time_s)
                        row[f"{sn}_wall_s"] = _fmt_opt_float(st.wall_time_s)
                        row[f"{sn}_rss_kb"] = _fmt_opt_int(st.max_rss_kb)
                        row[f"{sn}_cpu_eff"] = _fmt_opt_float(_as_dict(st.execution).get("cpu_efficiency"))
                writer.writerow(row)

    @staticmethod
    def _stage_cell(s: StageTrace) -> str:
        if s.dry_run and s.executed:
            return f"{s.stage}(WOULD_RUN)"
        if s.time_s is not None:
            timing = f"{s.time_s:.3f}s"
        else:
            timing = f"{s.wall_time_s:.3f}s wall"
        flag = " !" if s.warnings else ""
        return f"{s.stage}({s.status} {timing}{flag})"

    def format_table(self) -> str:
        """Render a compact Markdown table summarizing all branches and cache hits/misses.

        Stage timings show the stage's own `info["time"]`; `wall` marks a fallback to process
        wall time for stages that do not report one. `!` marks stages with warnings.
        """
        choice_keys = self._choice_keys()
        headers = ["#", *choice_keys, "stages", "wall_s", "status"]
        rows: list[list[str]] = []
        for b in self.branches:
            stage_summary = ", ".join(self._stage_cell(s) for s in b.stages)
            if b.error is not None:
                status = f"ERR: {b.error}"
            else:
                status = "DRY_RUN" if self.dry_run else "OK"
            rows.append(
                [
                    str(b.index),
                    *[b.choice_labels.get(k, "-") for k in choice_keys],
                    stage_summary or "-",
                    f"{b.wall_time_s:.3f}",
                    status,
                ]
            )

        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
        lines = [
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
            "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
        ]
        for r in rows:
            lines.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |")
        miss_word = "would run" if self.dry_run else "cache misses"
        shard_suffix = f" [shard {self.shard[0]}/{self.shard[1]}]" if self.shard is not None else ""
        reused_part = f", {self.reused} reused in this run" if self.reused else ""
        lines.append(
            f"\nTotal: {len(self.branches)} branches{shard_suffix}, {self.cache_hits} cache hits{reused_part}, "
            f"{self.cache_misses} {miss_word}, {self.total_wall_time_s:.3f}s"
        )
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return "\n".join(lines)
