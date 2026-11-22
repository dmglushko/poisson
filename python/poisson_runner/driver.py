"""Loop-free Compare(...) replay driver, repeat statistics, dry-run planning, and reporting."""

from __future__ import annotations

import contextvars
import csv
import inspect
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar

from .cache import Artifact, Cache, CacheLease, CacheMode, file_signature, normalize_cache_mode
from .executor import ExecutionRequest, ExecutionResult, Executor, LocalExecutor
from .hashing import canonical_json
from .stages import Stage, StageCall, make_adapt_stage, make_import_free_charges_stage, make_import_mesh_stage

T = TypeVar("T")

_ACTIVE_RUN: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("poisson_active_run", default=None)


def _summarize_numbers(samples: Sequence[float | int]) -> dict[str, Any]:
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


@dataclass(frozen=True)
class _ChoiceStep:
    """Recorded choice-site expectation used to detect non-deterministic replays."""

    name: str
    index: int
    label: str
    num_options: int


class Compare(Generic[T]):
    """Declarative choice object for branching without writing explicit loops.

    Examples:
        Compare(["P1", "P2"], name="assembly")
        Compare({"coarse": ("coarse.msh", 1e-4), "fine": ("fine.msh", 1e-6)}, name="case")
    """

    def __init__(self, options: Sequence[T] | Mapping[str, T], *, name: str | None = None) -> None:
        self.name = name
        if isinstance(options, Mapping):
            self.labels = [str(k) for k in options.keys()]
            self.values: list[T] = list(options.values())
        else:
            if isinstance(options, (str, bytes)):
                raise TypeError("Compare options must be a sequence of choices, not a single string/bytes")
            self.values = list(options)
            self.labels = [str(v) for v in self.values]
        if not self.values:
            raise ValueError("Compare requires at least one option")


@dataclass(frozen=True)
class StageTrace:
    """Record of one stage invocation inside a branch."""

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
    def cache_misses(self) -> int:
        return sum(1 for b in self.branches for s in b.stages if not s.cached)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dry_run": self.dry_run,
            "total_wall_time_s": self.total_wall_time_s,
            "branch_count": len(self.branches),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
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
                        "cached": s.cached,
                        "dry_run": s.dry_run,
                        "request_key": s.request_key,
                        "content_ids": s.content_ids,
                        "settings": s.settings,
                        "wall_time_s": s.wall_time_s,
                        "max_rss_kb": s.max_rss_kb,
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

    @staticmethod
    def _stage_status_label(st: StageTrace) -> str:
        if st.cached:
            return "HIT"
        return "WOULD_RUN" if st.dry_run else "MISS"

    @staticmethod
    def _fmt_opt_float(val: Any) -> str:
        return f"{float(val):.6f}" if isinstance(val, (int, float)) and not isinstance(val, bool) else ""

    @staticmethod
    def _fmt_opt_int(val: Any) -> str:
        return str(int(val)) if isinstance(val, (int, float)) and not isinstance(val, bool) else ""

    def save_csv(self, path: str | os.PathLike[str], *, layout: Literal["wide", "long"] = "wide") -> None:
        """Export results as CSV.

        - `layout="wide"` (default): one row per branch (repeated stages in a loop are disambiguated
          as `stage#0`, `stage#1`, ...).
        - `layout="long"`: tidy table with one row per `(branch, step_index, stage)` including full
          repeat statistics (`min`/`median`/`mean`/`max`/`std`) and OS metrics (`cpu_user_s`,
          `cpu_efficiency`, `major_page_faults`, `max_rss_kb`). Ideal for pandas/matplotlib.
        """
        if layout not in ("wide", "long"):
            raise ValueError(f"Unsupported CSV layout {layout!r}; expected 'wide' or 'long'")

        choice_keys: list[str] = []
        for b in self.branches:
            for k in b.choice_labels:
                if k not in choice_keys:
                    choice_keys.append(k)

        if layout == "long":
            fixed_tail = [
                "step_index",
                "stage",
                "status",
                "repeats",
                "warmup",
                "wall_s",
                "wall_min_s",
                "wall_median_s",
                "wall_mean_s",
                "wall_max_s",
                "wall_std_s",
                "info_time_s",
                "info_time_min_s",
                "info_time_median_s",
                "info_time_mean_s",
                "info_time_max_s",
                "info_time_std_s",
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
            with Path(path).open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for b in self.branches:
                    for step_idx, st in enumerate(b.stages):
                        bench = st.execution.get("benchmark") if isinstance(st.execution, dict) else None
                        bench = bench if isinstance(bench, dict) else {}
                        w_stats = bench.get("wall_time_s") if isinstance(bench.get("wall_time_s"), dict) else {}
                        i_stats = bench.get("info_time_s") if isinstance(bench.get("info_time_s"), dict) else {}
                        env_meta = st.execution.get("environment") if isinstance(st.execution, dict) else None
                        env_meta = env_meta if isinstance(env_meta, dict) else {}
                        threading_meta = env_meta.get("threading") if isinstance(env_meta.get("threading"), dict) else {}
                        slurm_meta = env_meta.get("slurm") if isinstance(env_meta.get("slurm"), dict) else {}
                        row: dict[str, Any] = {
                            "branch": b.index,
                            **{choice_col_map[k]: b.choice_labels.get(k, "") for k in choice_keys},
                            "step_index": step_idx,
                            "stage": st.stage,
                            "status": self._stage_status_label(st),
                            "repeats": self._fmt_opt_int(bench.get("repeats", 1 if not st.dry_run else None)),
                            "warmup": self._fmt_opt_int(bench.get("warmup", 0 if not st.dry_run else None)),
                            "wall_s": self._fmt_opt_float(st.wall_time_s),
                            "wall_min_s": self._fmt_opt_float(w_stats.get("min")),
                            "wall_median_s": self._fmt_opt_float(w_stats.get("median")),
                            "wall_mean_s": self._fmt_opt_float(w_stats.get("mean")),
                            "wall_max_s": self._fmt_opt_float(w_stats.get("max")),
                            "wall_std_s": self._fmt_opt_float(w_stats.get("std")),
                            "info_time_s": self._fmt_opt_float(st.info.get("time")),
                            "info_time_min_s": self._fmt_opt_float(i_stats.get("min")),
                            "info_time_median_s": self._fmt_opt_float(i_stats.get("median")),
                            "info_time_mean_s": self._fmt_opt_float(i_stats.get("mean")),
                            "info_time_max_s": self._fmt_opt_float(i_stats.get("max")),
                            "info_time_std_s": self._fmt_opt_float(i_stats.get("std")),
                            "max_rss_kb": self._fmt_opt_int(st.max_rss_kb),
                            "cpu_user_s": self._fmt_opt_float(st.execution.get("cpu_user_s")),
                            "cpu_sys_s": self._fmt_opt_float(st.execution.get("cpu_sys_s")),
                            "cpu_efficiency": self._fmt_opt_float(st.execution.get("cpu_efficiency")),
                            "major_page_faults": self._fmt_opt_int(st.execution.get("major_page_faults")),
                            "hostname": str(st.execution.get("hostname", "") or ""),
                            "cpu_model": str(env_meta.get("cpu_model", "") or ""),
                            "omp_num_threads": str(threading_meta.get("OMP_NUM_THREADS", "") or ""),
                            "slurm_job_id": str(slurm_meta.get("SLURM_JOB_ID", "") or ""),
                            "request_key": st.request_key or "",
                            "error": b.error or "",
                        }
                        writer.writerow(row)
            return

        # Wide layout: disambiguate repeated stage names within any branch as stage#0, stage#1, ...
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
                [f"{sn}_status", f"{sn}_wall_s", f"{sn}_info_time_s", f"{sn}_rss_kb", f"{sn}_cpu_eff"]
            )

        with Path(path).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for b, smap in zip(self.branches, branch_stage_maps):
                row = {
                    "branch": b.index,
                    "branch_wall_s": f"{b.wall_time_s:.6f}",
                    "error": b.error or "",
                }
                for k in choice_keys:
                    row[k] = b.choice_labels.get(k, "")
                for sn in stage_cols:
                    st = smap.get(sn)
                    if st is not None:
                        row[f"{sn}_status"] = self._stage_status_label(st)
                        row[f"{sn}_wall_s"] = self._fmt_opt_float(st.wall_time_s)
                        row[f"{sn}_info_time_s"] = self._fmt_opt_float(st.info.get("time"))
                        row[f"{sn}_rss_kb"] = self._fmt_opt_int(st.max_rss_kb)
                        row[f"{sn}_cpu_eff"] = self._fmt_opt_float(st.execution.get("cpu_efficiency"))
                writer.writerow(row)

    def format_table(self) -> str:
        """Render a compact Markdown table summarizing all branches and cache hits/misses."""
        choice_keys: list[str] = []
        for b in self.branches:
            for k in b.choice_labels:
                if k not in choice_keys:
                    choice_keys.append(k)

        headers = ["#", *choice_keys, "stages", "wall_s", "status"]
        rows: list[list[str]] = []
        for b in self.branches:
            parts: list[str] = []
            for s in b.stages:
                label = self._stage_status_label(s)
                if s.dry_run and not s.cached:
                    parts.append(f"{s.stage}(WOULD_RUN)")
                else:
                    parts.append(f"{s.stage}({label} {s.wall_time_s:.3f}s)")
            stage_summary = ", ".join(parts)
            rows.append(
                [
                    str(b.index),
                    *[b.choice_labels.get(k, "-") for k in choice_keys],
                    stage_summary or "-",
                    f"{b.wall_time_s:.3f}",
                    "DRY_RUN" if self.dry_run and b.error is None else ("OK" if b.error is None else f"ERR: {b.error}"),
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
        lines.append(
            f"\nTotal: {len(self.branches)} branches{shard_suffix}, {self.cache_hits} cache hits, "
            f"{self.cache_misses} {miss_word}, {self.total_wall_time_s:.3f}s"
        )
        return "\n".join(lines)


class RunContext:
    """Execution context for a single branch inside `run_experiment`."""

    def __init__(
        self,
        *,
        cache: Cache,
        executor: Executor,
        build_dir: Path | str | None,
        path: tuple[_ChoiceStep, ...],
        todo_stack: list[tuple[_ChoiceStep, ...]],
        lease: CacheLease,
        cache_mode: CacheMode | None = None,
        stage_cache_modes: Mapping[str, CacheMode] | None = None,
        default_repeats: int = 1,
        default_warmup: int = 0,
        dry_run: bool = False,
    ) -> None:
        self.cache = cache
        self.executor = executor
        self.build_dir = build_dir
        self._path = path
        self._prefix: list[_ChoiceStep] = []
        self._todo_stack = todo_stack
        self._step = 0
        self._lease = lease
        self._cache_mode = cache_mode
        self._stage_cache_modes: dict[str, CacheMode] = dict(stage_cache_modes) if stage_cache_modes else {}
        self._default_repeats = default_repeats
        self._default_warmup = default_warmup
        self.dry_run = dry_run
        self.choices: dict[str, Any] = {}
        self.choice_labels: dict[str, str] = {}
        self.trace: list[StageTrace] = []
        self._import_mesh_stage: Stage | None = None
        self._import_free_charges_stage: Stage | None = None
        self._adapt_stages: dict[tuple[bool, bool], Stage] = {}

    def _resolve_cache_mode(
        self,
        stage: Stage,
        cache: CacheMode | str | bool | None,
        use_cache: bool | None,
    ) -> CacheMode:
        if cache is not None and use_cache is not None:
            raise ValueError("Pass either 'cache' or 'use_cache', not both")
        if cache is not None:
            return normalize_cache_mode(cache)
        if use_cache is not None:
            return normalize_cache_mode(use_cache)
        if stage.name in self._stage_cache_modes:
            return self._stage_cache_modes[stage.name]
        if self._cache_mode is not None:
            return self._cache_mode
        return normalize_cache_mode(stage.default_cache)

    def compare(
        self,
        name_or_options: str | Sequence[Any] | Mapping[str, Any] | Compare[Any],
        options: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any:
        """Select one option on this branch and automatically schedule all remaining options."""
        if isinstance(name_or_options, Compare):
            spec = name_or_options
        elif isinstance(name_or_options, str) and options is not None:
            spec = Compare(options, name=name_or_options)
        elif options is None:
            spec = Compare(name_or_options)  # type: ignore[arg-type]
        else:
            raise TypeError("Expected compare(Compare(...)), compare(options), or compare(name, options)")

        choice_name = spec.name if spec.name is not None else f"choice_{self._step}"
        if choice_name in self.choices:
            raise ValueError(
                f"Duplicate compare() name '{choice_name}' in the same branch; use distinct names for each choice site"
            )

        if self._step < len(self._path):
            expected = self._path[self._step]
            if (
                choice_name != expected.name
                or len(spec.values) != expected.num_options
                or expected.index >= len(spec.values)
                or spec.labels[expected.index] != expected.label
            ):
                raise RuntimeError(
                    f"Non-deterministic compare() replay at step {self._step}: "
                    f"expected ({expected.name!r}, idx={expected.index}, label={expected.label!r}, n={expected.num_options}), "
                    f"got ({choice_name!r}, n={len(spec.values)}, labels={spec.labels!r})"
                )
            idx = expected.index
            chosen_step = expected
        else:
            idx = 0
            chosen_step = _ChoiceStep(choice_name, 0, spec.labels[0], len(spec.values))
            # Push remaining options in reverse order so DFS pops them in natural user order.
            for other_idx in range(len(spec.values) - 1, 0, -1):
                other_step = _ChoiceStep(choice_name, other_idx, spec.labels[other_idx], len(spec.values))
                self._todo_stack.append(tuple(self._prefix) + (other_step,))

        self._prefix.append(chosen_step)
        self._step += 1
        value = spec.values[idx]
        self.choices[choice_name] = value
        self.choice_labels[choice_name] = spec.labels[idx]
        return value

    def raw_file(self, path: str | os.PathLike[str], *, name: str = "file") -> Artifact:
        """Wrap an external file on disk as a content-hashed Artifact."""
        return self.cache.raw_file(path, name=name)

    def _pin_if_cached_object(self, art: Artifact) -> None:
        if art.path.parent == self.cache.objects_dir:
            self._lease.pin([art.path.name])

    @staticmethod
    def _is_dry_run_placeholder(art: Artifact) -> bool:
        return art.content_id.startswith("dryrun:")

    @staticmethod
    def _hit_satisfies_repeats(hit: dict[str, Artifact], repeats: int, warmup: int) -> bool:
        if repeats <= 1 and warmup <= 0:
            return True
        first = next(iter(hit.values()))
        bench = first.execution.get("benchmark") if isinstance(first.execution, dict) else None
        if not isinstance(bench, dict):
            return False
        return int(bench.get("repeats", 1)) >= repeats and int(bench.get("warmup", 0)) >= warmup

    def run_stage(
        self,
        stage: Stage,
        *,
        inputs: Mapping[str, Artifact | str | os.PathLike[str] | Sequence[Artifact | str | os.PathLike[str]] | None] = (),  # type: ignore[assignment]
        settings: Mapping[str, Any] | None = None,
        cache: CacheMode | str | bool | None = None,
        use_cache: bool | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
        env: Mapping[str, str] | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> Artifact | dict[str, Artifact]:
        """Run a stage with BLAKE3 content-addressed caching, 4-mode control, repeats, and dry-run."""
        mode = self._resolve_cache_mode(stage, cache, use_cache)
        n_repeats = self._default_repeats if repeats is None else repeats
        n_warmup = self._default_warmup if warmup is None else warmup
        if n_repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {n_repeats}")
        if n_warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {n_warmup}")

        resolved_inputs: dict[str, Artifact | list[Artifact]] = {}
        inputs_spec: dict[str, Any] = {}
        watched_paths: list[Path] = []
        has_placeholder_input = False

        for key, raw_val in dict(inputs).items():
            if raw_val is None:
                continue
            if isinstance(raw_val, Artifact):
                resolved_inputs[key] = raw_val
                inputs_spec[key] = raw_val.content_id
                if self._is_dry_run_placeholder(raw_val):
                    has_placeholder_input = True
                else:
                    self._pin_if_cached_object(raw_val)
                    watched_paths.append(raw_val.path)
            elif isinstance(raw_val, (str, os.PathLike)):
                art = self.raw_file(raw_val, name=key)
                resolved_inputs[key] = art
                inputs_spec[key] = art.content_id
                watched_paths.append(art.path)
            elif isinstance(raw_val, Sequence) and not isinstance(raw_val, (bytes, bytearray)):
                arts: list[Artifact] = []
                for item in raw_val:
                    if isinstance(item, Artifact):
                        arts.append(item)
                        if self._is_dry_run_placeholder(item):
                            has_placeholder_input = True
                        else:
                            self._pin_if_cached_object(item)
                            watched_paths.append(item.path)
                    else:
                        art = self.raw_file(item, name=key)
                        arts.append(art)
                        watched_paths.append(art.path)
                resolved_inputs[key] = arts
                inputs_spec[key] = [a.content_id for a in arts]
            else:
                raise TypeError(f"Unsupported input type for '{key}': {type(raw_val).__name__}")

        tracked_code_files = stage.tracked_code_files()
        watched_paths.extend(tracked_code_files)
        pre_signatures = {p: file_signature(p) for p in watched_paths}

        settings_dict = dict(settings) if settings is not None else {}
        env_dict: dict[str, str] | None = None
        if env is not None:
            env_dict = {}
            for k, v in env.items():
                if not isinstance(k, str):
                    raise TypeError(f"Stage env keys must be str, got {type(k).__name__}")
                env_dict[k] = str(v)
        code_hash = stage.compute_code_hash(self.cache)
        request_key = Cache.compute_request_key(
            stage.name,
            code_hash,
            inputs_spec,
            settings_dict,
            output_specs=stage.output_specs,
            env=env_dict,
        )

        if mode.reads and not has_placeholder_input:
            hit = self.cache.lookup(request_key, expected_outputs=stage.output_specs, lease=self._lease)
            if hit is not None and self._hit_satisfies_repeats(hit, n_repeats, n_warmup):
                first = next(iter(hit.values()))
                self.trace.append(
                    StageTrace(
                        stage=stage.name,
                        cached=True,
                        request_key=request_key,
                        content_ids={k: a.content_id for k, a in hit.items()},
                        settings=settings_dict,
                        wall_time_s=float(first.execution.get("wall_time_s", 0.0)),
                        max_rss_kb=first.execution.get("max_rss_kb"),
                        info=first.info,
                        execution=first.execution,
                        dry_run=self.dry_run,
                    )
                )
                return hit["out"] if len(hit) == 1 and "out" in hit else hit

        # Dry-run mode: do NOT execute binaries; return synthetic placeholder artifacts.
        if self.dry_run:
            placeholders: dict[str, Artifact] = {}
            for out_name, ext in stage.output_specs.items():
                cid = f"dryrun:{request_key}:{out_name}"
                placeholders[out_name] = Artifact(
                    path=Path(f"<dry-run:{stage.name}:{out_name}{ext}>"),
                    content_id=cid,
                    name=out_name,
                    stage=stage.name,
                    request_key=request_key,
                    cached=False,
                    size_bytes=0,
                )
            self.trace.append(
                StageTrace(
                    stage=stage.name,
                    cached=False,
                    request_key=request_key,
                    content_ids={k: a.content_id for k, a in placeholders.items()},
                    settings=settings_dict,
                    wall_time_s=0.0,
                    max_rss_kb=None,
                    info={},
                    execution={},
                    dry_run=True,
                )
            )
            return placeholders["out"] if len(placeholders) == 1 and "out" in placeholders else placeholders

        # Cache miss (or reads disabled): execute warmup + measured repeats.
        path_allocator = self.cache.make_temp_path if mode.writes else self.cache.make_ephemeral_path
        temp_outputs: dict[str, tuple[Path, str]] = {
            out_name: (path_allocator(ext), ext) for out_name, ext in stage.output_specs.items()
        }
        info_tmp = self.cache.make_temp_path(".info.json")
        config_tmp: Path | None = None
        if stage.uses_config_file:
            config_tmp = self.cache.make_temp_path(".config.json")
            config_tmp.write_text(canonical_json(settings_dict) + "\n", encoding="utf-8")

        try:
            call = StageCall(
                executable=stage.executable,
                inputs=resolved_inputs,
                outputs={k: p for k, (p, _) in temp_outputs.items()},
                info_path=info_tmp,
                settings=settings_dict,
                config_path=config_tmp,
            )
            cmd = stage.build_command(call)
            req = ExecutionRequest(
                stage_name=stage.name,
                command=cmd,
                env=env_dict,
                resources=dict(resources) if resources is not None else {},
            )

            # Warmup iterations (results and timings discarded).
            for _ in range(n_warmup):
                warm_res = self.executor.run(req)
                if warm_res.exit_code != 0:
                    detail = warm_res.stderr.strip() or warm_res.stdout.strip()
                    raise RuntimeError(
                        f"Stage '{stage.name}' failed during warmup with exit code {warm_res.exit_code}: {detail}"
                    )

            measured_execs: list[ExecutionResult] = []
            measured_infos: list[dict[str, Any]] = []
            for _ in range(n_repeats):
                info_tmp.unlink(missing_ok=True)
                exec_result = self.executor.run(req)
                if exec_result.exit_code != 0:
                    detail = exec_result.stderr.strip() or exec_result.stdout.strip()
                    raise RuntimeError(
                        f"Stage '{stage.name}' failed with exit code {exec_result.exit_code}: {detail}"
                    )
                measured_execs.append(exec_result)
                iter_info: dict[str, Any] = {}
                if info_tmp.is_file():
                    iter_info = json.loads(info_tmp.read_text(encoding="utf-8"))
                measured_infos.append(iter_info)

            # Verify no input or stage binary mutated mid-run before publishing under request_key.
            if mode.writes:
                for p, before_sig in pre_signatures.items():
                    if not p.is_file() or file_signature(p) != before_sig:
                        raise RuntimeError(
                            f"File '{p}' changed while stage '{stage.name}' was running; "
                            "refusing to publish result under stale cache key"
                        )

            last_exec = measured_execs[-1]
            info = measured_infos[-1]

            wall_samples = [r.wall_time_s for r in measured_execs]
            wall_stats = _summarize_numbers(wall_samples)

            benchmark_meta: dict[str, Any] = {
                "repeats": n_repeats,
                "warmup": n_warmup,
                "wall_time_s": wall_stats,
            }

            info_times = [
                float(inf["time"])
                for inf in measured_infos
                if isinstance(inf.get("time"), (int, float)) and not isinstance(inf.get("time"), bool)
            ]
            if info_times:
                benchmark_meta["info_time_s"] = _summarize_numbers(info_times)

            rss_samples = [r.max_rss_kb for r in measured_execs if r.max_rss_kb is not None]
            if rss_samples:
                benchmark_meta["max_rss_kb"] = _summarize_numbers(rss_samples)

            cpu_user_samples = [r.cpu_user_s for r in measured_execs if r.cpu_user_s is not None]
            if cpu_user_samples:
                benchmark_meta["cpu_user_s"] = _summarize_numbers(cpu_user_samples)

            cpu_sys_samples = [r.cpu_sys_s for r in measured_execs if r.cpu_sys_s is not None]
            if cpu_sys_samples:
                benchmark_meta["cpu_sys_s"] = _summarize_numbers(cpu_sys_samples)

            cpu_eff_samples = [r.cpu_efficiency for r in measured_execs if r.cpu_efficiency is not None]
            if cpu_eff_samples:
                benchmark_meta["cpu_efficiency"] = _summarize_numbers(cpu_eff_samples)

            execution_meta: dict[str, Any] = {
                "wall_time_s": wall_stats["median"],
                "max_rss_kb": int(max(rss_samples)) if rss_samples else None,
                "cpu_user_s": statistics.median(cpu_user_samples) if cpu_user_samples else None,
                "cpu_sys_s": statistics.median(cpu_sys_samples) if cpu_sys_samples else None,
                "cpu_efficiency": statistics.median(cpu_eff_samples) if cpu_eff_samples else None,
                "minor_page_faults": last_exec.minor_page_faults,
                "major_page_faults": last_exec.major_page_faults,
                "voluntary_ctx_switches": last_exec.voluntary_ctx_switches,
                "involuntary_ctx_switches": last_exec.involuntary_ctx_switches,
                "exit_code": last_exec.exit_code,
                "hostname": last_exec.hostname,
                "environment": last_exec.environment,
                "command": cmd,
                "benchmark": benchmark_meta,
            }

            if mode.writes:
                artifacts = self.cache.store(
                    request_key=request_key,
                    stage_name=stage.name,
                    code_hash=code_hash,
                    inputs_spec=inputs_spec,
                    settings=settings_dict,
                    temp_outputs=temp_outputs,
                    info=info,
                    execution=execution_meta,
                    lease=self._lease,
                )
            else:
                artifacts = {}
                for out_name, (out_path, _) in temp_outputs.items():
                    if not out_path.is_file():
                        raise RuntimeError(
                            f"Stage '{stage.name}' did not produce expected output '{out_name}' at {out_path}"
                        )
                    cid = self.cache.hash_file(out_path)
                    artifacts[out_name] = Artifact(
                        path=out_path,
                        content_id=cid,
                        name=out_name,
                        stage=stage.name,
                        request_key=None,
                        cached=False,
                        size_bytes=out_path.stat().st_size,
                        info=info,
                        execution=execution_meta,
                    )
        except Exception:
            for tmp_path, _ in temp_outputs.values():
                tmp_path.unlink(missing_ok=True)
            raise
        finally:
            info_tmp.unlink(missing_ok=True)
            if config_tmp is not None:
                config_tmp.unlink(missing_ok=True)

        self.trace.append(
            StageTrace(
                stage=stage.name,
                cached=False,
                request_key=request_key if mode.writes else None,
                content_ids={k: a.content_id for k, a in artifacts.items()},
                settings=settings_dict,
                wall_time_s=float(execution_meta["wall_time_s"]),
                max_rss_kb=execution_meta["max_rss_kb"],
                info=info,
                execution=execution_meta,
                dry_run=False,
            )
        )
        return artifacts["out"] if len(artifacts) == 1 and "out" in artifacts else artifacts

    def import_mesh(
        self,
        mesh_file: Artifact | str | os.PathLike[str],
        *,
        materials: Sequence[Artifact | str | os.PathLike[str]] = (),
        charges: Artifact | str | os.PathLike[str] | None = None,
        cache: CacheMode | str | bool | None = None,
        use_cache: bool | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Artifact:
        """Run the cached `poisson_import --mesh` stage and return the `MeshProperties` Artifact."""
        if self._import_mesh_stage is None:
            self._import_mesh_stage = make_import_mesh_stage(build_dir=self.build_dir)

        mat_pairs = []
        for m in materials:
            art = m if isinstance(m, Artifact) else self.raw_file(m, name="material")
            parser_mode = "mat" if art.path.suffix == ".mat" else "yaml"
            mat_pairs.append((art.content_id, parser_mode, art.path.name, art))
        mat_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

        mat_arts = [item[3] for item in mat_pairs]
        mat_formats = [item[1] for item in mat_pairs]

        inputs: dict[str, Any] = {"mesh": mesh_file}
        settings: dict[str, Any] = {}
        if mat_arts:
            inputs["materials"] = mat_arts
            settings["material_formats"] = mat_formats
        if charges is not None:
            inputs["charges"] = charges

        result = self.run_stage(
            self._import_mesh_stage,
            inputs=inputs,
            settings=settings,
            cache=cache,
            use_cache=use_cache,
            repeats=repeats,
            warmup=warmup,
            env=env,
        )
        assert isinstance(result, Artifact)
        return result

    def import_free_charges(
        self,
        primaries_file: Artifact | str | os.PathLike[str],
        *,
        cache: CacheMode | str | bool | None = None,
        use_cache: bool | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Artifact:
        """Run the cached `poisson_import --primaries` stage and return the `FreeCharges` Artifact."""
        if self._import_free_charges_stage is None:
            self._import_free_charges_stage = make_import_free_charges_stage(build_dir=self.build_dir)
        result = self.run_stage(
            self._import_free_charges_stage,
            inputs={"primaries": primaries_file},
            cache=cache,
            use_cache=use_cache,
            repeats=repeats,
            warmup=warmup,
            env=env,
        )
        assert isinstance(result, Artifact)
        return result

    def adapt_mesh(
        self,
        mesh: Artifact | str | os.PathLike[str],
        solution: Artifact | str | os.PathLike[str],
        *,
        config: Mapping[str, Any],
        history: Artifact | str | os.PathLike[str] | None = None,
        with_parents: bool = False,
        with_history: bool = False,
        cache: CacheMode | str | bool | None = None,
        use_cache: bool | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Artifact | dict[str, Artifact]:
        """Run the cached `poisson_adapt` AMR stage and return the refined mesh (or dict if parents/history requested)."""
        key = (with_parents, with_history)
        stage = self._adapt_stages.get(key)
        if stage is None:
            stage = make_adapt_stage(build_dir=self.build_dir, with_parents=with_parents, with_history=with_history)
            self._adapt_stages[key] = stage
        inputs: dict[str, Any] = {"mesh": mesh, "solution": solution}
        if history is not None:
            inputs["history"] = history
        return self.run_stage(
            stage,
            inputs=inputs,
            settings=config,
            cache=cache,
            use_cache=use_cache,
            repeats=repeats,
            warmup=warmup,
            env=env,
        )


def compare(
    name_or_options: str | Sequence[Any] | Mapping[str, Any] | Compare[Any],
    options: Sequence[Any] | Mapping[str, Any] | None = None,
) -> Any:
    """Module-level `compare(...)` usable inside any function executed by `run_experiment`."""
    ctx = _ACTIVE_RUN.get()
    if ctx is None:
        raise RuntimeError("compare() can only be called inside run_experiment(...)")
    return ctx.compare(name_or_options, options)


def resolve_slurm_shard(env: Mapping[str, str] | None = None) -> tuple[int, int] | None:
    """Return `(shard_index, shard_total)` from SLURM job-array environment variables, or `None`."""
    src = os.environ if env is None else env
    raw_id = src.get("SLURM_ARRAY_TASK_ID")
    if raw_id is None or raw_id == "":
        return None
    task_id = int(raw_id)
    task_min = int(src.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_step = int(src.get("SLURM_ARRAY_TASK_STEP", "1"))
    if task_step <= 0:
        raise ValueError(f"SLURM_ARRAY_TASK_STEP must be positive, got {task_step}")
    if (task_id - task_min) < 0 or (task_id - task_min) % task_step != 0:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={task_id} does not align with min={task_min}, step={task_step}"
        )
    index = (task_id - task_min) // task_step

    raw_count = src.get("SLURM_ARRAY_TASK_COUNT")
    raw_max = src.get("SLURM_ARRAY_TASK_MAX")
    expected_from_range: int | None = None
    if raw_max is not None and raw_max != "":
        expected_from_range = ((int(raw_max) - task_min) // task_step) + 1

    if raw_count is not None and raw_count != "":
        total = int(raw_count)
        if expected_from_range is not None and total != expected_from_range:
            raise ValueError(
                f"Sparse SLURM job array detected (min={task_min}, max={raw_max}, step={task_step}, "
                f"count={total} != {expected_from_range}); pass an explicit shard=(index, total) mapping instead"
            )
    elif expected_from_range is not None:
        total = expected_from_range
    else:
        raise ValueError(
            "SLURM_ARRAY_TASK_ID is set but neither SLURM_ARRAY_TASK_COUNT nor SLURM_ARRAY_TASK_MAX is set"
        )

    if total < 1 or not (0 <= index < total):
        raise ValueError(f"Invalid SLURM shard ({index}, {total})")
    return (index, total)


def _normalize_shard(shard: tuple[int, int] | Literal["slurm"] | None) -> tuple[int, int] | None:
    if shard is None:
        return None
    if shard == "slurm":
        return resolve_slurm_shard()
    if not isinstance(shard, tuple) or len(shard) != 2:
        raise ValueError(f"shard must be (index, total), 'slurm', or None; got {shard!r}")
    index, total = shard
    if (
        isinstance(index, bool)
        or isinstance(total, bool)
        or not isinstance(index, int)
        or not isinstance(total, int)
        or total < 1
        or not (0 <= index < total)
    ):
        raise ValueError(f"Invalid shard ({index!r}, {total!r}); expected 0 <= index < total and total >= 1")
    return (index, total)


def _discover_branch_paths(
    pipeline_fn: Callable[..., Any],
    *,
    takes_ctx: bool,
    cache: Cache,
    executor: Executor,
    build_dir: Path | str | None,
    stage_cache_modes: Mapping[str, CacheMode],
    repeats: int,
    warmup: int,
) -> list[tuple[_ChoiceStep, ...]]:
    """Discover all branch choice paths via a zero-execution dry-run pass."""
    todo_stack: list[tuple[_ChoiceStep, ...]] = [()]
    discovered: list[tuple[_ChoiceStep, ...]] = []
    while todo_stack:
        path = todo_stack.pop()
        with cache.lease() as branch_lease:
            ctx = RunContext(
                cache=cache,
                executor=executor,
                build_dir=build_dir,
                path=path,
                todo_stack=todo_stack,
                lease=branch_lease,
                cache_mode=CacheMode.READ_ONLY,
                stage_cache_modes=stage_cache_modes,
                default_repeats=repeats,
                default_warmup=warmup,
                dry_run=True,
            )
            token = _ACTIVE_RUN.set(ctx)
            try:
                pipeline_fn(ctx) if takes_ctx else pipeline_fn()
                if ctx._step < len(path):
                    raise RuntimeError(
                        f"Non-deterministic compare() replay: branch returned early at step {ctx._step} "
                        f"without reaching queued choice '{path[ctx._step].name}'"
                    )
            finally:
                _ACTIVE_RUN.reset(token)
        discovered.append(tuple(ctx._prefix))
    return discovered


def run_experiment(
    pipeline_fn: Callable[..., Any],
    *,
    cache: Cache | None = None,
    executor: Executor | None = None,
    build_dir: Path | str | None = None,
    cache_mode: CacheMode | str | bool | None = None,
    stage_cache_modes: Mapping[str, CacheMode | str | bool] | None = None,
    repeats: int = 1,
    warmup: int = 0,
    dry_run: bool = False,
    shard: tuple[int, int] | Literal["slurm"] | None = None,
    continue_on_error: bool = False,
    max_branches: int | None = None,
) -> ExperimentReport:
    """Execute `pipeline_fn` across every combination of `compare(...)` choices without loops.

    Pass `shard=(index, total)` or `shard="slurm"` (reads `SLURM_ARRAY_TASK_ID` /
    `SLURM_ARRAY_TASK_COUNT`) to execute only branches where `branch_index % total == index`
    for multi-node SLURM job-array sweeps against a shared cache.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    resolved_shard = _normalize_shard(shard)
    cache = cache if cache is not None else Cache()
    executor = executor if executor is not None else LocalExecutor()
    resolved_cache_mode = normalize_cache_mode(cache_mode) if cache_mode is not None else None
    resolved_stage_modes = (
        {name: normalize_cache_mode(mode) for name, mode in stage_cache_modes.items()}
        if stage_cache_modes is not None
        else {}
    )

    sig = inspect.signature(pipeline_fn)
    takes_ctx = len(sig.parameters) > 0

    branches: list[BranchResult] = []
    overall_start = time.perf_counter()

    if resolved_shard is not None and resolved_shard[1] > 1:
        shard_idx, shard_total = resolved_shard
        all_paths = _discover_branch_paths(
            pipeline_fn,
            takes_ctx=takes_ctx,
            cache=cache,
            executor=executor,
            build_dir=build_dir,
            stage_cache_modes=resolved_stage_modes,
            repeats=repeats,
            warmup=warmup,
        )
        selected = [(idx, path) for idx, path in enumerate(all_paths) if idx % shard_total == shard_idx]
        if max_branches is not None:
            selected = selected[:max_branches]

        for global_idx, path in selected:
            dummy_stack: list[tuple[_ChoiceStep, ...]] = []
            with cache.lease() as branch_lease:
                ctx = RunContext(
                    cache=cache,
                    executor=executor,
                    build_dir=build_dir,
                    path=path,
                    todo_stack=dummy_stack,
                    lease=branch_lease,
                    cache_mode=resolved_cache_mode,
                    stage_cache_modes=resolved_stage_modes,
                    default_repeats=repeats,
                    default_warmup=warmup,
                    dry_run=dry_run,
                )
                token = _ACTIVE_RUN.set(ctx)
                branch_start = time.perf_counter()
                result: Any = None
                error: str | None = None
                try:
                    result = pipeline_fn(ctx) if takes_ctx else pipeline_fn()
                    if ctx._step < len(path):
                        raise RuntimeError(
                            f"Non-deterministic compare() replay: branch returned early at step {ctx._step} "
                            f"without reaching queued choice '{path[ctx._step].name}'"
                        )
                    if dummy_stack:
                        raise RuntimeError(
                            "Sharded run discovered unplanned compare() choices not seen during dry-run discovery; "
                            "compare() branching in sharded runs must not depend on uncached stage outputs"
                        )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    _ACTIVE_RUN.reset(token)

            branches.append(
                BranchResult(
                    index=global_idx,
                    choices=dict(ctx.choices),
                    choice_labels=dict(ctx.choice_labels),
                    stages=list(ctx.trace),
                    result=result,
                    error=error,
                    wall_time_s=time.perf_counter() - branch_start,
                )
            )

        return ExperimentReport(
            branches=branches,
            total_wall_time_s=time.perf_counter() - overall_start,
            dry_run=dry_run,
            shard=resolved_shard,
        )

    todo_stack: list[tuple[_ChoiceStep, ...]] = [()]
    while todo_stack:
        if max_branches is not None and len(branches) >= max_branches:
            break
        path = todo_stack.pop()
        with cache.lease() as branch_lease:
            ctx = RunContext(
                cache=cache,
                executor=executor,
                build_dir=build_dir,
                path=path,
                todo_stack=todo_stack,
                lease=branch_lease,
                cache_mode=resolved_cache_mode,
                stage_cache_modes=resolved_stage_modes,
                default_repeats=repeats,
                default_warmup=warmup,
                dry_run=dry_run,
            )
            token = _ACTIVE_RUN.set(ctx)
            branch_start = time.perf_counter()
            result = None
            error = None
            try:
                result = pipeline_fn(ctx) if takes_ctx else pipeline_fn()
                if ctx._step < len(path):
                    raise RuntimeError(
                        f"Non-deterministic compare() replay: branch returned early at step {ctx._step} "
                        f"without reaching queued choice '{path[ctx._step].name}'"
                    )
            except Exception as exc:
                if not continue_on_error:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            finally:
                _ACTIVE_RUN.reset(token)

        branches.append(
            BranchResult(
                index=len(branches),
                choices=dict(ctx.choices),
                choice_labels=dict(ctx.choice_labels),
                stages=list(ctx.trace),
                result=result,
                error=error,
                wall_time_s=time.perf_counter() - branch_start,
            )
        )

    return ExperimentReport(
        branches=branches,
        total_wall_time_s=time.perf_counter() - overall_start,
        dry_run=dry_run,
        shard=resolved_shard,
    )
