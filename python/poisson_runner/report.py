"""Per-stage traces, per-branch results, and the aggregated `ExperimentReport` (table/JSON/CSV)."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .cache import Artifact
from .records import encode

REPORT_FORMAT_VERSION = 1

Scalar = bool | int | float | str


@dataclass(frozen=True)
class StageTrace:
    """Record of one stage invocation inside a branch.

    - `cached`: served from the persistent cache (`HIT`).
    - `reused`: served from an execution earlier in the same `run_experiment` call (`REUSED`),
      e.g. a shared prefix stage under `refresh`/`off`/`read_only` that already ran on a previous branch.
    - `time_s`: the stage's self-reported `info["time"]` (median over repeats) — the headline timing.
    - `wall_time_s`: median process wall time, including process startup and file I/O.
    - `outputs`: the output `Artifact`s of this call, for provenance (evaluation uses the
      branch's returned outputs, see `BranchResult.outputs`).
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
    outputs: dict[str, Artifact] = field(default_factory=dict)

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

    @property
    def metrics(self) -> dict[str, Scalar]:
        """Flat scalar view of `info` used for CSV / DataFrame columns (see `info_scalars`)."""
        return info_scalars(self.info)


@dataclass(frozen=True)
class BranchResult:
    """Outcome of one complete branch through the user's pipeline.

    `result` is whatever the pipeline returned. Returning a dict of named outputs
    (`{"mesh": mesh_artifact, "dofs": 5200, "case": case}`) is the contract for evaluation:
    `outputs` is that dict, `scalars` its scalar entries (CSV columns `res.<key>`), and
    `branch["mesh"]` looks one output up.
    """

    index: int
    choices: dict[str, Any]
    choice_labels: dict[str, str]
    stages: list[StageTrace]
    result: Any = None
    error: str | None = None
    wall_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def outputs(self) -> dict[str, Any]:
        """Named outputs: the returned dict, `{}` for `None`, else `{"result": <value>}`."""
        if self.result is None:
            return {}
        if isinstance(self.result, Mapping):
            return dict(self.result)
        return {"result": self.result}

    @property
    def scalars(self) -> dict[str, Scalar]:
        """The outputs that are plain scalars (`bool`, `int`, `float`, `str`)."""
        return {str(k): v for k, v in self.outputs.items() if _is_scalar(v)}

    def __getitem__(self, key: str) -> Any:
        outputs = self.outputs
        if key not in outputs:
            available = ", ".join(sorted(outputs)) or "none"
            raise KeyError(f"Branch {self.index} has no output {key!r} (outputs: {available})")
        value = outputs[key]
        if isinstance(value, Artifact) and not value.content_id.startswith("dryrun:") and not value.path.is_file():
            raise FileNotFoundError(
                f"Output {key!r} of branch {self.index} is missing at {value.path} "
                "(the cache object may have been evicted or cleaned)"
            )
        return value


# ---------------------------------------------------------------------------------------------
# Flattening stage info into scalar columns
# ---------------------------------------------------------------------------------------------

_SKIPPED_INFO_KEYS = frozenset({"type", "time", "warnings"})


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str))


def info_scalars(info: Mapping[str, Any] | None) -> dict[str, Scalar]:
    """Flatten a stage's `.info.json` into `{column: scalar}`.

    - Top-level scalars (number, bool, string), except `type`, `time` (already the
      `info_time_s` column) and `warnings` (its own column). `null` values are skipped.
    - Top-level dicts one level deep as `<key>.<subkey>` (e.g. a `details` map).
    - A composite's `stages` list by step *name*: `<step>.<field>` for the step's scalars
      (`method`, `time`, ...) and `<step>.<metric>` for its `metrics`. Repeated step names
      become `<step>#0`, `<step>#1`, ...
    - Other lists and deeper nesting are skipped. Two fields that flatten to the same name are
      both kept: the later one gets `~2` (`~3`, ...) appended.
    """
    out: dict[str, Scalar] = {}
    if not isinstance(info, Mapping):
        return out
    for key, value in info.items():
        key = str(key)
        if key in _SKIPPED_INFO_KEYS or value is None:
            continue
        if _is_scalar(value):
            _put(out, key, value)
        elif key == "stages" and isinstance(value, list):
            for k, v in _flatten_steps(value).items():
                _put(out, k, v)
        elif isinstance(value, Mapping):
            for sub, sub_val in value.items():
                if _is_scalar(sub_val):
                    _put(out, f"{key}.{sub}", sub_val)
    return out


def _put(out: dict[str, Scalar], key: str, value: Scalar) -> None:
    """Insert without overwriting: a second field flattening to the same name gets `~2`, ..."""
    out[_unique(key, out)] = value


def _flatten_steps(steps: list[Any]) -> dict[str, Scalar]:
    names: list[str | None] = []
    for i, step in enumerate(steps):
        if not isinstance(step, Mapping):
            names.append(None)
            continue
        name = step.get("stage")
        names.append(name if isinstance(name, str) and name else f"step{i}")
    totals: dict[str, int] = {}
    for n in names:
        if n is not None:
            totals[n] = totals.get(n, 0) + 1

    out: dict[str, Scalar] = {}
    seen: dict[str, int] = {}
    for step, name in zip(steps, names):
        if name is None:
            continue
        if totals[name] > 1:
            idx = seen.get(name, 0)
            seen[name] = idx + 1
            name = f"{name}#{idx}"
        for k, v in step.items():
            if k in ("stage", "warnings") or v is None:
                continue
            if _is_scalar(v):
                _put(out, f"{name}.{k}", v)
            elif k == "metrics" and isinstance(v, Mapping):
                for mk, mv in v.items():
                    if _is_scalar(mv):
                        _put(out, f"{name}.{mk}", mv)
    return out


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------


def _fmt_opt_float(val: Any) -> str:
    return f"{float(val):.6f}" if isinstance(val, (int, float)) and not isinstance(val, bool) else ""


def _fmt_opt_int(val: Any) -> str:
    return str(int(val)) if isinstance(val, (int, float)) and not isinstance(val, bool) else ""


def _opt_float(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def _opt_int(val: Any) -> int | None:
    return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None


def _fmt_scalar(val: Any) -> str:
    """Full-precision cell for metric / result columns (floats round-trip exactly)."""
    if val is None:
        return ""
    if isinstance(val, float):
        return repr(val) if math.isfinite(val) else ("nan" if math.isnan(val) else ("inf" if val > 0 else "-inf"))
    return str(val)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique(name: str, used: Any) -> str:
    """`name`, or `name~2`, `name~3`, ... — the first one not in `used`."""
    if name not in used:
        return name
    n = 2
    while f"{name}~{n}" in used:
        n += 1
    return f"{name}~{n}"


class _Names:
    """Allocates collision-free column names; each distinct source gets its own column."""

    def __init__(self, reserved: list[str]) -> None:
        self._used = set(reserved)

    def alloc(self, name: str, source: Any) -> str:
        col = _unique(name, self._used)
        self._used.add(col)
        return col

    def alloc_choice(self, key: str) -> str:
        col = key
        if col in self._used:
            col = f"choice_{key}"
            suffix = 2
            while col in self._used:
                col = f"choice_{key}_{suffix}"
                suffix += 1
        self._used.add(col)
        return col


_LONG_FIXED = [
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

_WIDE_STAGE_SUFFIXES = ("status", "info_time_s", "wall_s", "rss_kb", "cpu_eff")


@dataclass
class ExperimentReport:
    """Aggregated results across all branches explored by `run_experiment`."""

    branches: list[BranchResult]
    total_wall_time_s: float
    dry_run: bool = False
    shard: tuple[int, int] | None = None
    cache_root: Path | None = None
    selection: dict[str, list[str]] | None = None

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

    @property
    def _objects_dir(self) -> Path | None:
        return self.cache_root / "objects" if self.cache_root is not None else None

    # ------------------------------------------------------------------ JSON

    def to_dict(self, *, strict: bool = False) -> dict[str, Any]:
        """JSON-ready dict. `strict=True` raises `RecordError` on outputs that cannot be saved;
        otherwise they become `$unserializable` placeholders. Choice values are always lenient."""
        objects_dir = self._objects_dir
        payload: dict[str, Any] = {
            "dry_run": self.dry_run,
            "total_wall_time_s": self.total_wall_time_s,
            "branch_count": len(self.branches),
            "cache_hits": self.cache_hits,
            "reused": self.reused,
            "cache_misses": self.cache_misses,
            "warnings": self.warnings,
            "cache_root": str(self.cache_root) if self.cache_root is not None else None,
        }
        if self.shard is not None:
            payload["shard"] = {"index": self.shard[0], "total": self.shard[1]}
        if self.selection:
            payload["selection"] = self.selection
        payload["branches"] = [
            {
                "index": b.index,
                "choices": b.choice_labels,
                "choice_values": encode(
                    b.choices, strict=False, objects_dir=objects_dir, where=f"branch {b.index} choices"
                ),
                "outputs": encode(
                    b.outputs, strict=strict, objects_dir=objects_dir, where=f"branch {b.index} outputs"
                ),
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
                        "outputs": encode(s.outputs, strict=False, objects_dir=objects_dir),
                        "settings": encode(s.settings, strict=False, objects_dir=objects_dir),
                        "time_s": s.time_s,
                        "wall_time_s": s.wall_time_s,
                        "max_rss_kb": s.max_rss_kb,
                        "warnings": s.warnings,
                        "info": encode(s.info, strict=False, objects_dir=objects_dir),
                        "execution": encode(s.execution, strict=False, objects_dir=objects_dir),
                    }
                    for s in b.stages
                ],
            }
            for b in self.branches
        ]
        return payload

    def save_json(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ saved runs

    def save(
        self,
        directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        allow_partial: bool = False,
    ) -> Path:
        """Save this run for later evaluation (`load_run(directory)`).

        Writes `report.json` (everything, including named outputs as artifact descriptors),
        `report.csv` (wide) and `report_long.csv`. Refuses dry runs, partial SLURM shards
        and `select=` subsets (unless `allow_partial=True`) and outputs that cannot be saved. Output files stay in
        the cache; the report references them.
        """
        if self.dry_run:
            raise ValueError("Cannot save a dry-run report: its outputs are placeholders")
        if self.shard is not None and self.shard[1] > 1 and not allow_partial:
            raise ValueError(
                f"Report holds only shard {self.shard[0]}/{self.shard[1]}; save the final collecting "
                "run (cache_mode='read_only', no shard) or pass allow_partial=True"
            )
        if self.selection and not allow_partial:
            raise ValueError(
                f"Report holds only the branches selected by {self.selection!r}; save the full "
                "collecting run or pass allow_partial=True"
            )
        payload = self.to_dict(strict=True)
        payload["saved_run"] = True
        payload["format_version"] = REPORT_FORMAT_VERSION
        json_text = json.dumps(payload, indent=2) + "\n"   # fails here, before anything is written

        out_dir = Path(directory)
        report_json = out_dir / "report.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / ".report.lock").open("a") as lock_fp:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another process is saving a report into {out_dir}") from None
            except OSError:
                pass  # filesystem without flock support: proceed unlocked
            if report_json.exists() and not overwrite:
                raise FileExistsError(f"{report_json} already exists; pass overwrite=True to replace it")

            writers: list[tuple[Path, Any]] = [
                (out_dir / "report.csv", lambda p: self.save_csv(p, layout="wide")),
                (out_dir / "report_long.csv", lambda p: self.save_csv(p, layout="long")),
                (report_json, lambda p: p.write_text(json_text, encoding="utf-8")),
            ]
            staged: list[tuple[Path, Path]] = []
            try:
                for target, writer in writers:
                    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=out_dir)
                    os.close(fd)
                    staged.append((Path(tmp_name), target))
                    writer(Path(tmp_name))
                # report.json is the commit point and is published last. The old one is removed
                # first, so an interrupted publish leaves no loadable report instead of a mixed one.
                report_json.unlink(missing_ok=True)
                for tmp, target in staged:
                    os.replace(tmp, target)
            finally:
                for tmp, _ in staged:
                    tmp.unlink(missing_ok=True)
        return out_dir

    # ------------------------------------------------------------------ tables

    def save_csv(self, path: str | os.PathLike[str], *, layout: Literal["wide", "long"] = "wide") -> None:
        """Export results as CSV.

        - `layout="wide"` (default): one row per branch (repeated stages in a loop are disambiguated
          as `stage#0`, `stage#1`, ...).
        - `layout="long"`: tidy table with one row per `(branch, step_index, stage)` including full
          repeat statistics (`min`/`median`/`mean`/`max`/`std`) and OS metrics. Ideal for pandas.

        Both layouts also carry the branch's scalar outputs (`res.<key>`) and every scalar of each
        stage's info (`<stage>.<field>`, see `info_scalars`). The stage's self-reported
        `info["time"]` columns come before the process wall-time columns.
        """
        if layout not in ("wide", "long"):
            raise ValueError(f"Unsupported CSV layout {layout!r}; expected 'wide' or 'long'")
        fieldnames, rows = self._table(layout, formatted=True)
        with Path(path).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def to_dataframe(self, *, layout: Literal["wide", "long"] = "wide") -> Any:
        """The CSV table as a pandas DataFrame with typed values (pandas is imported lazily)."""
        if layout not in ("wide", "long"):
            raise ValueError(f"Unsupported layout {layout!r}; expected 'wide' or 'long'")
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("to_dataframe() needs pandas; install it or use save_csv()") from exc
        fieldnames, rows = self._table(layout, formatted=False)
        return pd.DataFrame(rows, columns=fieldnames)

    def _table(self, layout: str, *, formatted: bool) -> tuple[list[str], list[dict[str, Any]]]:
        return self._table_long(formatted) if layout == "long" else self._table_wide(formatted)

    def _table_long(self, formatted: bool) -> tuple[list[str], list[dict[str, Any]]]:
        f_float = _fmt_opt_float if formatted else _opt_float
        f_int = _fmt_opt_int if formatted else _opt_int
        f_val = _fmt_scalar if formatted else (lambda v: v)

        # Fixed columns, then result columns, then stage metric columns, each name allocated once;
        # choices are named last so they can never shadow a data column.
        names = _Names(["branch", *_LONG_FIXED])
        res_cols = {k: names.alloc(f"res.{k}", ("res", k)) for k in self._result_keys()}
        metric_cols: dict[tuple[str, str], str] = {}
        for b in self.branches:
            for st in b.stages:
                for k in sorted(st.metrics):
                    if (st.stage, k) not in metric_cols:
                        metric_cols[(st.stage, k)] = names.alloc(f"{st.stage}.{k}", (st.stage, k))
        choice_keys = self._choice_keys()
        choice_col_map = {k: names.alloc_choice(k) for k in choice_keys}
        order = [
            "branch",
            *(choice_col_map[k] for k in choice_keys),
            *_LONG_FIXED,
            *res_cols.values(),
            *metric_cols.values(),
        ]

        rows: list[dict[str, Any]] = []
        for b in self.branches:
            res = {res_cols[str(k)]: f_val(v) for k, v in b.scalars.items()}
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
                    "repeats": f_int(bench.get("repeats", 1 if not st.dry_run else None)),
                    "warmup": f_int(bench.get("warmup", 0 if not st.dry_run else None)),
                    "info_time_s": f_float(st.time_s),
                    "info_time_min_s": f_float(i_stats.get("min")),
                    "info_time_median_s": f_float(i_stats.get("median")),
                    "info_time_mean_s": f_float(i_stats.get("mean")),
                    "info_time_max_s": f_float(i_stats.get("max")),
                    "info_time_std_s": f_float(i_stats.get("std")),
                    "wall_s": f_float(st.wall_time_s),
                    "wall_min_s": f_float(w_stats.get("min")),
                    "wall_median_s": f_float(w_stats.get("median")),
                    "wall_mean_s": f_float(w_stats.get("mean")),
                    "wall_max_s": f_float(w_stats.get("max")),
                    "wall_std_s": f_float(w_stats.get("std")),
                    "max_rss_kb": f_int(st.max_rss_kb),
                    "cpu_user_s": f_float(execution.get("cpu_user_s")),
                    "cpu_sys_s": f_float(execution.get("cpu_sys_s")),
                    "cpu_efficiency": f_float(execution.get("cpu_efficiency")),
                    "major_page_faults": f_int(execution.get("major_page_faults")),
                    "hostname": str(execution.get("hostname", "") or ""),
                    "cpu_model": str(env_meta.get("cpu_model", "") or ""),
                    "omp_num_threads": str(threading_meta.get("OMP_NUM_THREADS", "") or ""),
                    "slurm_job_id": str(slurm_meta.get("SLURM_JOB_ID", "") or ""),
                    "request_key": st.request_key or "",
                    "warnings": "; ".join(st.warnings),
                    "error": b.error or "",
                    **res,
                    **{metric_cols[(st.stage, k)]: f_val(v) for k, v in st.metrics.items()},
                }
                rows.append(_fill(row, order, formatted))
        return order, rows

    def _result_keys(self) -> list[str]:
        keys: list[str] = []
        for b in self.branches:
            for k in b.scalars:
                if str(k) not in keys:
                    keys.append(str(k))
        return keys

    def _table_wide(self, formatted: bool) -> tuple[list[str], list[dict[str, Any]]]:
        f_float = _fmt_opt_float if formatted else _opt_float
        f_int = _fmt_opt_int if formatted else _opt_int
        f_val = _fmt_scalar if formatted else (lambda v: v)
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

        fixed = ["branch_wall_s", "error"]
        names = _Names(["branch", *fixed])
        res_cols = {k: names.alloc(f"res.{k}", ("res", k)) for k in self._result_keys()}
        stage_fixed: dict[str, dict[str, str]] = {}
        metric_cols: dict[tuple[str, str], str] = {}
        data_cols: list[str] = []
        for sn in stage_cols:
            stage_fixed[sn] = {sfx: names.alloc(f"{sn}_{sfx}", (sn, f"_{sfx}")) for sfx in _WIDE_STAGE_SUFFIXES}
            data_cols.extend(stage_fixed[sn].values())
            for smap in branch_stage_maps:
                st = smap.get(sn)
                if st is None:
                    continue
                for k in sorted(st.metrics):
                    if (sn, k) not in metric_cols:
                        metric_cols[(sn, k)] = names.alloc(f"{sn}.{k}", (sn, k))
                        data_cols.append(metric_cols[(sn, k)])
        choice_col_map = {k: names.alloc_choice(k) for k in choice_keys}
        order = ["branch", *(choice_col_map[k] for k in choice_keys), *fixed, *res_cols.values(), *data_cols]

        rows: list[dict[str, Any]] = []
        for b, smap in zip(self.branches, branch_stage_maps):
            row: dict[str, Any] = {
                "branch": b.index,
                "branch_wall_s": f"{b.wall_time_s:.6f}" if formatted else b.wall_time_s,
                "error": b.error or "",
            }
            for k in choice_keys:
                row[choice_col_map[k]] = b.choice_labels.get(k, "")
            for k, v in b.scalars.items():
                row[res_cols[str(k)]] = f_val(v)
            for sn in stage_cols:
                st = smap.get(sn)
                if st is None:
                    continue
                cols = stage_fixed[sn]
                row[cols["status"]] = st.status
                row[cols["info_time_s"]] = f_float(st.time_s)
                row[cols["wall_s"]] = f_float(st.wall_time_s)
                row[cols["rss_kb"]] = f_int(st.max_rss_kb)
                row[cols["cpu_eff"]] = f_float(_as_dict(st.execution).get("cpu_efficiency"))
                for k, v in st.metrics.items():
                    row[metric_cols[(sn, k)]] = f_val(v)
            rows.append(_fill(row, order, formatted))
        return order, rows

    # ------------------------------------------------------------------ terminal table

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


def _fill(row: dict[str, Any], order: list[str], formatted: bool) -> dict[str, Any]:
    """Every column present: `""` in CSV cells, `None` (NaN in pandas) in typed rows."""
    empty: Any = "" if formatted else None
    return {c: row.get(c, empty) for c in order}
