"""Saved experiment runs and post-hoc evaluation, fully separate from the experiment itself.

    report = run_experiment(experiment, cache=cache)
    report.save("results/sweep")                     # report.json + report.csv + report_long.csv

    saved = load_run("results/sweep")                # later, in another script
    def analysis(run, b):                            # b: one successful saved branch
        ref = saved.match(b, solver="direct")        # same choices except solver
        return {"err": run.run_stage(diff_stage, inputs={"a": b["solution"], "b": ref["solution"]})}
    ana = evaluate(saved, analysis, cache=cache)     # an ordinary ExperimentReport
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .bundle import list_bundled_names
from .cache import Cache
from .driver import RunContext, run_experiment
from .records import decode
from .report import REPORT_FORMAT_VERSION, BranchResult, ExperimentReport, StageTrace

__all__ = ["SOURCE_BRANCH", "SavedRun", "evaluate", "load_run"]

SOURCE_BRANCH = "source_branch"
"""Choice key `evaluate` uses for the saved branch each evaluation branch ran on."""


def _label(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


@dataclass
class SavedRun:
    """A run reloaded from `ExperimentReport.save(...)`."""

    path: Path
    report: ExperimentReport

    @property
    def branches(self) -> list[BranchResult]:
        return self.report.branches

    @property
    def successful(self) -> list[BranchResult]:
        return [b for b in self.report.branches if b.ok]

    def __iter__(self) -> Iterator[BranchResult]:
        return iter(self.report.branches)

    def __len__(self) -> int:
        return len(self.report.branches)

    def _known_keys(self) -> set[str]:
        return {k for b in self.report.branches for k in b.choice_labels}

    def _check_keys(self, keys: Any) -> None:
        unknown = sorted(set(keys) - self._known_keys())
        if unknown:
            raise KeyError(f"Unknown choice name(s) {unknown}; saved choices: {sorted(self._known_keys())}")

    def select(self, **labels: Any) -> list[BranchResult]:
        """All branches (failed ones included) whose choice labels equal the given values."""
        self._check_keys(labels)
        want = {k: _label(v) for k, v in labels.items()}
        return [b for b in self.report.branches if all(b.choice_labels.get(k) == v for k, v in want.items())]

    def one(self, **labels: Any) -> BranchResult:
        """The single successful branch with these choice labels; raises on 0, >1 or a failed match."""
        return self._exactly_one(self.select(**labels), {k: _label(v) for k, v in labels.items()})

    def match(self, branch: BranchResult, **overrides: Any) -> BranchResult:
        """The branch with `branch`'s choices, except the overridden ones.

        `saved.match(b, solver="direct")` finds the run that differs from `b` only in `solver`.
        The candidate must have exactly the resulting set of choices: pass `key=None` to ignore
        a choice on both sides (e.g. one that only exists on some branches). Raises when nothing,
        several branches, or a failed branch matches.
        """
        self._check_keys(overrides)
        ignored = {k for k, v in overrides.items() if v is None}
        target = {k: v for k, v in branch.choice_labels.items() if k not in ignored}
        for k, v in overrides.items():
            if v is not None:
                target[k] = _label(v)
        candidates = [
            b
            for b in self.report.branches
            if {k: v for k, v in b.choice_labels.items() if k not in ignored} == target
        ]
        return self._exactly_one(candidates, target)

    @staticmethod
    def _exactly_one(candidates: list[BranchResult], target: dict[str, str]) -> BranchResult:
        if not candidates:
            raise LookupError(f"No saved branch has choices {target}")
        if len(candidates) > 1:
            listed = "; ".join(f"#{b.index} {b.choice_labels}" for b in candidates)
            raise LookupError(
                f"{len(candidates)} saved branches match {target}: {listed}. "
                "Fix more choices, or pass key=None to drop one"
            )
        found = candidates[0]
        if not found.ok:
            raise LookupError(f"Saved branch #{found.index} matching {target} failed: {found.error}")
        return found

    def to_dataframe(self, *, layout: Literal["wide", "long"] = "wide") -> Any:
        return self.report.to_dataframe(layout=layout)

    def save_csv(self, path: str | os.PathLike[str], *, layout: Literal["wide", "long"] = "wide") -> None:
        self.report.save_csv(path, layout=layout)


def _stage_from_dict(
    d: dict[str, Any],
    objects_dir: Path | None,
    bundle_objects_dir: Path | None,
    bundled_names: set[str],
) -> StageTrace:
    kw = dict(objects_dir=objects_dir, bundle_objects_dir=bundle_objects_dir, bundled_names=bundled_names)
    outputs = decode(d.get("outputs") or {}, **kw)
    return StageTrace(
        stage=d["stage"],
        cached=bool(d.get("cached", False)),
        reused=bool(d.get("reused", False)),
        dry_run=bool(d.get("dry_run", False)),
        request_key=d.get("request_key"),
        content_ids=dict(d.get("content_ids") or {}),
        settings=dict(decode(d.get("settings") or {}, **kw)),
        time_s=d.get("time_s"),
        wall_time_s=float(d.get("wall_time_s") or 0.0),
        max_rss_kb=d.get("max_rss_kb"),
        warnings=list(d.get("warnings") or []),
        info=dict(decode(d.get("info") or {}, **kw)),
        execution=dict(decode(d.get("execution") or {}, **kw)),
        outputs=outputs if isinstance(outputs, dict) else {},
    )


def load_run(directory: str | os.PathLike[str], *, cache: Cache | None = None) -> SavedRun:
    """Reload a run saved with `ExperimentReport.save(directory)`.

    Output artifacts resolve first in `<directory>/objects/` when bundled, else in the cache
    the run used (pass `cache=` to resolve unbundled artifacts under another mount or copy of
    that cache). Records come back through `register_record` types (import the module that
    registers them before loading), else as `UnresolvedRecord`.
    """
    path = Path(directory)
    report_json = path / "report.json" if path.is_dir() else path
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if payload.get("saved_run") is not True or not isinstance(version, int):
        raise ValueError(f"{report_json} is not a saved run (use ExperimentReport.save, not save_json)")
    if version > REPORT_FORMAT_VERSION:
        raise ValueError(f"{report_json} has format_version {version}; this poisson_runner reads up to {REPORT_FORMAT_VERSION}")

    saved_root = payload.get("cache_root")
    cache_root = cache.root if cache is not None else (Path(saved_root) if saved_root else None)
    objects_dir = cache_root / "objects" if cache_root is not None else None
    bundle_objects_dir = report_json.parent / "objects"
    bundled_names = list_bundled_names(bundle_objects_dir)
    decode_kw = dict(objects_dir=objects_dir, bundle_objects_dir=bundle_objects_dir, bundled_names=bundled_names)

    branches: list[BranchResult] = []
    for bd in payload.get("branches", []):
        labels = {str(k): str(v) for k, v in (bd.get("choices") or {}).items()}
        values = decode(bd["choice_values"], **decode_kw) if "choice_values" in bd else dict(labels)
        branches.append(
            BranchResult(
                index=int(bd["index"]),
                choices=values if isinstance(values, dict) else dict(labels),
                choice_labels=labels,
                stages=[_stage_from_dict(sd, objects_dir, bundle_objects_dir, bundled_names) for sd in bd.get("stages", [])],
                result=decode(bd.get("outputs") or {}, **decode_kw),
                error=bd.get("error"),
                wall_time_s=float(bd.get("wall_time_s") or 0.0),
            )
        )
    shard = payload.get("shard")
    report = ExperimentReport(
        branches=branches,
        total_wall_time_s=float(payload.get("total_wall_time_s") or 0.0),
        dry_run=bool(payload.get("dry_run", False)),
        shard=(int(shard["index"]), int(shard["total"])) if isinstance(shard, dict) else None,
        cache_root=cache_root,
        selection=payload.get("selection") or None,
    )
    report._from_saved = True
    return SavedRun(path=report_json.parent, report=report)


def evaluate(
    saved: SavedRun | str | os.PathLike[str],
    fn: Callable[[RunContext, BranchResult], Any],
    **run_options: Any,
) -> ExperimentReport:
    """Run `fn(run, branch)` once per successful saved branch, as an ordinary experiment.

    `fn` may call cached stages and `compare(...)` like any pipeline. The returned report has the
    saved branch's index under `source_branch` and its choices as columns; an evaluation choice
    whose name clashes with a saved choice is renamed `eval.<name>` (`eval.<name>~2`, ... if that
    is taken too). `run_options` go to `run_experiment`; `cache` is where evaluation stages are
    cached and defaults to the cache the saved run used. Saved outputs are always read from where
    `load_run` resolved them (pass a `SavedRun` loaded with `cache=` to relocate them).
    """
    if not isinstance(saved, SavedRun):
        saved = load_run(saved)
    if saved.report.dry_run:
        raise ValueError("Cannot evaluate a dry-run report: its outputs are placeholders")
    sources = saved.successful
    if not sources:
        raise ValueError(f"Saved run at {saved.path} has no successful branches to evaluate")
    source_keys = {k for b in saved.branches for k in b.choice_labels}
    if SOURCE_BRANCH in source_keys:
        raise ValueError(f"Saved run already uses the reserved choice name {SOURCE_BRANCH!r}")
    if run_options.get("cache") is None and saved.report.cache_root is not None:
        run_options["cache"] = Cache(saved.report.cache_root)

    options = {str(b.index): b for b in sources}

    def pipeline(run: RunContext) -> Any:
        branch = run.compare(SOURCE_BRANCH, options)
        return fn(run, branch)

    report = run_experiment(pipeline, **run_options)

    # One name per evaluation choice across the whole report, never shadowing a saved choice.
    used = set(source_keys) | {SOURCE_BRANCH}
    rename: dict[str, str] = {}
    for eb in report.branches:
        for k in eb.choice_labels:
            if k == SOURCE_BRANCH or k in rename:
                continue
            name = k
            if k in used:
                name = f"eval.{k}"
                n = 2
                while name in used:
                    name = f"eval.{k}~{n}"
                    n += 1
            rename[k] = name
            used.add(name)

    merged: list[BranchResult] = []
    for eb in report.branches:
        label = eb.choice_labels.get(SOURCE_BRANCH, "")
        src = options.get(label)
        labels: dict[str, str] = {SOURCE_BRANCH: label}
        values: dict[str, Any] = {SOURCE_BRANCH: int(label) if label.isdigit() else label}
        if src is not None:
            labels.update(src.choice_labels)
            values.update({k: src.choices.get(k, v) for k, v in src.choice_labels.items()})
        for k, v in eb.choice_labels.items():
            if k == SOURCE_BRANCH:
                continue
            key = rename[k]
            labels[key] = v
            values[key] = eb.choices.get(k)
        merged.append(dataclasses.replace(eb, choice_labels=labels, choices=values))
    return dataclasses.replace(report, branches=merged)
