"""Loop-free Compare(...) replay driver: branch exploration, cached stage execution, sharding."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import inspect
import os
import re
import shutil
import signal
import threading
import time
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, Union

from . import stages as _stages
from .benchmark import measure
from .cache import Artifact, Cache, CacheLease, CacheMode, file_signature, normalize_cache_mode
from .executor import ExecutionRequest, Executor, LocalExecutor
from .hashing import blake3_file, canonical_json
from .report import BranchResult, ExperimentReport, StageTrace
from .stages import Stage, StageCall

__all__ = [
    "BranchResult",
    "CacheMissError",
    "Compare",
    "ExperimentReport",
    "RunContext",
    "StageTrace",
    "compare",
    "resolve_slurm_shard",
    "run_experiment",
]


class CacheMissError(RuntimeError):
    """Raised when `require_cached=True` and a stage output is not found in the cache."""

    def __init__(self, stage_name: str, request_key: str) -> None:
        self.stage_name = stage_name
        self.request_key = request_key
        super().__init__(
            f"Stage '{stage_name}' (request {request_key[:12]}) is not in the cache and execution is disabled"
        )


_SAFE_OUT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

T = TypeVar("T")

_ACTIVE_RUN: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("poisson_active_run", default=None)

_PathLike = Union[Artifact, str, "os.PathLike[str]"]
InputValue = Optional[Union[_PathLike, Sequence[_PathLike]]]


@dataclass(frozen=True)
class _ChoiceStep:
    """Recorded choice-site expectation used to detect non-deterministic replays.

    `split` is the site's `compare(..., split=True)` flag (DAG array grouping only); it takes no
    part in the replay check or in equality."""

    name: str
    index: int
    label: str
    num_options: int
    split: bool = field(default=False, compare=False)


class Compare(Generic[T]):
    """Declarative choice object for branching without writing explicit loops.

    Examples:
        Compare(["P1", "P2"], name="assembly")
        Compare({"coarse": ("coarse.msh", 1e-4), "fine": ("fine.msh", 1e-6)}, name="case")

    `split=True` gives each option its own DAG arrays (as `experiment_main(split_by=name)` does);
    it changes nothing in shard mode or local runs.
    """

    def __init__(
        self, options: Sequence[T] | Mapping[str, T], *, name: str | None = None, split: bool = False
    ) -> None:
        if not isinstance(split, bool):
            raise TypeError(f"Compare split must be True or False, got {split!r}")
        self.name = name
        self.split = split
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


@dataclass
class _ExperimentState:
    """State shared by every branch of one `run_experiment` call.

    - `executed`: outputs of every stage request executed (or planned, in a dry run) during this
      experiment, keyed by `request_key`. Later branches reuse them instead of re-running a shared
      prefix stage under `refresh` / `off` / `read_only`-miss.
    - `stages`: `Stage` objects built by the project helpers in `stages.py`, keyed by helper key.
    """

    executed: dict[str, dict[str, Artifact]] = field(default_factory=dict)
    stages: dict[Hashable, Stage] = field(default_factory=dict)


@dataclass(frozen=True)
class _BranchConfig:
    """Experiment-wide settings handed to every `RunContext`."""

    cache: Cache
    executor: Executor
    build_dir: Path | str | None
    cache_mode: CacheMode | None
    stage_cache_modes: Mapping[str, CacheMode]
    repeats: int
    warmup: int
    stage_repeats: Mapping[str, int]
    stage_warmup: Mapping[str, int]
    dry_run: bool
    require_cached: bool = False
    force_cache_mode: CacheMode | None = None


@dataclass
class _StageCallRecord:
    """One `run_stage` call seen while planning a DAG (see `dag.py`).

    `outcome` is "hit" (already cached), "run" (would execute: a new task) or "reused" (the same
    request as an earlier planned call). `parents` are the request keys of the planned calls whose
    placeholder outputs this call reads.
    """

    step: int
    stage: str
    request_key: str
    outcome: str
    parents: list[str]
    resources: dict[str, Any] | None
    writes: bool
    check: str
    known_inputs: dict[str, Any]
    mode: str
    repeats: int
    warmup: int
    threads: int | None = None  # the most threads the call's env asks for (`threads=` or *_NUM_THREADS)
    threads_floor: bool = False  # from `threads=`: raises a default cpus count instead of conflicting


@dataclass(frozen=True)
class _DagTarget:
    """The one stage call a DAG task executes: call number `step` of its branch, whose code,
    settings, env and outputs must hash to `check` (the request key without inputs), whose inputs
    known at planning time must still have the content ids in `known_inputs` (`None` marks an
    input a parent task produces), and which runs with the planned cache mode and repeats."""

    step: int
    stage: str
    check: str
    known_inputs: Mapping[str, Any]
    mode: CacheMode
    repeats: int
    warmup: int


def _known_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Input content ids with dry-run placeholders (outputs of planned tasks) replaced by `None`."""

    def known(cid: str) -> str | None:
        return None if cid.startswith("dryrun:") else cid

    return {k: [known(c) for c in v] if isinstance(v, list) else known(v) for k, v in spec.items()}


class _DagTaskDone(BaseException):
    """Stops a DAG task's replay right after its last stage ran (a `BaseException`, so a
    pipeline's `except Exception:` can't swallow it). `done` maps the steps of the task's own
    stage calls that finished to their traces, in call order."""

    def __init__(self, done: dict[int, StageTrace]) -> None:
        super().__init__("DAG task done")
        self.done = done


@dataclass(frozen=True)
class _ResolvedInputs:
    artifacts: dict[str, Artifact | list[Artifact]]
    spec: dict[str, Any]
    watched_paths: list[Path]
    has_placeholder: bool


def _validate_repeats(repeats: int, warmup: int, where: str = "") -> None:
    if repeats < 1:
        raise ValueError(f"repeats{where} must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup{where} must be >= 0, got {warmup}")


def _single_or_dict(artifacts: dict[str, Artifact]) -> Artifact | dict[str, Artifact]:
    return artifacts["out"] if len(artifacts) == 1 and "out" in artifacts else artifacts


def _is_dry_run_placeholder(art: Artifact) -> bool:
    return art.content_id.startswith("dryrun:")


THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""The thread-count variables `run_stage(..., threads=N)` sets (the ones `job.sbatch` exports)."""


def _thread_count(key: str, value: str) -> int:
    """A `*_NUM_THREADS` value as a thread count; `OMP_NUM_THREADS` may list nested levels (`4,2`),
    of which the first is the count started at once."""
    parts = [p.strip() for p in value.split(",")] if key == "OMP_NUM_THREADS" else [value.strip()]
    if not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"Stage env {key} must be a positive integer, got {value!r}")
    return int(parts[0])


def _apply_threads(env: dict[str, str] | None, threads: int | None) -> tuple[dict[str, str] | None, int | None]:
    """Add `threads=` to a stage's env and return it with the most threads any of its
    `THREAD_ENV_KEYS` asks for (`None` when it sets none: the stage inherits the job's)."""
    if threads is not None:
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError(f"threads must be a positive integer, got {threads!r}")
        env = dict(env or {})
        for key in THREAD_ENV_KEYS:
            if key in env and env[key].strip() != str(threads):
                raise ValueError(f"threads={threads} contradicts env {key}={env[key]!r}; set the thread count once")
            env[key] = str(threads)
    counts = [_thread_count(k, env[k]) for k in THREAD_ENV_KEYS if env and k in env]
    return env, (max(counts) if counts else None)


def _oversubscription(execution: Mapping[str, Any]) -> tuple[int, int] | None:
    """`(threads, cpus)` when the measured run started more threads than it had CPUs, from the
    executor's own host metadata (the stage's effective env, inherited or set)."""
    host = execution.get("environment")
    if not isinstance(host, Mapping):
        return None
    cpus = host.get("cpu_count")
    threading_env = host.get("threading")
    if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1 or not isinstance(threading_env, Mapping):
        return None
    counts = []
    for key in THREAD_ENV_KEYS:
        try:
            counts.append(_thread_count(key, str(threading_env[key])))
        except (KeyError, ValueError):
            continue
    threads = max(counts, default=0)
    return (threads, cpus) if threads > cpus else None


def _explicit_cpus(resources: Mapping[str, Any] | None) -> int | None:
    """The `cpus` a call's own `resources=` asks for, if it is a valid count (`dag.py` validates)."""
    value = (resources or {}).get("cpus")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


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
        stage_repeats: Mapping[str, int] | None = None,
        stage_warmup: Mapping[str, int] | None = None,
        dry_run: bool = False,
        require_cached: bool = False,
        state: _ExperimentState | None = None,
        force_cache_mode: CacheMode | None = None,
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
        self._stage_repeats: dict[str, int] = dict(stage_repeats) if stage_repeats else {}
        self._stage_warmup: dict[str, int] = dict(stage_warmup) if stage_warmup else {}
        self.dry_run = dry_run
        self._require_cached = require_cached
        self._force_cache_mode = force_cache_mode
        self._state = state if state is not None else _ExperimentState()
        self.choices: dict[str, Any] = {}
        self.choice_labels: dict[str, str] = {}
        self.trace: list[StageTrace] = []
        self._stage_calls = 0
        self._call_log: list[_StageCallRecord] | None = None  # set while planning a DAG
        # Set while running one DAG task: the stage calls (by step) it executes, in order.
        self._dag_targets: dict[int, _DagTarget] | None = None
        self._dag_done: dict[int, StageTrace] = {}

    @classmethod
    def _for_branch(
        cls,
        config: _BranchConfig,
        *,
        path: tuple[_ChoiceStep, ...],
        todo_stack: list[tuple[_ChoiceStep, ...]],
        lease: CacheLease,
        state: _ExperimentState,
    ) -> RunContext:
        return cls(
            cache=config.cache,
            executor=config.executor,
            build_dir=config.build_dir,
            path=path,
            todo_stack=todo_stack,
            lease=lease,
            cache_mode=config.cache_mode,
            stage_cache_modes=config.stage_cache_modes,
            default_repeats=config.repeats,
            default_warmup=config.warmup,
            stage_repeats=config.stage_repeats,
            stage_warmup=config.stage_warmup,
            dry_run=config.dry_run,
            require_cached=config.require_cached,
            state=state,
            force_cache_mode=config.force_cache_mode,
        )

    # ------------------------------------------------------------------ choices

    def compare(
        self,
        name_or_options: str | Sequence[Any] | Mapping[str, Any] | Compare[Any],
        options: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        split: bool = False,
    ) -> Any:
        """Select one option on this branch and automatically schedule all remaining options.

        `split=True` gives each option its own DAG arrays (see `Compare`)."""
        if isinstance(name_or_options, Compare):
            if options is not None:
                raise TypeError("Expected compare(Compare(...)), compare(options), or compare(name, options)")
            if split:
                raise TypeError("compare(Compare(...), split=True): pass split=True to Compare(...) instead")
            spec = name_or_options
        elif isinstance(name_or_options, str) and options is not None:
            spec = Compare(options, name=name_or_options, split=split)
        elif options is None:
            spec = Compare(name_or_options, split=split)  # type: ignore[arg-type]
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
            chosen_step = dataclasses.replace(expected, split=spec.split)
        else:
            idx = 0
            chosen_step = _ChoiceStep(choice_name, 0, spec.labels[0], len(spec.values), spec.split)
            # Push remaining options in reverse order so DFS pops them in natural user order.
            for other_idx in range(len(spec.values) - 1, 0, -1):
                other_step = _ChoiceStep(choice_name, other_idx, spec.labels[other_idx], len(spec.values), spec.split)
                self._todo_stack.append(tuple(self._prefix) + (other_step,))

        self._prefix.append(chosen_step)
        self._step += 1
        value = spec.values[idx]
        self.choices[choice_name] = value
        self.choice_labels[choice_name] = spec.labels[idx]
        return value

    # ------------------------------------------------------------------ helpers

    def raw_file(self, path: str | os.PathLike[str], *, name: str = "file") -> Artifact:
        """Wrap an external file on disk as a content-hashed Artifact."""
        return self.cache.raw_file(path, name=name)

    def get_stage(self, key: Hashable, factory: Callable[[Path | str | None], Stage]) -> Stage:
        """Return the experiment-wide `Stage` for `key`, building it once via `factory(build_dir)`."""
        stage = self._state.stages.get(key)
        if stage is None:
            stage = factory(self.build_dir)
            self._state.stages[key] = stage
        return stage

    def _pin_if_cached_object(self, art: Artifact) -> None:
        if art.path.parent == self.cache.objects_dir:
            self._lease.pin([art.path.name])

    def _resolve_cache_mode(
        self,
        stage: Stage,
        cache: CacheMode | str | bool | None,
        use_cache: bool | None,
    ) -> CacheMode:
        """`force_cache_mode` > per-call > `stage_cache_modes[stage]` > experiment `cache_mode` > `Stage.default_cache`."""
        if cache is not None and use_cache is not None:
            raise ValueError("Pass either 'cache' or 'use_cache', not both")
        if self._force_cache_mode is not None:
            return self._force_cache_mode
        if cache is not None:
            return normalize_cache_mode(cache)
        if use_cache is not None:
            return normalize_cache_mode(use_cache)
        if stage.name in self._stage_cache_modes:
            return self._stage_cache_modes[stage.name]
        if self._cache_mode is not None:
            return self._cache_mode
        return normalize_cache_mode(stage.default_cache)

    def _resolve_repeats(self, stage: Stage, repeats: int | None, warmup: int | None) -> tuple[int, int]:
        """Per-call > `stage_repeats` / `stage_warmup`[stage] > experiment `repeats` / `warmup`."""
        n_repeats = repeats if repeats is not None else self._stage_repeats.get(stage.name, self._default_repeats)
        n_warmup = warmup if warmup is not None else self._stage_warmup.get(stage.name, self._default_warmup)
        _validate_repeats(n_repeats, n_warmup, f" for stage '{stage.name}'")
        return n_repeats, n_warmup

    def _resolve_inputs(self, inputs: Mapping[str, InputValue]) -> _ResolvedInputs:
        artifacts: dict[str, Artifact | list[Artifact]] = {}
        spec: dict[str, Any] = {}
        watched: list[Path] = []
        has_placeholder = False

        def resolve_one(key: str, value: Any) -> Artifact:
            nonlocal has_placeholder
            if isinstance(value, Artifact):
                if _is_dry_run_placeholder(value):
                    has_placeholder = True
                else:
                    self._pin_if_cached_object(value)
                    watched.append(value.path)
                return value
            if isinstance(value, (str, os.PathLike)):
                art = self.raw_file(value, name=key)
                watched.append(art.path)
                return art
            raise TypeError(f"Unsupported input type for '{key}': {type(value).__name__}")

        for key, raw_val in dict(inputs).items():
            if raw_val is None:
                continue
            if isinstance(raw_val, (Artifact, str, os.PathLike)):
                art = resolve_one(key, raw_val)
                artifacts[key] = art
                spec[key] = art.content_id
            elif isinstance(raw_val, Sequence) and not isinstance(raw_val, (bytes, bytearray)):
                arts = [resolve_one(key, item) for item in raw_val]
                artifacts[key] = arts
                spec[key] = [a.content_id for a in arts]
            else:
                raise TypeError(f"Unsupported input type for '{key}': {type(raw_val).__name__}")
        return _ResolvedInputs(artifacts, spec, watched, has_placeholder)

    def _record(
        self,
        stage: Stage,
        artifacts: dict[str, Artifact],
        *,
        request_key: str | None,
        settings: dict[str, Any],
        cached: bool = False,
        reused: bool = False,
    ) -> Artifact | dict[str, Artifact]:
        first = next(iter(artifacts.values()))
        execution = first.execution if isinstance(first.execution, dict) else {}
        info = first.info if isinstance(first.info, dict) else {}
        is_placeholder = _is_dry_run_placeholder(first)

        time_s = execution.get("info_time_s")
        if not isinstance(time_s, (int, float)) or isinstance(time_s, bool):
            bench = execution.get("benchmark")
            i_stats = bench.get("info_time_s") if isinstance(bench, dict) else None
            time_s = i_stats.get("median") if isinstance(i_stats, dict) else info.get("time")
        if not isinstance(time_s, (int, float)) or isinstance(time_s, bool):
            time_s = None
        # Warnings the runner noticed (nondeterminism, oversubscription), then those the stage reported.
        runner_warnings = execution.get("warnings")
        warnings = [str(w) for w in runner_warnings] if isinstance(runner_warnings, list) else []
        stage_warnings = info.get("warnings")
        if isinstance(stage_warnings, list):
            warnings.extend(str(w) for w in stage_warnings)

        self.trace.append(
            StageTrace(
                stage=stage.name,
                cached=cached,
                reused=reused,
                request_key=request_key,
                content_ids={k: a.content_id for k, a in artifacts.items()},
                settings=settings,
                time_s=float(time_s) if time_s is not None else None,
                wall_time_s=float(execution.get("wall_time_s", 0.0) or 0.0),
                max_rss_kb=execution.get("max_rss_kb"),
                info=info,
                execution=execution,
                warnings=warnings,
                dry_run=(self.dry_run and self._dag_targets is None) or is_placeholder,
                outputs=dict(artifacts),
            )
        )
        return _single_or_dict(artifacts)

    def _reusable(self, request_key: str, *, placeholders: bool) -> dict[str, Artifact] | None:
        """Outputs of the same request executed earlier in this experiment, if still on disk.

        Dry-run placeholders count only when `placeholders` (a dry-run call); a call that really
        executes, such as a DAG task's own stage, must never "reuse" a placeholder."""
        prior = self._state.executed.get(request_key)
        if prior is None:
            return None
        for art in prior.values():
            if _is_dry_run_placeholder(art):
                if not placeholders:
                    return None
                continue
            if not art.path.is_file():
                del self._state.executed[request_key]
                return None
            self._pin_if_cached_object(art)
        return prior

    # ------------------------------------------------------------------ stage execution

    def run_stage(
        self,
        stage: Stage,
        *,
        inputs: Mapping[str, InputValue] = (),  # type: ignore[assignment]
        settings: Mapping[str, Any] | None = None,
        cache: CacheMode | str | bool | None = None,
        use_cache: bool | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
        env: Mapping[str, str] | None = None,
        resources: Mapping[str, Any] | None = None,
        threads: int | None = None,
    ) -> Artifact | dict[str, Artifact]:
        """Run a stage with content-addressed caching, 4-mode control, repeats, and dry-run.

        Lookup order: persistent cache (if the mode reads it) -> outputs of the same request
        already executed earlier in this experiment -> execute. A cache hit is used whatever
        `repeats` it was measured with; pass `cache="refresh"` to re-measure.

        `threads=N` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to N in
        the stage's `env`, so the thread count is part of the request key, and in DAG mode makes
        the task's cpus at least N. Without it (and without those keys in `env`) the stage inherits
        the job's thread count and its cache entry is reused whatever that count was. A call whose
        own `resources=` gives fewer cpus than its env's thread count is refused.
        """
        step = self._stage_calls
        self._stage_calls += 1
        targets = self._dag_targets
        target = targets.get(step) if targets is not None else None
        if target is None:
            forced = CacheMode.READ_ONLY if targets is not None else None  # other calls of a DAG task only look up
            return self._run_stage(
                stage, inputs=inputs, settings=settings, cache=cache, use_cache=use_cache, repeats=repeats,
                warmup=warmup, env=env, resources=resources, threads=threads, step=step, forced_mode=forced,
                target=None,
            )
        assert targets is not None
        if any(s < step and s not in self._dag_done for s in targets):
            # An earlier stage of this chain failed and the pipeline carried on (an `except`
            # around it): stop before this one runs on a fallback input; the task then fails.
            raise _DagTaskDone(dict(self._dag_done))
        result = self._run_stage(
            stage, inputs=inputs, settings=settings, cache=cache, use_cache=use_cache, repeats=repeats,
            warmup=warmup, env=env, resources=resources, threads=threads, step=step, forced_mode=None,
            target=target,
        )
        self._dag_done[step] = self.trace[-1]
        if step == max(targets):
            raise _DagTaskDone(dict(self._dag_done))
        return result  # a fused task: later stage calls of the chain read this output in memory

    def _run_stage(
        self,
        stage: Stage,
        *,
        inputs: Mapping[str, InputValue],
        settings: Mapping[str, Any] | None,
        cache: CacheMode | str | bool | None,
        use_cache: bool | None,
        repeats: int | None,
        warmup: int | None,
        env: Mapping[str, str] | None,
        resources: Mapping[str, Any] | None,
        threads: int | None,
        step: int,
        forced_mode: CacheMode | None,
        target: _DagTarget | None,
    ) -> Artifact | dict[str, Artifact]:
        is_target = target is not None
        mode = self._resolve_cache_mode(stage, cache, use_cache)
        if forced_mode is not None:
            mode = forced_mode
        dry_run = self.dry_run and not is_target
        planning = self._call_log is not None
        n_repeats, n_warmup = self._resolve_repeats(stage, repeats, warmup)
        if target is not None:  # the plan decided how the task's stage runs
            mode = target.mode
            n_repeats, n_warmup = target.repeats, target.warmup
        resolved = self._resolve_inputs(inputs)
        watched = resolved.watched_paths + stage.tracked_code_files()
        pre_signatures = {p: file_signature(p) for p in watched}

        settings_dict = dict(settings) if settings is not None else {}
        env_dict, env_threads = _apply_threads(self._normalize_env(env), threads)
        cpus = _explicit_cpus(resources)
        if env_threads is not None and cpus is not None and cpus < env_threads:
            raise ValueError(
                f"Stage '{stage.name}' asks for {env_threads} threads but resources cpus={cpus}; "
                "give it at least as many cpus as threads"
            )
        code_hash = stage.compute_code_hash(self.cache)
        request_key = Cache.compute_request_key(
            stage.name,
            code_hash,
            resolved.spec,
            settings_dict,
            output_specs=stage.output_specs,
            env=env_dict,
            normalize_settings=stage.normalize_settings,
        )

        check = ""
        if planning or is_target:
            check = Cache.compute_request_key(
                stage.name, code_hash, {}, settings_dict, output_specs=stage.output_specs, env=env_dict,
                normalize_settings=stage.normalize_settings,
            )
        if target is not None:
            self._check_dag_target(target, stage, check, resolved, mode)

        def log(outcome: str) -> None:
            if self._call_log is None:
                return
            parents: list[str] = []
            for value in resolved.artifacts.values():
                for art in value if isinstance(value, list) else [value]:
                    if _is_dry_run_placeholder(art) and art.request_key and art.request_key not in parents:
                        parents.append(art.request_key)
            self._call_log.append(
                _StageCallRecord(
                    step=step,
                    stage=stage.name,
                    request_key=request_key,
                    outcome=outcome,
                    parents=parents,
                    resources=dict(resources) if resources is not None else None,
                    writes=mode.writes,
                    check=check,
                    known_inputs=_known_inputs(resolved.spec),
                    mode=mode.value,
                    repeats=n_repeats,
                    warmup=n_warmup,
                    threads=env_threads,
                    threads_floor=threads is not None,
                )
            )

        # While planning a DAG, a request planned earlier wins over an old cache entry, so a
        # `refresh` call and a later `use` call of the same request agree on what will run.
        prior = self._reusable(request_key, placeholders=True) if planning else None
        if mode.reads and not resolved.has_placeholder and prior is None:
            hit = self.cache.lookup(request_key, expected_outputs=stage.output_specs, lease=self._lease)
            if hit is not None:
                log("hit")
                return self._record(stage, hit, request_key=request_key, settings=settings_dict, cached=True)

        # A DAG task's planned `refresh` must execute, even when an earlier stage of its fused chain
        # happened to produce the same request (e.g. identical bytes): the plan asked for a re-run.
        if prior is None and not (target is not None and not mode.reads):
            prior = self._reusable(request_key, placeholders=dry_run)
        if prior is not None:
            first = next(iter(prior.values()))
            log("reused")
            return self._record(stage, prior, request_key=first.request_key, settings=settings_dict, reused=True)

        if dry_run:
            artifacts = self._placeholders(stage, request_key)
            self._state.executed[request_key] = artifacts
            log("run")
            return self._record(stage, artifacts, request_key=request_key, settings=settings_dict)

        if self._require_cached:
            raise CacheMissError(stage.name, request_key)

        artifacts = self._execute(
            stage,
            mode=mode,
            resolved=resolved,
            settings=settings_dict,
            env=env_dict,
            resources=resources,
            code_hash=code_hash,
            request_key=request_key,
            repeats=n_repeats,
            warmup=n_warmup,
            pre_signatures=pre_signatures,
        )
        self._state.executed[request_key] = artifacts
        return self._record(
            stage, artifacts, request_key=request_key if mode.writes else None, settings=settings_dict
        )

    def _check_dag_target(
        self, target: _DagTarget, stage: Stage, check: str, resolved: _ResolvedInputs, mode: CacheMode
    ) -> None:
        where = f"DAG task (stage call {target.step} of this branch)"
        if stage.name != target.stage:
            raise RuntimeError(
                f"Plan mismatch: {where} is '{stage.name}', the plan expected '{target.stage}'. "
                "The experiment changed since it was planned, or its shape depends on stage outputs; "
                "plan it again or use pieces (shards)."
            )
        if check != target.check:
            raise RuntimeError(
                f"Plan mismatch: {where} '{stage.name}' has a different program, settings, env or outputs "
                "than when it was planned: the stage was rebuilt, the experiment edited, or this node links "
                "different shared libraries than the node that planned it (see Stage track_shared_libs); "
                "plan it again where the tasks run."
            )
        planned = dict(target.known_inputs)
        current = dict(resolved.spec)
        changed = sorted(set(planned) ^ set(current))
        for key in set(planned) & set(current):
            want, have = planned[key], current[key]
            if isinstance(want, list) or isinstance(have, list):
                if not (isinstance(want, list) and isinstance(have, list) and len(want) == len(have)) or any(
                    w is not None and w != h for w, h in zip(want, have)
                ):
                    changed.append(key)
            elif want is not None and want != have:
                changed.append(key)
        if changed:
            raise RuntimeError(
                f"Plan mismatch: {where} '{stage.name}': input(s) {', '.join(sorted(changed))} differ from when "
                "it was planned (an input file was edited, or an upstream result changed); plan it again."
            )
        if resolved.has_placeholder:
            missing = [k for k, v in resolved.artifacts.items() if any(
                _is_dry_run_placeholder(a) for a in (v if isinstance(v, list) else [v]))]
            raise RuntimeError(
                f"{where} '{stage.name}': input(s) {', '.join(missing)} are not in the cache. A parent task "
                "failed or ran with a different build, or the cache evicted its output."
            )
        if not mode.writes:
            raise RuntimeError(f"{where} '{stage.name}' runs with a cache mode that doesn't write ({mode.value})")

    @staticmethod
    def _normalize_env(env: Mapping[str, str] | None) -> dict[str, str] | None:
        if env is None:
            return None
        out: dict[str, str] = {}
        for k, v in env.items():
            if not isinstance(k, str):
                raise TypeError(f"Stage env keys must be str, got {type(k).__name__}")
            out[k] = str(v)
        return out

    @staticmethod
    def _placeholders(stage: Stage, request_key: str) -> dict[str, Artifact]:
        """Synthetic outputs for a dry run: deterministic ids so downstream request keys are stable."""
        return {
            out_name: Artifact(
                path=Path(f"<dry-run:{stage.name}:{out_name}{ext}>"),
                content_id=f"dryrun:{request_key}:{out_name}",
                name=out_name,
                stage=stage.name,
                request_key=request_key,
                cached=False,
                size_bytes=0,
            )
            for out_name, ext in stage.output_specs.items()
        }

    def _execute(
        self,
        stage: Stage,
        *,
        mode: CacheMode,
        resolved: _ResolvedInputs,
        settings: dict[str, Any],
        env: dict[str, str] | None,
        resources: Mapping[str, Any] | None,
        code_hash: str,
        request_key: str,
        repeats: int,
        warmup: int,
        pre_signatures: Mapping[Path, tuple[str, int, int, int, int]],
    ) -> dict[str, Artifact]:
        """Run warmup + measured repeats, then publish to the cache (or keep as ephemeral outputs)."""
        # Runner-owned staging directory ensures sibling temp files created by stage binaries
        # (e.g. poisson_adapt's *.tmp.* files) are wiped together if the stage fails or gets SIGTERM.
        stage_dir = self.cache.make_temp_path()
        ephemeral_dir: Path | None = None if mode.writes else self.cache.make_ephemeral_path()
        try:
            stage_dir.mkdir(parents=True, exist_ok=False)
            if ephemeral_dir is not None:
                ephemeral_dir.mkdir(parents=True, exist_ok=False)
            out_root = stage_dir if mode.writes else ephemeral_dir
            assert out_root is not None
            temp_outputs: dict[str, tuple[Path, str]] = {}
            for idx, (name, ext) in enumerate(stage.output_specs.items()):
                safe_name = _SAFE_OUT_NAME_RE.sub("_", name).strip("._") or "out"
                temp_outputs[name] = (out_root / f"{idx}_{safe_name}{ext}", ext)
            info_tmp = stage_dir / "info.json"
            config_tmp: Path | None = None
            if stage.uses_config_file:
                config_tmp = stage_dir / "config.json"
                config_tmp.write_text(canonical_json(settings) + "\n", encoding="utf-8")

            output_paths = {name: p for name, (p, _) in temp_outputs.items()}
            call = StageCall(
                executable=stage.executable,
                inputs=resolved.artifacts,
                outputs=output_paths,
                info_path=info_tmp,
                settings=settings,
                config_path=config_tmp,
            )
            request = ExecutionRequest(
                stage_name=stage.name,
                command=stage.build_command(call),
                env=env,
                resources=dict(resources) if resources is not None else {},
            )
            m = measure(
                self.executor,
                request,
                repeats=repeats,
                warmup=warmup,
                outputs=output_paths,
                info_path=info_tmp,
                hash_file=blake3_file,
            )
            over = _oversubscription(m.execution)
            if over is not None:
                m.execution.setdefault("warnings", []).append(
                    f"Stage '{stage.name}' requested {over[0]} threads on {over[1]} usable CPUs; "
                    "its timings are oversubscribed (give the job more CPUs, or fewer threads)"
                )

            if not mode.writes:
                expected = {p.resolve() for p in output_paths.values()}
                assert ephemeral_dir is not None
                for stray in list(ephemeral_dir.iterdir()):
                    if stray.resolve() not in expected:
                        if stray.is_dir() and not stray.is_symlink():
                            shutil.rmtree(stray, ignore_errors=True)
                        else:
                            stray.unlink(missing_ok=True)
                return self._ephemeral_artifacts(stage, output_paths, m.info, m.execution)

            # Refuse to publish under a stale key if an input or the binary changed mid-run.
            # A cache object is named by its content, but other processes touch it (LRU) or
            # re-publish identical content, which changes its mtime, ctime or inode; for those
            # only its presence and size are checked.
            objects_dir = self.cache.objects_dir.resolve()
            for p, before in pre_signatures.items():
                if Path(before[0]).parent == objects_dir:
                    if not p.is_file() or p.stat().st_size != before[2]:
                        raise RuntimeError(f"Cache object '{p}' disappeared while stage '{stage.name}' was running")
                    continue
                if not p.is_file() or file_signature(p) != before:
                    raise RuntimeError(
                        f"File '{p}' changed while stage '{stage.name}' was running; "
                        "refusing to publish result under stale cache key"
                    )
            return self.cache.store(
                request_key=request_key,
                stage_name=stage.name,
                code_hash=code_hash,
                inputs_spec=resolved.spec,
                settings=settings,
                temp_outputs=temp_outputs,
                info=m.info,
                execution=m.execution,
                lease=self._lease,
            )
        except BaseException:  # also SystemExit from SIGTERM: don't leave staged/ephemeral outputs behind
            if ephemeral_dir is not None:
                shutil.rmtree(ephemeral_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def _ephemeral_artifacts(
        self,
        stage: Stage,
        output_paths: Mapping[str, Path],
        info: dict[str, Any],
        execution: dict[str, Any],
    ) -> dict[str, Artifact]:
        artifacts: dict[str, Artifact] = {}
        for out_name, out_path in output_paths.items():
            if not out_path.is_file():
                raise RuntimeError(f"Stage '{stage.name}' did not produce expected output '{out_name}' at {out_path}")
            artifacts[out_name] = Artifact(
                path=out_path,
                content_id=self.cache.hash_file(out_path),
                name=out_name,
                stage=stage.name,
                request_key=None,
                cached=False,
                size_bytes=out_path.stat().st_size,
                info=info,
                execution=execution,
            )
        return artifacts

    # ------------------------------------------------------------------ project stage shortcuts
    # Thin wrappers over the helpers in `stages.py`; extra keywords go to `run_stage`.

    def import_mesh(
        self,
        mesh_file: Artifact | str | os.PathLike[str],
        *,
        materials: Sequence[Artifact | str | os.PathLike[str]] = (),
        charges: Artifact | str | os.PathLike[str] | None = None,
        screening: bool = True,
        **options: Any,
    ) -> Artifact:
        """See `stages.import_mesh`."""
        return _stages.import_mesh(self, mesh_file, materials=materials, charges=charges, screening=screening, **options)

    def import_free_charges(self, primaries_file: Artifact | str | os.PathLike[str], **options: Any) -> Artifact:
        """See `stages.import_free_charges`."""
        return _stages.import_free_charges(self, primaries_file, **options)

    def adapt_mesh(
        self,
        mesh: Artifact | str | os.PathLike[str],
        solution: Artifact | str | os.PathLike[str],
        *,
        config: Mapping[str, Any],
        state: Artifact | str | os.PathLike[str] | None = None,
        **options: Any,
    ) -> dict[str, Artifact]:
        """See `stages.adapt_mesh`."""
        return _stages.adapt_mesh(self, mesh, solution, config=config, state=state, **options)

    def case_mesh(self, case: Any, **mesh_options: Any) -> Artifact:
        """See `stages.case_mesh`."""
        return _stages.case_mesh(self, case, **mesh_options)


def compare(
    name_or_options: str | Sequence[Any] | Mapping[str, Any] | Compare[Any],
    options: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    split: bool = False,
) -> Any:
    """Module-level `compare(...)` usable inside any function executed by `run_experiment`."""
    ctx = _ACTIVE_RUN.get()
    if ctx is None:
        raise RuntimeError("compare() can only be called inside run_experiment(...)")
    return ctx.compare(name_or_options, options, split=split)


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


def _normalize_select(select: Mapping[str, str | Sequence[str]] | None) -> dict[str, list[str]]:
    """`{"mesh": "fine"}` / `{"mesh": ["coarse", "fine"]}` -> `{"mesh": ["fine"]}` / both labels."""
    if not select:
        return {}
    out: dict[str, list[str]] = {}
    for name, labels in select.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"select names must be non-empty strings, got {name!r}")
        values = [labels] if isinstance(labels, str) else list(labels)
        if not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"select[{name!r}] must be a label or a non-empty list of labels, got {labels!r}")
        out[name] = list(dict.fromkeys(values))
    return out


def _select_paths(
    indexed: list[tuple[int, tuple[_ChoiceStep, ...]]],
    selection: Mapping[str, Sequence[str]],
) -> list[tuple[int, tuple[_ChoiceStep, ...]]]:
    """Keep branches whose every selected choice has an allowed label; reject unknown names/labels.

    A branch that never reaches a selected choice (it sits behind an `if`) is left out.
    """
    if not selection:
        return indexed
    seen: dict[str, list[str]] = {}
    for _, path in indexed:
        for step in path:
            labels = seen.setdefault(step.name, [])
            if step.label not in labels:
                labels.append(step.label)
    for name, labels in selection.items():
        if name not in seen:
            known = ", ".join(repr(n) for n in seen) or "none"
            raise ValueError(f"select: this experiment has no compare() named {name!r} (choices: {known})")
        unknown = [label for label in labels if label not in seen[name]]
        if unknown:
            raise ValueError(
                f"select: compare({name!r}) has no option labelled {', '.join(map(repr, unknown))} "
                f"(labels: {', '.join(map(repr, seen[name]))})"
            )
    kept = []
    for index, path in indexed:
        labels = {step.name: step.label for step in path}
        if all(labels.get(name) in allowed for name, allowed in selection.items()):
            kept.append((index, path))
    if not kept:
        raise ValueError(f"select: no branch matches {dict(selection)!r}")
    return kept


def _shard_paths(
    indexed: list[tuple[int, tuple[_ChoiceStep, ...]]],
    shard: tuple[int, int],
    order: str,
) -> list[tuple[int, tuple[_ChoiceStep, ...]]]:
    """This shard's part of `indexed`: every `total`-th entry ("stride") or one contiguous run ("block")."""
    index, total = shard
    if order == "stride":
        return [item for pos, item in enumerate(indexed) if pos % total == index]
    n = len(indexed)
    return indexed[(index * n) // total : ((index + 1) * n) // total]


def _call_pipeline(pipeline_fn: Callable[..., Any], ctx: RunContext, takes_ctx: bool, path: tuple[_ChoiceStep, ...]) -> Any:
    token = _ACTIVE_RUN.set(ctx)
    try:
        result = pipeline_fn(ctx) if takes_ctx else pipeline_fn()
        if ctx._step < len(path):
            raise RuntimeError(
                f"Non-deterministic compare() replay: branch returned early at step {ctx._step} "
                f"without reaching queued choice '{path[ctx._step].name}'"
            )
        return result
    finally:
        _ACTIVE_RUN.reset(token)


def _discover_branch_paths(
    pipeline_fn: Callable[..., Any],
    *,
    takes_ctx: bool,
    config: _BranchConfig,
) -> list[tuple[_ChoiceStep, ...]]:
    """Discover all branch choice paths via a zero-execution dry-run pass.

    Uses a single short-lived lease whose pin set resets on each branch so discovery never
    accumulates quadratic lease writes or pins other shards' objects for the shard's lifetime.
    """
    discovery = dataclasses.replace(config, cache_mode=CacheMode.READ_ONLY, dry_run=True, require_cached=False)
    state = _ExperimentState()
    todo_stack: list[tuple[_ChoiceStep, ...]] = [()]
    discovered: list[tuple[_ChoiceStep, ...]] = []
    with config.cache.lease() as discovery_lease:
        while todo_stack:
            discovery_lease.begin_ephemeral_scope()
            path = todo_stack.pop()
            ctx = RunContext._for_branch(discovery, path=path, todo_stack=todo_stack, lease=discovery_lease, state=state)
            _call_pipeline(pipeline_fn, ctx, takes_ctx, path)
            discovered.append(tuple(ctx._prefix))
    return discovered


@contextlib.contextmanager
def _sigterm_as_exit() -> Iterator[None]:
    """Turn SIGTERM into `SystemExit` while the block runs, so `finally:` cleanup still happens.

    SLURM sends SIGTERM when a job hits its time limit or is cancelled (SIGKILL follows after a
    grace period). Python's default action kills the process at once, leaving the cache lockfile,
    leases and temp files behind for other jobs to wait out. Only installed from the main thread
    and only when no other handler is set; out-of-memory kills (SIGKILL) can't be caught.
    """
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        yield
        return

    def _handler(signum: int, _frame: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # a second SIGTERM kills outright
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _run_branch(
    pipeline_fn: Callable[..., Any],
    *,
    takes_ctx: bool,
    config: _BranchConfig,
    lease: CacheLease,
    state: _ExperimentState,
    index: int,
    path: tuple[_ChoiceStep, ...],
    todo_stack: list[tuple[_ChoiceStep, ...]],
    planned: bool,
    continue_on_error: bool,
) -> BranchResult:
    """Execute one branch. `planned=True` (sharding) forbids discovering new compare() choices."""
    ctx = RunContext._for_branch(config, path=path, todo_stack=todo_stack, lease=lease, state=state)
    start = time.perf_counter()
    result: Any = None
    error: str | None = None
    try:
        result = _call_pipeline(pipeline_fn, ctx, takes_ctx, path)
        if planned and todo_stack:
            raise RuntimeError(
                "Sharded run discovered unplanned compare() choices not seen during dry-run discovery; "
                "compare() branching in sharded runs must not depend on uncached stage outputs"
            )
    except Exception as exc:
        if not continue_on_error:
            raise
        error = f"{type(exc).__name__}: {exc}"

    return BranchResult(
        index=index,
        choices=dict(ctx.choices),
        choice_labels=dict(ctx.choice_labels),
        stages=list(ctx.trace),
        result=result,
        error=error,
        wall_time_s=time.perf_counter() - start,
    )


def _make_config(
    *,
    cache: Cache | None = None,
    executor: Executor | None = None,
    build_dir: Path | str | None = None,
    cache_mode: CacheMode | str | bool | None = None,
    stage_cache_modes: Mapping[str, CacheMode | str | bool] | None = None,
    repeats: int = 1,
    warmup: int = 0,
    stage_repeats: Mapping[str, int] | None = None,
    stage_warmup: Mapping[str, int] | None = None,
    dry_run: bool = False,
    require_cached: bool = False,
    force_cache_mode: CacheMode | str | bool | None = None,
) -> _BranchConfig:
    """Validate the experiment-wide options shared by `run_experiment` and the DAG mode."""
    _validate_repeats(repeats, warmup)
    stage_repeats = dict(stage_repeats) if stage_repeats else {}
    stage_warmup = dict(stage_warmup) if stage_warmup else {}
    for name, n in stage_repeats.items():
        _validate_repeats(n, 0, f" for stage '{name}'")
    for name, n in stage_warmup.items():
        _validate_repeats(1, n, f" for stage '{name}'")
    return _BranchConfig(
        cache=cache if cache is not None else Cache(),
        executor=executor if executor is not None else LocalExecutor(),
        build_dir=build_dir,
        cache_mode=normalize_cache_mode(cache_mode) if cache_mode is not None else None,
        stage_cache_modes={n: normalize_cache_mode(m) for n, m in (stage_cache_modes or {}).items()},
        repeats=repeats,
        warmup=warmup,
        stage_repeats=stage_repeats,
        stage_warmup=stage_warmup,
        dry_run=dry_run,
        require_cached=require_cached,
        force_cache_mode=normalize_cache_mode(force_cache_mode) if force_cache_mode is not None else None,
    )


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
    stage_repeats: Mapping[str, int] | None = None,
    stage_warmup: Mapping[str, int] | None = None,
    dry_run: bool = False,
    require_cached: bool = False,
    shard: tuple[int, int] | Literal["slurm"] | None = None,
    shard_order: Literal["stride", "block"] = "stride",
    select: Mapping[str, str | Sequence[str]] | None = None,
    continue_on_error: bool = False,
    max_branches: int | None = None,
    force_cache_mode: CacheMode | str | bool | None = None,
) -> ExperimentReport:
    """Execute `pipeline_fn` across every combination of `compare(...)` choices without loops.

    Per-stage settings win over experiment-wide ones (and per-call arguments win over both):
    - cache mode: per-call `cache=` > `stage_cache_modes[name]` > `cache_mode` > `Stage.default_cache`
    - repeats/warmup: per-call `repeats=`/`warmup=` > `stage_repeats[name]`/`stage_warmup[name]` >
      `repeats`/`warmup`

    A stage request executed once in this call is reused by every later branch that makes the
    same request, even under `refresh`/`off`. Pass `shard=(index, total)` or `shard="slurm"` to
    execute only some branches against a shared cache: with `shard_order="stride"` (default)
    the branches where `branch_index % total == index`, with `"block"` one contiguous run of
    branches per shard (siblings sharing a prefix stay in one shard, so it runs once there).
    `select={"mesh": "fine"}` (or `{"mesh": ["coarse", "fine"]}`) keeps only the branches whose
    choice has one of those labels; several names must all match. Branch indices stay those of
    the full experiment, and sharding splits the selected branches.
    Pass `require_cached=True` to raise `CacheMissError` instead of executing any stage on miss.
    `force_cache_mode` overrides every cache mode, per-call ones included (the collect job uses
    `"read_only"`, so a call that asks for `refresh` still finds the result a piece published).
    """
    resolved_shard = _normalize_shard(shard)
    if shard_order not in ("stride", "block"):
        raise ValueError(f"shard_order must be 'stride' or 'block', got {shard_order!r}")
    selection = _normalize_select(select)
    config = _make_config(
        cache=cache,
        executor=executor,
        build_dir=build_dir,
        cache_mode=cache_mode,
        stage_cache_modes=stage_cache_modes,
        repeats=repeats,
        warmup=warmup,
        stage_repeats=stage_repeats,
        stage_warmup=stage_warmup,
        dry_run=dry_run,
        require_cached=require_cached,
        force_cache_mode=force_cache_mode,
    )
    cache = config.cache
    takes_ctx = len(inspect.signature(pipeline_fn).parameters) > 0
    state = _ExperimentState()
    branches: list[BranchResult] = []
    overall_start = time.perf_counter()

    with _sigterm_as_exit():
        selected: list[tuple[int, tuple[_ChoiceStep, ...]]] | None = None
        sharded = resolved_shard is not None and resolved_shard[1] > 1
        if sharded or selection:
            all_paths = _discover_branch_paths(pipeline_fn, takes_ctx=takes_ctx, config=config)
            selected = _select_paths(list(enumerate(all_paths)), selection)
            if sharded:
                assert resolved_shard is not None
                selected = _shard_paths(selected, resolved_shard, shard_order)
            if max_branches is not None:
                selected = selected[:max_branches]

        with cache.lease() as experiment_lease:
            common = dict(
                takes_ctx=takes_ctx,
                config=config,
                lease=experiment_lease,
                state=state,
                continue_on_error=continue_on_error,
            )
            if selected is not None:
                for global_idx, path in selected:
                    branches.append(
                        _run_branch(pipeline_fn, index=global_idx, path=path, todo_stack=[], planned=True, **common)  # type: ignore[arg-type]
                    )
            else:
                todo_stack: list[tuple[_ChoiceStep, ...]] = [()]
                while todo_stack:
                    if max_branches is not None and len(branches) >= max_branches:
                        break
                    path = todo_stack.pop()
                    branches.append(
                        _run_branch(
                            pipeline_fn, index=len(branches), path=path, todo_stack=todo_stack, planned=False, **common  # type: ignore[arg-type]
                        )
                    )

    return ExperimentReport(
        branches=branches,
        total_wall_time_s=time.perf_counter() - overall_start,
        dry_run=dry_run,
        shard=resolved_shard,
        cache_root=cache.root,
        selection=selection or None,
    )
