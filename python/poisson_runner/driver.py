"""Loop-free Compare(...) replay driver: branch exploration, cached stage execution, sharding."""

from __future__ import annotations

import contextvars
import dataclasses
import inspect
import os
import time
from collections.abc import Hashable, Mapping, Sequence
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
    "Compare",
    "ExperimentReport",
    "RunContext",
    "StageTrace",
    "compare",
    "resolve_slurm_shard",
    "run_experiment",
]

T = TypeVar("T")

_ACTIVE_RUN: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("poisson_active_run", default=None)

_PathLike = Union[Artifact, str, "os.PathLike[str]"]
InputValue = Optional[Union[_PathLike, Sequence[_PathLike]]]


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
        state: _ExperimentState | None = None,
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
        self._state = state if state is not None else _ExperimentState()
        self.choices: dict[str, Any] = {}
        self.choice_labels: dict[str, str] = {}
        self.trace: list[StageTrace] = []

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
            state=state,
        )

    # ------------------------------------------------------------------ choices

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
        """Per-call > `stage_cache_modes[stage]` > experiment `cache_mode` > `Stage.default_cache`."""
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
        warnings = execution.get("warnings")

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
                warnings=[str(w) for w in warnings] if isinstance(warnings, list) else [],
                dry_run=self.dry_run or is_placeholder,
            )
        )
        return _single_or_dict(artifacts)

    def _reusable(self, request_key: str) -> dict[str, Artifact] | None:
        """Outputs of the same request executed earlier in this experiment, if still on disk."""
        prior = self._state.executed.get(request_key)
        if prior is None:
            return None
        for art in prior.values():
            if _is_dry_run_placeholder(art):
                if not self.dry_run:
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
    ) -> Artifact | dict[str, Artifact]:
        """Run a stage with content-addressed caching, 4-mode control, repeats, and dry-run.

        Lookup order: persistent cache (if the mode reads it) -> outputs of the same request
        already executed earlier in this experiment -> execute. A cache hit is used whatever
        `repeats` it was measured with; pass `cache="refresh"` to re-measure.
        """
        mode = self._resolve_cache_mode(stage, cache, use_cache)
        n_repeats, n_warmup = self._resolve_repeats(stage, repeats, warmup)
        resolved = self._resolve_inputs(inputs)
        watched = resolved.watched_paths + stage.tracked_code_files()
        pre_signatures = {p: file_signature(p) for p in watched}

        settings_dict = dict(settings) if settings is not None else {}
        env_dict = self._normalize_env(env)
        code_hash = stage.compute_code_hash(self.cache)
        request_key = Cache.compute_request_key(
            stage.name,
            code_hash,
            resolved.spec,
            settings_dict,
            output_specs=stage.output_specs,
            env=env_dict,
        )

        if mode.reads and not resolved.has_placeholder:
            hit = self.cache.lookup(request_key, expected_outputs=stage.output_specs, lease=self._lease)
            if hit is not None:
                return self._record(stage, hit, request_key=request_key, settings=settings_dict, cached=True)

        prior = self._reusable(request_key)
        if prior is not None:
            first = next(iter(prior.values()))
            return self._record(stage, prior, request_key=first.request_key, settings=settings_dict, reused=True)

        if self.dry_run:
            artifacts = self._placeholders(stage, request_key)
            self._state.executed[request_key] = artifacts
            return self._record(stage, artifacts, request_key=request_key, settings=settings_dict)

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
        allocate = self.cache.make_temp_path if mode.writes else self.cache.make_ephemeral_path
        temp_outputs = {name: (allocate(ext), ext) for name, ext in stage.output_specs.items()}
        info_tmp = self.cache.make_temp_path(".info.json")
        config_tmp: Path | None = None
        if stage.uses_config_file:
            config_tmp = self.cache.make_temp_path(".config.json")
            config_tmp.write_text(canonical_json(settings) + "\n", encoding="utf-8")

        try:
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

            if not mode.writes:
                return self._ephemeral_artifacts(stage, output_paths, m.info, m.execution)

            # Refuse to publish under a stale key if an input or the binary changed mid-run.
            for p, before in pre_signatures.items():
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
        except Exception:
            for tmp_path, _ in temp_outputs.values():
                tmp_path.unlink(missing_ok=True)
            raise
        finally:
            info_tmp.unlink(missing_ok=True)
            if config_tmp is not None:
                config_tmp.unlink(missing_ok=True)

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
        **options: Any,
    ) -> Artifact:
        """See `stages.import_mesh`."""
        return _stages.import_mesh(self, mesh_file, materials=materials, charges=charges, **options)

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
    """Discover all branch choice paths via a zero-execution dry-run pass."""
    discovery = dataclasses.replace(config, cache_mode=CacheMode.READ_ONLY, dry_run=True)
    state = _ExperimentState()
    todo_stack: list[tuple[_ChoiceStep, ...]] = [()]
    discovered: list[tuple[_ChoiceStep, ...]] = []
    while todo_stack:
        path = todo_stack.pop()
        with config.cache.lease() as lease:
            ctx = RunContext._for_branch(discovery, path=path, todo_stack=todo_stack, lease=lease, state=state)
            _call_pipeline(pipeline_fn, ctx, takes_ctx, path)
        discovered.append(tuple(ctx._prefix))
    return discovered


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
    shard: tuple[int, int] | Literal["slurm"] | None = None,
    continue_on_error: bool = False,
    max_branches: int | None = None,
) -> ExperimentReport:
    """Execute `pipeline_fn` across every combination of `compare(...)` choices without loops.

    Per-stage settings win over experiment-wide ones (and per-call arguments win over both):
    - cache mode: per-call `cache=` > `stage_cache_modes[name]` > `cache_mode` > `Stage.default_cache`
    - repeats/warmup: per-call `repeats=`/`warmup=` > `stage_repeats[name]`/`stage_warmup[name]` >
      `repeats`/`warmup`

    A stage request executed once in this call is reused by every later branch that makes the
    same request, even under `refresh`/`off`. Pass `shard=(index, total)` or `shard="slurm"` to
    execute only branches where `branch_index % total == index` against a shared cache.
    """
    _validate_repeats(repeats, warmup)
    stage_repeats = dict(stage_repeats) if stage_repeats else {}
    stage_warmup = dict(stage_warmup) if stage_warmup else {}
    for name, n in stage_repeats.items():
        _validate_repeats(n, 0, f" for stage '{name}'")
    for name, n in stage_warmup.items():
        _validate_repeats(1, n, f" for stage '{name}'")

    resolved_shard = _normalize_shard(shard)
    cache = cache if cache is not None else Cache()
    config = _BranchConfig(
        cache=cache,
        executor=executor if executor is not None else LocalExecutor(),
        build_dir=build_dir,
        cache_mode=normalize_cache_mode(cache_mode) if cache_mode is not None else None,
        stage_cache_modes={n: normalize_cache_mode(m) for n, m in (stage_cache_modes or {}).items()},
        repeats=repeats,
        warmup=warmup,
        stage_repeats=stage_repeats,
        stage_warmup=stage_warmup,
        dry_run=dry_run,
    )
    takes_ctx = len(inspect.signature(pipeline_fn).parameters) > 0
    state = _ExperimentState()
    branches: list[BranchResult] = []
    overall_start = time.perf_counter()

    with cache.lease() as experiment_lease:
        common = dict(
            takes_ctx=takes_ctx,
            config=config,
            lease=experiment_lease,
            state=state,
            continue_on_error=continue_on_error,
        )
        if resolved_shard is not None and resolved_shard[1] > 1:
            shard_idx, shard_total = resolved_shard
            all_paths = _discover_branch_paths(pipeline_fn, takes_ctx=takes_ctx, config=config)
            selected = [(i, p) for i, p in enumerate(all_paths) if i % shard_total == shard_idx]
            if max_branches is not None:
                selected = selected[:max_branches]
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
    )
