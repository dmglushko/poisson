"""DAG mode: plan an experiment as stage-level tasks grouped into Slurm job arrays, and run one task.

`plan_dag(...)` dry-runs every (selected) branch with the experiment's real cache modes and
records each stage call that would execute as a task, with the tasks whose outputs it reads as
parents. Chains of tasks where each one only feeds the next (same cpus and memory) are fused
into one unit, run back to back in one job; without fusion each task is its own unit. Units are
grouped into arrays by (stages, resources, split labels, parent arrays), in planning order, which
is a topological order. `run_dag_task(...)` replays one unit's branch: every other stage call
only looks up the cache (a miss becomes a placeholder, as in a dry run), the unit's own calls
execute, and the replay stops after the last. See `architecture.md` ("DAG mode").
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheMode
from .driver import (
    RunContext,
    _call_pipeline,
    _ChoiceStep,
    _DagTarget,
    _DagTaskDone,
    _discover_branch_paths,
    _ExperimentState,
    _make_config,
    _normalize_select,
    _select_paths,
    _sigterm_as_exit,
)
from .report import StageTrace

__all__ = [
    "DagPlan",
    "parse_resources",
    "plan_dag",
    "read_plan",
    "run_dag_task",
]

PLAN_FORMAT_VERSION = 3
DEFAULT_RESOURCES: dict[str, Any] = {"cpus": 1, "time": "01:00:00", "mem_per_cpu": "3900M"}
DEFAULT_MAX_ARRAY_SIZE = 1000
DEFAULT_MAX_FUSED_TIME = "24:00:00"  # DelftBlue's limit for education accounts; research allows more

_TIME_RE = re.compile(r"^(\d+-)?\d+(:\d{1,2}){0,2}$")
_MEM_RE = re.compile(r"^\d+[KMGT]?$")


# ---------------------------------------------------------------------------- resources

def normalize_resources(raw: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Validate `{"cpus", "time", "mem_per_cpu"}` (also `mem-per-cpu`); any subset."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key).replace("-", "_")
        if name == "cpus":
            try:
                cpus = int(value)
            except (TypeError, ValueError):
                cpus = 0
            if isinstance(value, bool) or cpus < 1 or str(cpus) != str(value).strip():
                raise ValueError(f"{where}: cpus must be a positive integer, got {value!r}")
            out["cpus"] = cpus
        elif name == "time":
            try:
                time_seconds(value)  # also rejects out-of-range fields such as 00:99:00
            except (AttributeError, ValueError):
                raise ValueError(
                    f"{where}: time must look like 02:00:00 or 1-00:00:00 (0 for no limit), got {value!r}"
                ) from None
            out["time"] = value.strip()
        elif name == "mem_per_cpu":
            if not isinstance(value, str) or not _MEM_RE.match(value.strip()):
                raise ValueError(f"{where}: mem_per_cpu must look like 3900M or 8G, got {value!r}")
            out["mem_per_cpu"] = value.strip()
        else:
            raise ValueError(f"{where}: unknown resource {key!r} (use cpus, time, mem_per_cpu)")
    return out


def time_seconds(value: str) -> int:
    """Seconds in a Slurm time limit: `M`, `M:S`, `H:M:S`, `D-H`, `D-H:M` or `D-H:M:S`."""
    text = value.strip()
    if not _TIME_RE.match(text):
        raise ValueError(f"not a Slurm time limit: {value!r}")
    days, sep, rest = text.partition("-")
    if not sep:
        days, rest = "0", text
    fields = [int(f) for f in rest.split(":")]
    if sep:  # D-H, D-H:M, D-H:M:S
        h, m, s = (fields + [0, 0])[:3]
        limits = [(h, 24, "hours")] + [(f, 60, n) for f, n in zip(fields[1:], ("minutes", "seconds"))]
    elif len(fields) == 1:  # M (any number of minutes)
        h, m, s = 0, fields[0], 0
        limits = []
    elif len(fields) == 2:  # M:S
        h, m, s = 0, fields[0], fields[1]
        limits = [(s, 60, "seconds")]
    else:
        h, m, s = fields
        limits = [(m, 60, "minutes"), (s, 60, "seconds")]
    for field_value, limit, what in limits:
        if field_value >= limit:
            raise ValueError(f"not a Slurm time limit: {value!r} ({what} must be below {limit})")
    return ((int(days) * 24 + h) * 60 + m) * 60 + s


def format_time(seconds: int) -> str:
    """`D-HH:MM:SS` (or `HH:MM:SS` under a day) for a number of seconds."""
    days, rest = divmod(int(seconds), 86400)
    h, rest = divmod(rest, 3600)
    m, s = divmod(rest, 60)
    hms = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{days}-{hms}" if days else hms


def parse_resources(text: str, *, where: str = "resources") -> dict[str, Any]:
    """Parse `"cpus=8,time=02:00:00,mem-per-cpu=3900M"`."""
    raw: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"{where}: expected KEY=VALUE, got {part!r}")
        raw[key.strip()] = value.strip()
    return normalize_resources(raw, where=where)


def normalize_split_by(split_by: str | Sequence[str] | None) -> list[str]:
    """`split_by` as a list of distinct `compare()` names (one name, or several)."""
    if split_by is None:
        return []
    names = [split_by] if isinstance(split_by, str) else list(split_by)
    out: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"split_by: expected compare() names, got {split_by!r}")
        if name.strip() not in out:
            out.append(name.strip())
    return out


def parse_stage_resources(text: str) -> tuple[str, dict[str, Any]]:
    """Parse `"solve:cpus=8,time=02:00:00"` into `("solve", {...})`."""
    stage, sep, rest = text.partition(":")
    if not sep or not stage.strip():
        raise ValueError(f"stage resources: expected STAGE:KEY=VALUE,..., got {text!r}")
    return stage.strip(), parse_resources(rest, where=f"resources of stage {stage.strip()!r}")


# ---------------------------------------------------------------------------- plan

@dataclass
class DagTask:
    id: int
    branch: int
    step: int
    stage: str
    check: str
    parents: list[int]
    resources: dict[str, Any]
    known_inputs: dict[str, Any] = field(default_factory=dict)
    mode: str = "use"
    repeats: int = 1
    warmup: int = 0
    split: tuple[str | None, ...] | None = None  # one label per split name; None without split names
    unit: int = -1
    explicit: dict[str, Any] | None = None  # the call's own `resources=`, for conflict checks
    users: list[int] = field(default_factory=list)  # branches making this request
    threads: int | None = None  # the request's env thread count (part of its request key)
    threads_floor: bool = False  # some call of it used `threads=`, which raises a default cpus count
    where: str = ""  # the call that gave `explicit` (or the first call), for error messages


@dataclass
class DagUnit:
    """One Slurm array element: a chain of tasks run back to back in one job (one task unfused).

    The chain runs in `branch`, its last task's branch, which calls every earlier member too;
    `steps` are the members' stage-call numbers in that branch. `time` is the members' sum."""

    id: int
    branch: int
    tasks: list[int]
    steps: list[int]
    stages: list[str]
    parents: list[int]
    resources: dict[str, Any]
    split: tuple[str | None, ...] | None
    group: int = -1


@dataclass
class DagGroup:
    id: int
    stage: str  # display label: the stage, or a bounded "a+b+c" label for fused chains
    stages: list[str]
    resources: dict[str, Any]
    split: tuple[str | None, ...] | None
    parents: list[int]
    units: list[int]


@dataclass
class DagPlan:
    branches: dict[int, tuple[_ChoiceStep, ...]]
    tasks: list[DagTask]
    units: list[DagUnit]
    groups: list[DagGroup]
    selection: dict[str, list[str]]
    split_by: list[str]
    cached_calls: int
    shared_calls: int
    fused: bool = True

    @property
    def leaves(self) -> list[int]:
        has_child = {p for g in self.groups for p in g.parents}
        return [g.id for g in self.groups if g.id not in has_child]

    def to_dict(self) -> dict[str, Any]:
        used = sorted({u.branch for u in self.units})
        return {
            "format_version": PLAN_FORMAT_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "selection": self.selection,
            "split_by": self.split_by,
            "fused": self.fused,
            "cached_calls": self.cached_calls,
            "shared_calls": self.shared_calls,
            "branches": {
                str(b): [[s.name, s.index, s.label, s.num_options] for s in self.branches[b]] for b in used
            },
            "tasks": [
                {
                    "id": t.id,
                    "branch": t.branch,
                    "step": t.step,
                    "stage": t.stage,
                    "check": t.check,
                    "known_inputs": t.known_inputs,
                    "mode": t.mode,
                    "repeats": t.repeats,
                    "warmup": t.warmup,
                    "parents": t.parents,
                    "resources": t.resources,
                    "split": _split_json(t.split),
                    "unit": t.unit,
                }
                for t in self.tasks
            ],
            "units": [
                {
                    "id": u.id,
                    "branch": u.branch,
                    "tasks": u.tasks,
                    "steps": u.steps,
                    "stages": u.stages,
                    "parents": u.parents,
                    "resources": u.resources,
                    "split": _split_json(u.split),
                    "group": u.group,
                }
                for u in self.units
            ],
            "groups": [
                {
                    "id": g.id,
                    "stage": g.stage,
                    "stages": g.stages,
                    "resources": g.resources,
                    "split": _split_json(g.split),
                    "parents": g.parents,
                    "units": g.units,
                }
                for g in self.groups
            ],
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        return out

    def group_lines(self) -> list[str]:
        """`id|size|parents|cpus|time|mem_per_cpu|stage|leaf` per array, in submission order.

        Empty parent lists are written as `-` so `IFS='|' read` keeps the columns aligned."""
        leaves = set(self.leaves)
        return [
            "|".join(
                [
                    str(g.id),
                    str(len(g.units)),
                    ",".join(map(str, g.parents)) or "-",
                    str(g.resources["cpus"]),
                    g.resources["time"],
                    g.resources["mem_per_cpu"],
                    g.stage,
                    "1" if g.id in leaves else "0",
                ]
            )
            for g in self.groups
        ]


def _split_json(split: tuple[str | None, ...] | None) -> list[str | None] | None:
    return list(split) if split is not None else None


def read_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != PLAN_FORMAT_VERSION:
        raise ValueError(f"{path}: not a DAG plan of format version {PLAN_FORMAT_VERSION}")
    return payload


def plan_dag(
    pipeline_fn: Callable[..., Any],
    *,
    select: Mapping[str, str | Sequence[str]] | None = None,
    resources: Mapping[str, Any] | None = None,
    stage_resources: Mapping[str, Mapping[str, Any]] | None = None,
    stage_resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    split_by: str | Sequence[str] | None = None,
    compare_splits: bool = True,
    max_array_size: int = DEFAULT_MAX_ARRAY_SIZE,
    fuse: bool = True,
    max_fused_time: str = DEFAULT_MAX_FUSED_TIME,
    **run_options: Any,
) -> DagPlan:
    """Plan the stage calls that still need to run as Slurm array groups.

    Resources merge key by key; from weakest to strongest: `DEFAULT_RESOURCES`, `resources` (for
    every task), `stage_resources[stage]`, a call's own `run_stage(..., resources=...)`, and
    `stage_resource_overrides[stage]` (the command line's `--stage-resources`). A call's
    `threads=N` then raises its cpus to at least N, unless the call's own resources or the
    override set cpus, which must then be at least N; a raw env `*_NUM_THREADS` above the cpus is
    refused. `split_by` names one or more `compare()`s whose options get separate arrays even
    when their resources match, so e.g. the coarse-mesh chain never waits for the fine one; with
    `compare_splits`, every `compare(..., split=True)` of the selected branches is added to them.
    With `fuse`, a chain of tasks where each one
    is the only (non-redundant) reader of the one before, with the same cpus and memory, runs as
    one array element, saving a scheduler round per link; its time limit is the members' sum
    (each rounded up to a minute, as Slurm does), and a chain stops growing before it would pass
    `max_fused_time`. Tasks without a time limit (`0`) aren't fused. `run_options` are `run_experiment`
    options (cache, executor, build_dir, cache_mode, stage_cache_modes, repeats, warmup, ...).
    """
    if isinstance(max_array_size, bool) or not isinstance(max_array_size, int) or max_array_size < 1:
        raise ValueError(f"max_array_size must be a positive integer, got {max_array_size!r}")
    fused_limit = time_seconds(str(max_fused_time))
    if fused_limit <= 0:
        raise ValueError(f"max_fused_time must be a positive time limit, got {max_fused_time!r}")
    unknown = set(run_options) - {
        "cache", "executor", "build_dir", "cache_mode", "stage_cache_modes",
        "repeats", "warmup", "stage_repeats", "stage_warmup", "continue_on_error",
    }
    if unknown:
        raise TypeError(f"plan_dag got unsupported option(s): {', '.join(sorted(unknown))}")
    run_options.pop("continue_on_error", None)
    defaults = dict(DEFAULT_RESOURCES)
    defaults.update(normalize_resources(resources or {}, where="default resources"))
    per_stage = {
        name: normalize_resources(r, where=f"resources of stage {name!r}") for name, r in (stage_resources or {}).items()
    }
    overrides = {
        name: normalize_resources(r, where=f"resource overrides of stage {name!r}")
        for name, r in (stage_resource_overrides or {}).items()
    }
    split_names = normalize_split_by(split_by)
    selection = _normalize_select(select)

    def resolve(task: DagTask) -> dict[str, Any]:
        """The task's resources by the precedence above, checked against its thread count; run
        once every call of its request was seen, so the result doesn't depend on branch order."""
        forced = overrides.get(task.stage, {})
        explicit = task.explicit or {}
        merged = {**defaults, **per_stage.get(task.stage, {}), **explicit, **forced}
        threads = task.threads
        if threads is not None and merged["cpus"] < threads:
            pinned = "cpus" in forced or "cpus" in explicit
            if task.threads_floor and not pinned:
                merged["cpus"] = threads
            else:
                source = ("--stage-resources" if "cpus" in forced else "its resources=" if pinned
                          else "the default resources")
                raise ValueError(
                    f"{task.where} asks for {threads} threads but gets cpus={merged['cpus']} from {source}; "
                    "give it at least as many cpus as threads"
                )
        return merged

    config = _make_config(dry_run=True, **run_options)
    takes_ctx = len(inspect.signature(pipeline_fn).parameters) > 0

    with _sigterm_as_exit():
        all_paths = _discover_branch_paths(pipeline_fn, takes_ctx=takes_ctx, config=config)
        selected = _select_paths(list(enumerate(all_paths)), selection)
        seen = {s.name for p in all_paths for s in p}  # a --select may leave a name out: then no label
        for name in split_names:
            if name not in seen:
                raise ValueError(
                    f"split_by: no compare() named {name!r} (choices: {', '.join(sorted(seen)) or 'none'})"
                )
        if compare_splits:
            for _, p in selected:
                for s in p:
                    if s.split and s.name not in split_names:
                        split_names.append(s.name)

        state = _ExperimentState()
        tasks: list[DagTask] = []
        task_of: dict[str, int] = {}
        steps_in: dict[int, dict[int, int]] = {}  # branch -> task -> its first stage call there
        read_cached: set[str] = set()
        cached = shared = 0
        with config.cache.lease() as lease:
            for index, path in selected:
                lease.begin_ephemeral_scope()
                todo: list[tuple[_ChoiceStep, ...]] = []
                ctx = RunContext._for_branch(config, path=path, todo_stack=todo, lease=lease, state=state)
                ctx._call_log = []
                _call_pipeline(pipeline_fn, ctx, takes_ctx, path)
                if todo:
                    raise RuntimeError(
                        f"Branch {index} reached compare() choices that discovery didn't see; "
                        "DAG mode needs choices that don't depend on stage outputs"
                    )
                for call in ctx._call_log:
                    where = f"branch {index}, stage call {call.step} ('{call.stage}')"
                    explicit = normalize_resources(call.resources, where=where) if call.resources else None
                    if call.outcome == "hit":
                        cached += 1
                        read_cached.add(call.request_key)
                        continue
                    if call.outcome == "reused":
                        shared += 1
                        task = tasks[task_of[call.request_key]]
                        steps_in.setdefault(index, {}).setdefault(task.id, call.step)
                        forced = overrides.get(task.stage, {})
                        if explicit is not None and task.explicit is not None and any(
                            explicit.get(k) != task.explicit.get(k)
                            for k in set(explicit) | set(task.explicit) if k not in forced
                        ):
                            raise ValueError(
                                f"{where} asks for resources {explicit} but the same request in branch "
                                f"{task.branch} asks for {task.explicit}; give both calls the same resources "
                                "(or set the differing ones with --stage-resources)"
                            )
                        if task.explicit is None and explicit is not None:
                            task.explicit, task.where = explicit, where
                        task.threads_floor = task.threads_floor or call.threads_floor
                        if index not in task.users:
                            task.users.append(index)
                        continue
                    if not call.writes:
                        raise ValueError(
                            f"{where} would run with a cache mode that doesn't write, so later tasks and "
                            "the collect job could never see its output; use cache mode 'use' or 'refresh'"
                        )
                    if call.request_key in read_cached:
                        raise ValueError(
                            f"{where} refreshes a request that an earlier call already read from the cache; "
                            "its new output could break what was built on the old one. Refresh it on its "
                            "first use, or not at all, in a DAG sweep"
                        )
                    task = DagTask(
                        id=len(tasks),
                        branch=index,
                        step=call.step,
                        stage=call.stage,
                        check=call.check,
                        known_inputs=call.known_inputs,
                        mode=call.mode,
                        repeats=call.repeats,
                        warmup=call.warmup,
                        parents=[task_of[k] for k in call.parents],
                        resources={},
                        explicit=explicit,
                        users=[index],
                        threads=call.threads,
                        threads_floor=call.threads_floor,
                        where=where,
                    )
                    task_of[call.request_key] = task.id
                    tasks.append(task)
                    steps_in.setdefault(index, {}).setdefault(task.id, call.step)

    for task in tasks:
        task.resources = resolve(task)

    paths = dict(selected)
    if split_names:
        for task in tasks:
            split: list[str | None] = []
            for name in split_names:
                labels = {next((s.label for s in paths[b] if s.name == name), None) for b in task.users}
                split.append(labels.pop() if len(labels) == 1 else None)
            task.split = tuple(split)

    units = _fuse_chains(tasks, steps_in, max_seconds=fused_limit) if fuse else [
        _make_unit(i, [t], t.branch, [t.step]) for i, t in enumerate(tasks)
    ]
    for unit in units:
        for t in unit.tasks:
            tasks[t].unit = unit.id
    for unit in units:
        unit.parents = sorted({tasks[p].unit for t in unit.tasks for p in tasks[t].parents} - {unit.id})
        if any(p >= unit.id for p in unit.parents):  # defensive: units must stay topologically ordered
            raise RuntimeError(f"DAG fusion produced unit {unit.id} that depends on a later unit {unit.parents}")

    groups: list[DagGroup] = []
    open_group: dict[tuple[Any, ...], int] = {}
    for unit in units:  # topological order: parents always come first
        parent_groups = sorted({units[p].group for p in unit.parents})
        key = (tuple(unit.stages), tuple(sorted(unit.resources.items())), unit.split, tuple(parent_groups))
        gid = open_group.get(key)
        if gid is None or len(groups[gid].units) >= max_array_size:
            gid = len(groups)
            groups.append(
                DagGroup(
                    id=gid,
                    stage=group_label(unit.stages),
                    stages=list(unit.stages),
                    resources=dict(unit.resources),
                    split=unit.split,
                    parents=parent_groups,
                    units=[],
                )
            )
            open_group[key] = gid
        groups[gid].units.append(unit.id)
        unit.group = gid

    return DagPlan(
        branches={b: paths[b] for b in sorted({u.branch for u in units})},
        tasks=tasks,
        units=units,
        groups=groups,
        selection=selection,
        split_by=split_names,
        cached_calls=cached,
        shared_calls=shared,
        fused=fuse,
    )


_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
MAX_LABEL_LENGTH = 48


def group_label(stages: Sequence[str]) -> str:
    """A filename- and job-name-safe label for an array: `solve`, `assemble+solve`, or, for long
    chains, `first+N+last` (N stages in between), at most `MAX_LABEL_LENGTH` characters."""
    names = [_LABEL_UNSAFE.sub("_", s) or "_" for s in stages]
    label = "+".join(names)
    if len(names) > 3 or len(label) > MAX_LABEL_LENGTH:
        ends = [names[0], names[-1]] if len(names) > 1 else names
        middle = f"+{len(names) - 2}+" if len(names) > 2 else "+"
        budget = MAX_LABEL_LENGTH - (len(middle) if len(ends) == 2 else 0)
        if sum(map(len, ends)) > budget:  # shorten each end to fit, keeping both visible
            ends = [e[: max(1, budget // len(ends))] for e in ends]
        label = middle.join(ends)
    return label[:MAX_LABEL_LENGTH]


def _billed_seconds(task: DagTask) -> int:
    """A task's time limit as Slurm applies it to a job: rounded up to whole minutes."""
    return -(-time_seconds(task.resources["time"]) // 60) * 60


def _make_unit(uid: int, members: list[DagTask], branch: int, steps: list[int]) -> DagUnit:
    if len(members) == 1:  # unfused: keep the limit exactly as given (e.g. 0 for no limit)
        resources = dict(members[0].resources)
    else:
        resources = {**members[0].resources, "time": format_time(sum(_billed_seconds(t) for t in members))}
    return DagUnit(
        id=uid,
        branch=branch,
        tasks=[t.id for t in members],
        steps=steps,
        stages=[t.stage for t in members],
        parents=[],
        resources=resources,
        split=members[-1].split,
    )


def _fuse_chains(
    tasks: list[DagTask], steps_in: Mapping[int, Mapping[int, int]], *, max_seconds: int
) -> list[DagUnit]:
    """Group tasks into chains run back to back in one job.

    Works on the transitive reduction of the task graph: a parent that is also an ancestor of
    another parent adds no waiting, so `case_mesh`-style skip edges don't block fusion. Task `c`
    joins its parent `t`'s chain when, in that reduction, `t` is `c`'s only parent and `c` is
    `t`'s only child, both need the same cpus and memory, `c`'s branch calls every member in chain
    order (it always does for a pipeline that doesn't share artifacts between branches), and the
    summed time limit stays within `max_seconds`. Tasks without a time limit (`0`) stay alone.
    """
    ancestors: list[int] = []  # bitmask per task; task ids are a topological order
    reduced_parents: list[list[int]] = []
    for task in tasks:
        mask = 0
        for p in task.parents:
            mask |= ancestors[p] | (1 << p)
        ancestors.append(mask)
        direct = [p for p in task.parents if not any(q != p and (ancestors[q] >> p) & 1 for q in task.parents)]
        reduced_parents.append(direct)
    reduced_children: list[list[int]] = [[] for _ in tasks]
    for task in tasks:
        for p in reduced_parents[task.id]:
            reduced_children[p].append(task.id)

    def can_join(chain: list[int], task: DagTask) -> bool:
        members = [tasks[t] for t in chain] + [task]
        if any(m.resources[k] != task.resources[k] for m in members for k in ("cpus", "mem_per_cpu")):
            return False
        if any(time_seconds(m.resources["time"]) == 0 for m in members):
            return False
        if sum(_billed_seconds(m) for m in members) > max_seconds:
            return False
        steps = steps_in.get(task.branch, {})
        if not all(m.id in steps for m in members):
            return False
        order = [steps[m.id] for m in members]
        return all(a < b for a, b in zip(order, order[1:]))  # replay runs them in chain order

    chains: list[list[int]] = []
    chain_of: dict[int, int] = {}
    for task in tasks:
        parents = reduced_parents[task.id]
        if len(parents) == 1 and reduced_children[parents[0]] == [task.id]:
            chain = chains[chain_of[parents[0]]]
            if can_join(chain, task):
                chain.append(task.id)
                chain_of[task.id] = chain_of[parents[0]]
                continue
        chain_of[task.id] = len(chains)
        chains.append([task.id])

    units: list[DagUnit] = []
    for uid, chain in enumerate(chains):  # chains start in topological order of their first task
        branch = tasks[chain[-1]].branch
        steps = [steps_in[branch][t] for t in chain]
        units.append(_make_unit(uid, [tasks[t] for t in chain], branch, steps))
    return units


# ---------------------------------------------------------------------------- one task

@dataclass(frozen=True)
class DagTaskResult:
    """What one array element did: its tasks' stages and traces, in chain order."""

    unit: int
    branch: int
    tasks: tuple[int, ...]
    stages: tuple[str, ...]
    traces: tuple[StageTrace, ...]

    @property
    def stage(self) -> str:
        return "+".join(self.stages)

    @property
    def status(self) -> str:
        """`HIT` if every stage was cached already, otherwise `MISS` (something executed)."""
        return "HIT" if all(t.cached for t in self.traces) else "MISS"


def run_dag_task(
    pipeline_fn: Callable[..., Any],
    plan: Mapping[str, Any] | str | os.PathLike[str],
    group: int,
    element: int,
    **run_options: Any,
) -> DagTaskResult:
    """Run element `element` of array `group`: execute its stage call, or its chain of calls when
    fused, back to back (each one found cached counts as done).

    Every other stage call of the branch only looks up the cache, so running this needs exactly
    the element's parent arrays to be done. Raises `RuntimeError` when the experiment no longer
    matches the plan (edited, rebuilt, or its shape depends on stage outputs), an input is
    missing, or a stage of the chain failed and the pipeline carried on without it.
    """
    payload = read_plan(plan) if not isinstance(plan, Mapping) else plan
    if payload.get("format_version") != PLAN_FORMAT_VERSION:
        raise ValueError(f"not a DAG plan of format version {PLAN_FORMAT_VERSION}")
    groups = payload["groups"]
    if not (0 <= group < len(groups)):
        raise ValueError(f"DAG plan has {len(groups)} arrays; there is no array {group}")
    members = groups[group]["units"]
    if not (0 <= element < len(members)):
        raise ValueError(f"DAG array {group} has {len(members)} tasks; there is no task {element}")
    unit = payload["units"][members[element]]
    chain = [payload["tasks"][t] for t in unit["tasks"]]
    steps = [int(s) for s in unit["steps"]]
    branch = int(unit["branch"])
    label = f"DAG task {group}/{element} ('{'+'.join(unit['stages'])}')"
    path = tuple(_ChoiceStep(str(n), int(i), str(lbl), int(k)) for n, i, lbl, k in payload["branches"][str(branch)])
    run_options.pop("continue_on_error", None)
    config = _make_config(dry_run=True, **run_options)
    takes_ctx = len(inspect.signature(pipeline_fn).parameters) > 0

    with _sigterm_as_exit(), config.cache.lease() as lease:
        ctx = RunContext._for_branch(
            config, path=path, todo_stack=[], lease=lease, state=_ExperimentState()
        )
        ctx._dag_targets = {
            step: _DagTarget(
                step=step,
                stage=str(task["stage"]),
                check=str(task["check"]),
                known_inputs=dict(task["known_inputs"]),
                mode=CacheMode(task["mode"]),
                repeats=int(task["repeats"]),
                warmup=int(task["warmup"]),
            )
            for step, task in zip(steps, chain)
        }
        try:
            _call_pipeline(pipeline_fn, ctx, takes_ctx, path)
        except _DagTaskDone as stop:
            missing = [f"'{t['stage']}' (call {s})" for s, t in zip(steps, chain) if s not in stop.done]
            if missing:
                raise RuntimeError(
                    f"{label}: stage {missing[0]} of this chain didn't finish, but the pipeline carried on "
                    "(an `except` around a stage call?); the rest of the chain was not run and the task "
                    "doesn't count as done"
                ) from None
            traces = tuple(stop.done[s] for s in steps)
            for trace in traces:
                if trace.dry_run or not (trace.cached or trace.request_key):
                    raise RuntimeError(  # defensive: each stage must end up published in the cache
                        f"{label}: stage '{trace.stage}' finished without a cached output"
                    ) from None
            return DagTaskResult(
                unit=int(unit["id"]),
                branch=branch,
                tasks=tuple(int(t) for t in unit["tasks"]),
                stages=tuple(str(t["stage"]) for t in chain),
                traces=traces,
            )
    raise RuntimeError(
        f"Plan mismatch: branch {branch} made only {ctx._stage_calls} stage calls, the plan expected "
        f"'{chain[-1]['stage']}' as call {steps[-1]}. The experiment changed since it was planned, or its "
        "shape depends on stage outputs; plan it again or use pieces (shards)."
    )
