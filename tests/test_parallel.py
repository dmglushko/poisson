"""Tests for parallel local runs: `run_experiment(cores=N)` and `experiment_main(cores=, --cores)`."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import (  # noqa: E402
    Cache,
    LocalExecutor,
    RunContext,
    Stage,
    compare,
    evaluate,
    experiment_main,
    load_run,
    run_experiment,
)
from poisson_runner import driver as driver_module  # noqa: E402
from poisson_runner import experiment_cli  # noqa: E402
from poisson_runner.executor import ExecutionRequest, StageCancelled  # noqa: E402

# A stage program: records when it ran, its pid and thread env, sleeps, optionally fails, and
# writes `<input> > <tag>` as its output. Its info `status` is "done" once the input chain is
# `--done-at` links long (an AMR-style loop reads it). `--spawn true` starts a long-lived child
# process first (its pid goes to the start log), like a wrapper script or MPI launcher.
STAGE_PROGRAM = """
import argparse, json, os, pathlib, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--sleep", type=float, default=0.0)
p.add_argument("--fail", default="false")
p.add_argument("--spawn", default="false")
p.add_argument("--done-at", type=int, default=0)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
start = time.time()
log = os.environ["TEST_STAGE_LOG"]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) if a.spawn == "true" else None
with open(log + ".start", "a") as f:
    f.write(json.dumps({"tag": a.tag, "pid": os.getpid(), "child": child.pid if child else None}) + "\\n")
time.sleep(a.sleep)
if a.fail == "true":
    sys.exit("failing on purpose")
src = pathlib.Path(a.inp).read_text() + ">" if a.inp else ""
pathlib.Path(a.out).write_text(src + a.tag)
depth = (src + a.tag).count(">")
status = "done" if a.done_at and depth + 1 >= a.done_at else "running"
pathlib.Path(a.info_out).write_text(json.dumps({"time": time.time() - start, "status": status}))
with open(log, "a") as f:
    f.write(json.dumps({"stage": NAME, "tag": a.tag, "start": start, "end": time.time(), "pid": os.getpid(),
                        "omp": os.environ.get("OMP_NUM_THREADS"),
                        "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "mkl": os.environ.get("MKL_NUM_THREADS")}) + "\\n")
"""


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _max_overlap(records: list[dict[str, Any]]) -> int:
    """Most records whose [start, end] intervals overlap at one moment."""
    events = sorted([(r["start"], 1) for r in records] + [(r["end"], -1) for r in records], key=lambda e: (e[0], e[1]))
    now = best = 0
    for _, delta in events:
        now += delta
        best = max(best, now)
    return best


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _wait_until(pred: Any, timeout: float = 10.0) -> bool:
    """Poll `pred()` until it holds (a condition check, not a timing assumption)."""
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _is_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # Killed but not yet reaped by its new parent (init) counts as dead.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1].startswith("Z")
    except OSError:
        return True


class _WaiterGate:
    """Counts branches that wait for another branch's in-flight request (patches
    `_LocalScheduler.wait_for`), so a test can hold a stage until its waiters are there."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.waiting = 0

    def wait_for_waiters(self, n: int, timeout: float = 10.0) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.waiting >= n, timeout)

    @contextlib.contextmanager
    def patch(self) -> Any:
        real = driver_module._LocalScheduler.wait_for
        gate = self

        def wait_for(sched: Any, flight: Any, stage_name: str, request_key: str) -> None:
            with gate.cond:
                gate.waiting += 1
                gate.cond.notify_all()
            return real(sched, flight, stage_name, request_key)

        with mock.patch.object(driver_module._LocalScheduler, "wait_for", wait_for):
            yield self


class _HookedExecutor(LocalExecutor):
    """A `LocalExecutor` that calls `hooks[stage_name](request)` before running a stage and
    counts the runs per stage."""

    def __init__(self, hooks: dict[str, Any]) -> None:
        super().__init__()
        self.hooks = hooks
        self.calls: dict[str, int] = {}
        self._count_lock = threading.Lock()

    def run(self, request: ExecutionRequest) -> Any:
        with self._count_lock:
            self.calls[request.stage_name] = self.calls.get(request.stage_name, 0) + 1
        hook = self.hooks.get(request.stage_name)
        if hook is not None:
            hook(request)
        return super().run(request)


class _Env:
    """Temporarily set (value) or unset (None) environment variables."""

    def __init__(self, **values: str | None) -> None:
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for k, v in self.values.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc: Any) -> None:
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ParallelTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_parallel_test_")
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "stages.log"
        self._env = _Env(
            TEST_STAGE_LOG=str(self.log), SLURM_JOB_ID=None, SLURM_ARRAY_TASK_ID=None, POISSON_CACHE_LOCK_MODE=None,
            OMP_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None, MKL_NUM_THREADS=None,
        )
        self._env.__enter__()
        self.cache = Cache(self.tmp / "cache")
        self.mesh = self._stage("mesh")
        self.assemble = self._stage("assemble")
        self.solve = self._stage("solve")

    def tearDown(self) -> None:
        self._env.__exit__()
        self._tmp.cleanup()

    def _stage(self, name: str) -> Stage:
        binary = _write_executable(self.tmp / f"{name}_bin", f"NAME = {name!r}\n" + STAGE_PROGRAM)
        return Stage.from_cli(
            name,
            binary=binary,
            input_flags={"inp": "--in"},
            setting_flags={"tag": "--tag", "sleep": "--sleep", "fail": "--fail", "done_at": "--done-at",
                           "spawn": "--spawn"},
            output_flags={"out": "--out"},
            output_specs={"out": ".txt"},
            info_flag="--info-out",
        )

    def records(self, stage: str | None = None) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]
        return [r for r in rows if stage is None or r["stage"] == stage]

    def assert_clean(self, cache: Cache | None = None, *, ephemeral: bool = True) -> None:
        cache = cache or self.cache
        self.assertEqual(list(cache.tmp_dir.iterdir()), [], "staging files left behind")
        if ephemeral:  # a successful cache="off" run keeps its outputs there, as sequential runs do
            self.assertEqual(list(cache.ephemeral_dir.iterdir()), [], "ephemeral files left behind")
        self.assertEqual(list(cache.leases_dir.iterdir()), [], "leases left behind")


class CoreBudgetTests(ParallelTestBase):
    def _sweep(self, n: int = 4, sleep: float = 0.4, **solve_options: Any):
        def pipe(run: RunContext) -> dict[str, Any]:
            k = compare("k", [f"b{i}" for i in range(n)])
            sol = run.run_stage(self.solve, settings={"tag": k, "sleep": sleep}, **solve_options)
            return {"solution": sol}

        return pipe

    def test_one_core_stages_run_in_parallel_and_keep_branch_order(self) -> None:
        pipe = self._sweep(n=6)
        report = run_experiment(pipe, cache=self.cache, cores=3)
        self.assertEqual([b.choice_labels["k"] for b in report.branches], [f"b{i}" for i in range(6)])
        self.assertEqual([b.index for b in report.branches], list(range(6)))
        self.assertTrue(all(b.ok for b in report.branches))
        recs = self.records("solve")
        self.assertEqual(len(recs), 6)
        self.assertGreaterEqual(_max_overlap(recs), 2)
        self.assertLessEqual(_max_overlap(recs), 3)
        self.assertEqual({r["omp"] for r in recs}, {"1"})  # default cpus=1: one runtime thread each
        self.assert_clean()

        # The same run sequentially: all hits, identical request keys and report shape.
        seq = run_experiment(pipe, cache=self.cache)
        self.assertEqual(seq.cache_hits, 6)
        self.assertEqual([s.request_key for b in seq.branches for s in b.stages],
                         [s.request_key for b in report.branches for s in b.stages])

    def test_multi_core_stage_runs_one_at_a_time_with_its_thread_count(self) -> None:
        report = run_experiment(self._sweep(n=3, sleep=0.3), cache=self.cache, cores=8,
                                stage_resources={"solve": {"cpus": 8}})
        self.assertTrue(all(b.ok for b in report.branches))
        recs = self.records("solve")
        self.assertEqual(_max_overlap(recs), 1)
        self.assertEqual({(r["omp"], r["blas"], r["mkl"]) for r in recs}, {("8", "8", "8")})

    def test_cpus_above_cores_are_clamped_and_never_deadlock(self) -> None:
        report = run_experiment(self._sweep(n=3, sleep=0.1), cache=self.cache, cores=4,
                                resources={"cpus": 16})
        self.assertTrue(all(b.ok for b in report.branches))
        recs = self.records("solve")
        self.assertEqual(_max_overlap(recs), 1)
        self.assertEqual({r["omp"] for r in recs}, {"4"})

    def test_two_core_stages_share_the_budget(self) -> None:
        report = run_experiment(self._sweep(n=4, sleep=0.5), cache=self.cache, cores=4,
                                stage_resources={"solve": {"cpus": 2}})
        self.assertTrue(all(b.ok for b in report.branches))
        recs = self.records("solve")
        self.assertEqual(_max_overlap(recs), 2)
        self.assertEqual({r["omp"] for r in recs}, {"2"})

    def test_exclusive_stages_run_alone_but_keep_their_thread_count(self) -> None:
        def pipe(run: RunContext) -> dict[str, Any]:
            k = compare("k", ["a", "b", "c", "d"])
            msh = run.run_stage(self.mesh, settings={"tag": k, "sleep": 0.3})
            sol = run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": k, "sleep": 0.3},
                                resources={"cpus": 2})
            return {"solution": sol}

        for options in (dict(exclusive_stages=["solve"]), dict(exclusive_stages={"solve"})):
            with self.subTest(**{k: sorted(v) for k, v in options.items()}):
                self.log.unlink(missing_ok=True)
                report = run_experiment(pipe, cache=self.cache, cache_mode="refresh", cores=4, **options)
                self.assertTrue(all(b.ok for b in report.branches))
                solves, meshes = self.records("solve"), self.records("mesh")
                self.assertEqual({r["omp"] for r in solves}, {"2"})  # exclusive, but still 2 threads
                self.assertGreaterEqual(_max_overlap(meshes), 2)  # the 1-core stages run side by side
                for s in solves:
                    self.assertFalse(any(_overlaps(s, o) for o in solves + meshes if o is not s), s)

        # The per-call flag does the same.
        def pipe_flag(run: RunContext) -> None:
            k = compare("k", ["a", "b", "c"])
            run.run_stage(self.mesh, settings={"tag": k, "sleep": 0.3})
            run.run_stage(self.solve, settings={"tag": k, "sleep": 0.3}, exclusive=True)

        self.log.unlink()
        report = run_experiment(pipe_flag, cache=self.cache, cores=3)
        self.assertTrue(all(b.ok for b in report.branches))
        solves, meshes = self.records("solve"), self.records("mesh")
        self.assertEqual({r["omp"] for r in solves}, {"1"})
        for s in solves:
            self.assertFalse(any(_overlaps(s, o) for o in solves + meshes if o is not s), s)

    def test_runtime_thread_env_stays_out_of_the_request_key(self) -> None:
        def pipe(run: RunContext) -> None:
            k = compare("k", ["a", "b"])
            run.run_stage(self.mesh, settings={"tag": k}, resources={"cpus": 2})
            run.run_stage(self.solve, settings={"tag": k}, threads=3)
            run.run_stage(self.assemble, settings={"tag": k}, env={"OMP_NUM_THREADS": "1"}, resources={"cpus": 2})

        par = run_experiment(pipe, cache=self.cache, cores=4)
        self.assertEqual(par.cache_misses, 6)
        by = {(r["stage"], r["tag"]): r for r in self.records()}
        self.assertEqual({(r["omp"], r["blas"]) for (s, _), r in by.items() if s == "mesh"}, {("2", "2")})
        self.assertEqual({(r["omp"], r["mkl"]) for (s, _), r in by.items() if s == "solve"}, {("3", "3")})
        # The call's own env wins for its key; the other keys get the runtime default.
        self.assertEqual({(r["omp"], r["blas"]) for (s, _), r in by.items() if s == "assemble"}, {("1", "2")})

        seq = run_experiment(pipe, cache=self.cache)  # sequential: every request is a hit
        self.assertEqual((seq.cache_hits, seq.cache_misses), (6, 0))
        for entry in self.cache.inspect():
            if entry["stage"] == "mesh":
                self.assertNotIn("env", json.dumps(entry["settings"]))
        # And the request key of a call without env equals the plain key (no env part).
        mesh_trace = seq.branches[0].stages[0]
        self.assertEqual(
            mesh_trace.request_key,
            Cache.compute_request_key("mesh", self.mesh.compute_code_hash(self.cache), {}, {"tag": "a"},
                                      output_specs=self.mesh.output_specs),
        )

    def test_cores_one_and_dry_run_stay_sequential(self) -> None:
        pipe = self._sweep(n=3, sleep=0.2)
        dry = run_experiment(pipe, cache=self.cache, cores=4, dry_run=True)
        self.assertEqual(dry.cache_misses, 3)
        self.assertEqual(self.records(), [])
        run_experiment(pipe, cache=self.cache, cores=1)
        recs = self.records("solve")
        self.assertEqual(_max_overlap(recs), 1)
        self.assertEqual({r["omp"] for r in recs}, {None})  # no runtime thread env when sequential

    def test_select_and_max_branches_with_cores(self) -> None:
        pipe = self._sweep(n=5, sleep=0.0)
        report = run_experiment(pipe, cache=self.cache, cores=2, select={"k": ["b1", "b3", "b4"]}, max_branches=2)
        self.assertEqual([b.index for b in report.branches], [1, 3])

    def test_validation(self) -> None:
        pipe = self._sweep(n=1, sleep=0.0)
        for bad in (0, -1, True, 2.0, "2"):
            with self.subTest(cores=bad), self.assertRaisesRegex(ValueError, "cores"):
                run_experiment(pipe, cache=self.cache, cores=bad)
        for bad in ("solve", b"solve", ["solve", ""], [3], 5):
            with self.subTest(exclusive_stages=bad), self.assertRaisesRegex(ValueError, "exclusive_stages"):
                run_experiment(pipe, cache=self.cache, cores=2, exclusive_stages=bad)
        with self.assertRaisesRegex(ValueError, "unknown resource"):
            run_experiment(pipe, cache=self.cache, resources={"gpus": 1})
        with self.assertRaisesRegex(ValueError, "cpus"):
            run_experiment(pipe, cache=self.cache, stage_resources={"solve": {"cpus": 0}})
        with self.assertRaisesRegex(ValueError, "dict"):
            run_experiment(pipe, cache=self.cache, stage_resources={"solve": 4})
        for shard in ((0, 2), (0, 1)):
            with self.subTest(shard=shard), self.assertRaisesRegex(ValueError, "cores > 1"):
                run_experiment(pipe, cache=self.cache, cores=2, shard=shard)
        with self.assertRaisesRegex(TypeError, "exclusive"):
            run_experiment(lambda run: run.run_stage(self.solve, settings={"tag": "x"}, exclusive=1),
                           cache=self.cache)
        # A bad per-call resource is refused in a parallel run (it fails that branch).
        bad_call = lambda run: run.run_stage(self.solve, settings={"tag": "x"}, resources={"cpus": "many"})  # noqa: E731
        with self.assertRaisesRegex(ValueError, "cpus must be a positive integer"):
            run_experiment(bad_call, cache=self.cache, cores=2)

    def test_thread_counts_above_cores_are_refused_on_cold_and_warm_caches(self) -> None:
        calls = {
            "threads=": dict(threads=4),
            "env": dict(env={"OMP_NUM_THREADS": "4"}),
            "env and cpus": dict(env={"OPENBLAS_NUM_THREADS": "4"}, resources={"cpus": 4}),
        }
        for name, options in calls.items():
            with self.subTest(name):
                def pipe(run: RunContext, options: dict[str, Any] = options) -> None:
                    run.run_stage(self.solve, settings={"tag": "t"}, **options)

                before = len(self.records())
                with self.assertRaisesRegex(ValueError, "asks for 4 threads.*cores=2; raise cores"):
                    run_experiment(pipe, cache=self.cache, cores=2)
                self.assertEqual(len(self.records()), before)  # refused before it ran
                run_experiment(pipe, cache=self.cache)  # sequentially it runs and is cached
                with self.assertRaisesRegex(ValueError, "asks for 4 threads.*cores=2; raise cores"):
                    run_experiment(pipe, cache=self.cache, cores=2)  # a cache hit is refused the same way
                self.assertEqual(len(self.records()), before + 1)  # only the sequential run executed
        # Enough cores: fine, and the warm entry is a hit.
        report = run_experiment(lambda run: run.run_stage(self.solve, settings={"tag": "t"}, threads=4),
                                cache=self.cache, cores=4)
        self.assertEqual(report.cache_hits, 1)

    def test_caller_context_vars_reach_branches(self) -> None:
        var: contextvars.ContextVar[str] = contextvars.ContextVar("poisson_test_var", default="unset")
        seen: list[str] = []

        def pipe(run: RunContext) -> None:
            k = compare("k", ["a", "b", "c"])
            if not run.dry_run:
                seen.append(var.get())
            run.run_stage(self.solve, settings={"tag": k})

        token = var.set("caller")
        try:
            run_experiment(pipe, cache=self.cache, cores=2)
        finally:
            var.reset(token)
        self.assertEqual(seen, ["caller"] * 3)

    def test_max_branches_zero_runs_nothing(self) -> None:
        pipe = self._sweep(n=3, sleep=0.0)
        for limit in (0, -1):
            for options in (dict(cores=2), dict(select={"k": "b1"}), dict(shard=(0, 2)), dict()):
                with self.subTest(limit=limit, **{k: repr(v) for k, v in options.items()}):
                    self.assertEqual(run_experiment(pipe, cache=self.cache, max_branches=limit, **options).branches, [])
        self.assertEqual(self.records(), [])

    def test_cpus_resolution_follows_dag_precedence(self) -> None:
        sched = driver_module._LocalScheduler(
            8, resources={"cpus": 2}, stage_resources={"solve": {"cpus": 3}}, exclusive_stages=frozenset({"big"}),
            executor=LocalExecutor(),
        )
        cost = lambda stage, resources=None, env_threads=None, floor=False, exclusive=False: (  # noqa: E731
            RunContext._local_cost(sched, dataclasses.replace(self.solve, name=stage), resources, env_threads,
                                   floor, exclusive))
        self.assertEqual(cost("mesh"), (2, 2))  # resources
        self.assertEqual(cost("solve"), (3, 3))  # stage_resources over resources
        self.assertEqual(cost("solve", {"cpus": 4}), (4, 4))  # per call over both
        self.assertEqual(cost("solve", {"cpus": 16}), (8, 8))  # clamped to the budget
        self.assertEqual(cost("solve", None, 6, True), (6, 6))  # threads= raises the cpus
        self.assertEqual(cost("solve", {"cpus": 6}, 6, True), (6, 6))
        self.assertEqual(cost("solve", exclusive=True), (8, 3))  # all seats, own thread count
        self.assertEqual(cost("big"), (8, 2))
        with self.assertRaisesRegex(ValueError, "asks for 4 threads but resources cpus=3"):
            cost("solve", None, 4, False)  # a raw env thread count above the configured cpus
        free = driver_module._LocalScheduler(
            8, resources={}, stage_resources={}, exclusive_stages=frozenset(), executor=LocalExecutor())
        self.assertEqual(RunContext._local_cost(free, self.solve, None, 4, False, False), (4, 4))  # no cpus declared: the env's count

    def test_evaluate_with_cores(self) -> None:
        pipe = self._sweep(n=3, sleep=0.0)
        report = run_experiment(pipe, cache=self.cache)
        saved = report.save(self.tmp / "run")
        self.log.unlink()

        def fn(run: RunContext, branch: Any) -> dict[str, Any]:
            out = run.run_stage(self.assemble, inputs={"inp": branch["solution"]},
                                settings={"tag": "eval", "sleep": 0.4})
            return {"evaluated": out}

        ev = evaluate(load_run(saved, cache=self.cache), fn, cores=3)
        self.assertEqual([b.choice_labels["k"] for b in ev.branches], ["b0", "b1", "b2"])
        self.assertTrue(all(b.ok for b in ev.branches))
        self.assertGreaterEqual(_max_overlap(self.records("assemble")), 2)


class InFlightTests(ParallelTestBase):
    def _shared_prefix(self, *, mesh_fail: bool = False, mesh_cache: str | None = None, meshes=("A",)):
        def pipe(run: RunContext) -> dict[str, Any]:
            m = compare("mesh", list(meshes))
            extra = {"cache": mesh_cache} if mesh_cache else {}
            msh = run.run_stage(self.mesh, settings={"tag": m, "sleep": 0.5, "fail": mesh_fail}, **extra)
            k = compare("k", [1, 2, 3, 4])
            sol = run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": f"{m}{k}", "sleep": 0.2})
            return {"solution": sol}

        return pipe

    def test_shared_stage_runs_once_and_waiters_reuse_it(self) -> None:
        for mode in (None, "refresh", "off"):
            with self.subTest(mode=mode):
                self.log.unlink(missing_ok=True)
                report = run_experiment(self._shared_prefix(mesh_cache=mode), cache=self.cache, cores=4,
                                        cache_mode="refresh" if mode is None else None,
                                        stage_cache_modes={"solve": "refresh"})
                self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
                self.assertEqual(len(self.records("mesh")), 1)
                statuses = sorted(b.stages[0].status for b in report.branches)
                self.assertEqual(statuses, ["MISS", "REUSED", "REUSED", "REUSED"])
                self.assertEqual(len(self.records("solve")), 4)
                self.assert_clean(ephemeral=mode != "off")

    def test_waiting_branches_hold_no_cores(self) -> None:
        # Mesh A (branches 0-3) is in flight; the waiters must not stop mesh B from starting.
        report = run_experiment(self._shared_prefix(meshes=("A", "B")), cache=self.cache, cores=2)
        self.assertTrue(all(b.ok for b in report.branches))
        meshes = self.records("mesh")
        self.assertEqual(len(meshes), 2)
        self.assertTrue(_overlaps(meshes[0], meshes[1]), meshes)

    def _gated_mesh(self, waiters: int) -> tuple[_WaiterGate, _HookedExecutor]:
        """An executor that holds the mesh stage until `waiters` branches wait for it."""
        gate = _WaiterGate()
        return gate, _HookedExecutor({"mesh": lambda _req: self.assertTrue(gate.wait_for_waiters(waiters))})

    def test_failed_shared_stage_reaches_every_waiter(self) -> None:
        gate, executor = self._gated_mesh(3)
        with gate.patch():
            report = run_experiment(self._shared_prefix(mesh_fail=True), cache=self.cache, cores=4,
                                    continue_on_error=True, executor=executor)
        self.assertEqual(executor.calls, {"mesh": 1})  # ran once
        errors = [b.error for b in report.branches]
        self.assertTrue(all(errors), errors)
        self.assertEqual(sum("failing on purpose" in e and "another branch" not in e for e in errors), 1)
        self.assertEqual(sum("failed in another branch" in e for e in errors), 3)
        self.assert_clean()

        # Without continue_on_error the first error is raised and nothing is left behind.
        gate, executor = self._gated_mesh(3)
        with gate.patch(), self.assertRaisesRegex(RuntimeError, "failing on purpose"):
            run_experiment(self._shared_prefix(mesh_fail=True), cache=self.cache, cores=4, executor=executor)
        self.assertEqual(executor.calls, {"mesh": 1})
        self.assert_clean()

    def test_fail_fast_raises_the_producer_error_and_never_retries(self) -> None:
        gate, executor = self._gated_mesh(3)
        woke = threading.Event()
        real_wait = driver_module._LocalScheduler.wait_for

        def wait_for(sched: Any, *args: Any) -> None:
            try:
                real_wait(sched, *args)
            finally:
                woke.set()

        def pipe(run: RunContext) -> None:
            k = compare("k", [1, 2, 3, 4])
            try:
                run.run_stage(self.mesh, settings={"tag": "shared", "fail": True})
            except RuntimeError:
                # The producer reports its error only after a waiter woke up: a waiter's own error
                # must not win the race, and none may run the stage again.
                woke.wait(10)
                raise
            run.run_stage(self.solve, settings={"tag": str(k)})

        with mock.patch.object(driver_module._LocalScheduler, "wait_for", wait_for), gate.patch(), \
                self.assertRaises(RuntimeError) as caught:
            run_experiment(pipe, cache=self.cache, cores=4, executor=executor)
        self.assertTrue(woke.is_set())
        self.assertIn("failing on purpose", str(caught.exception))
        self.assertNotIn("another branch", str(caught.exception))
        self.assertEqual(executor.calls, {"mesh": 1})
        self.assert_clean()

    def test_writing_call_publishes_when_an_off_branch_ran_first(self) -> None:
        off_started, use_started = threading.Event(), threading.Event()
        overlapped: list[bool] = []

        def hook(request: ExecutionRequest) -> None:
            if str(self.cache.ephemeral_dir) in " ".join(request.command):  # the `off` execution
                off_started.set()
                # Still running when the `use` branch makes the request, which must not wait for it.
                overlapped.append(use_started.wait(10))
            else:
                use_started.set()

        executor = _HookedExecutor({"mesh": hook})

        def pipe(run: RunContext) -> dict[str, Any]:
            mode = compare("mode", ["off", "use"])
            if mode == "use" and not run.dry_run:
                self.assertTrue(off_started.wait(10))
            return {"mesh": run.run_stage(self.mesh, settings={"tag": "m"}, cache=mode)}

        report = run_experiment(pipe, cache=self.cache, cores=2, executor=executor)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        self.assertEqual(executor.calls, {"mesh": 2})
        self.assertEqual(overlapped, [True])
        self.assertEqual([b.stages[0].status for b in report.branches], ["MISS", "MISS"])
        self.assertEqual([e["stage"] for e in self.cache.inspect()], ["mesh"])  # the `use` branch published
        self.assert_clean(ephemeral=False)

    def test_off_finishing_after_use_does_not_clobber_published_entry(self) -> None:
        off_started = threading.Event()
        off_release = threading.Event()
        off_finished = threading.Event()

        def hook(request: ExecutionRequest) -> None:
            if str(self.cache.ephemeral_dir) in " ".join(request.command):
                off_started.set()
                self.assertTrue(off_release.wait(10))

        executor = _HookedExecutor({"mesh": hook})

        def pipe(run: RunContext) -> dict[str, Any]:
            kind = compare("kind", ["off", "use", "refresh"])
            if kind == "use":
                if not run.dry_run:
                    self.assertTrue(off_started.wait(10))
                out = run.run_stage(self.mesh, settings={"tag": "m"}, cache="use")
                if not run.dry_run:
                    off_release.set()
                return {"mesh": out}
            if kind == "off":
                out = run.run_stage(self.mesh, settings={"tag": "m"}, cache="off")
                if not run.dry_run:
                    off_finished.set()
                return {"mesh": out}
            # Wait until `use` finished and `off` finished after it, then run `refresh`:
            # it must still reuse the published entry from `use` rather than re-executing.
            if not run.dry_run:
                self.assertTrue(off_finished.wait(10))
            return {"mesh": run.run_stage(self.mesh, settings={"tag": "m"}, cache="refresh")}

        report = run_experiment(pipe, cache=self.cache, cores=3, executor=executor)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        self.assertEqual(executor.calls, {"mesh": 2})  # only `off` and `use` executed; `refresh` reused `use`
        self.assertEqual(report.branches[2].stages[0].status, "REUSED")
        self.assertIsNotNone(report.branches[2].stages[0].request_key)
        self.assert_clean(ephemeral=False)

    def test_try_except_fallback_recovers_in_parallel_run(self) -> None:
        def pipe(run: RunContext) -> dict[str, Any]:
            k = compare("k", ["ok", "fallback"])
            if k == "fallback":
                try:
                    out = run.run_stage(self.solve, settings={"tag": "bad", "fail": True})
                except RuntimeError:
                    out = run.run_stage(self.solve, settings={"tag": "recovered"})
                return {"solution": out}
            return {"solution": run.run_stage(self.solve, settings={"tag": "good", "sleep": 0.1})}

        report = run_experiment(pipe, cache=self.cache, cores=2)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        self.assertEqual(len(report.branches), 2)
        self.assert_clean()

    def test_pipeline_except_cannot_swallow_cancellation(self) -> None:
        slow_started = threading.Event()
        swallowed: list[str] = []

        def hook(request: ExecutionRequest) -> None:
            if "slow" in " ".join(request.command):
                slow_started.set()

        executor = _HookedExecutor({"solve": hook})

        def pipe(run: RunContext) -> None:
            k = compare("k", ["slow", "boom"])
            if k == "slow":
                try:
                    run.run_stage(self.solve, settings={"tag": "slow", "sleep": 10.0})
                except Exception as exc:  # noqa: BLE001 - must not catch cancellation
                    swallowed.append(f"{type(exc).__name__}: {exc}")
                return
            if not run.dry_run:
                self.assertTrue(slow_started.wait(10))
            run.run_stage(self.mesh, settings={"tag": "boom", "fail": True})

        with self.assertRaisesRegex(RuntimeError, "failing on purpose"):
            run_experiment(pipe, cache=self.cache, cores=2, executor=executor)
        self.assertEqual(swallowed, [])
        self.assert_clean()

    def test_cache_locked_is_reentrant_in_all_lock_modes(self) -> None:
        for mode in ("flock", "lockfile", "none"):
            with self.subTest(mode=mode):
                cache = Cache(self.tmp / f"reentrant_{mode}", lock_mode=mode)
                with cache._locked():
                    with cache._locked():
                        self.assertEqual(cache._lock_depth, 2)
                self.assertEqual(cache._lock_depth, 0)

    def test_use_joins_a_running_refresh_instead_of_the_old_entry(self) -> None:
        run_experiment(lambda run: run.run_stage(self.mesh, settings={"tag": "m"}), cache=self.cache)  # warm
        gate = _WaiterGate()
        refresh_started = threading.Event()

        def hook(_request: ExecutionRequest) -> None:
            refresh_started.set()
            self.assertTrue(gate.wait_for_waiters(1))  # the `use` branch waits for this execution

        executor = _HookedExecutor({"mesh": hook})

        def pipe(run: RunContext) -> dict[str, Any]:
            mode = compare("mode", ["refresh", "use"])
            if mode == "use" and not run.dry_run:
                self.assertTrue(refresh_started.wait(10))
            return {"mesh": run.run_stage(self.mesh, settings={"tag": "m"}, cache=mode)}

        with gate.patch():
            report = run_experiment(pipe, cache=self.cache, cores=2, executor=executor)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        self.assertEqual(executor.calls, {"mesh": 1})
        self.assertEqual([b.stages[0].status for b in report.branches], ["MISS", "REUSED"])
        self.assertEqual(report.branches[0].stages[0].request_key, report.branches[1].stages[0].request_key)
        self.assert_clean()

    def test_dynamic_loop_without_compare_works(self) -> None:
        def pipe(run: RunContext) -> dict[str, Any]:
            m = compare("mesh", ["coarse", "fine"])
            target = {"coarse": 2, "fine": 4}[m]
            cur = run.run_stage(self.mesh, settings={"tag": m, "sleep": 0.1})
            cycles = 0
            for cycle in range(6):
                cur = run.run_stage(self.assemble, inputs={"inp": cur},
                                    settings={"tag": f"{m}{cycle}", "done_at": target, "sleep": 0.1})
                cycles += 1
                if not run.dry_run and cur.info.get("status") == "done":
                    break
            return {"cycles": cycles, "mesh": cur}

        par = run_experiment(pipe, cache=self.cache, cores=2)
        self.assertEqual([b["cycles"] for b in par.branches], [1, 3])
        seq = run_experiment(pipe, cache=self.cache)
        self.assertEqual([b["cycles"] for b in seq.branches], [1, 3])
        self.assertEqual(seq.cache_misses, 0)

    def test_choices_depending_on_stage_outputs_are_refused(self) -> None:
        def pipe(run: RunContext) -> None:
            out = run.run_stage(self.mesh, settings={"tag": "x"})
            if not run.dry_run and Path(out.path).read_text() == "x":
                compare("late", ["p", "q"])

        with self.assertRaisesRegex(RuntimeError, "cores=1"):
            run_experiment(pipe, cache=self.cache, cores=2)
        self.assertEqual(len(run_experiment(pipe, cache=self.cache).branches), 2)  # fine sequentially

    def test_unplanned_single_option_compare_is_refused(self) -> None:
        def pipe(run: RunContext) -> None:
            k = compare("k", ["a", "b"])
            run.run_stage(self.solve, settings={"tag": k})
            if not run.dry_run:
                compare("late", ["only"])  # one option: no new branch, but still not planned

        with self.assertRaisesRegex(RuntimeError, r"Unplanned compare\('late'\) at step 1.*cores=1"):
            run_experiment(pipe, cache=self.cache, cores=2)
        with self.assertRaisesRegex(RuntimeError, r"Unplanned compare\('late'\).*sharded or selected run"):
            run_experiment(pipe, cache=self.cache, shard=(0, 2))
        report = run_experiment(pipe, cache=self.cache)  # fine sequentially
        self.assertEqual([b.choice_labels for b in report.branches],
                         [{"k": "a", "late": "only"}, {"k": "b", "late": "only"}])

    def test_first_failure_stops_the_other_branches(self) -> None:
        def pipe(run: RunContext) -> None:
            k = compare("k", ["slow", "bad", "never1", "never2"])
            run.run_stage(self.solve, settings={"tag": k, "sleep": 30.0 if k == "slow" else 0.2,
                                                "fail": k == "bad", "spawn": k == "slow"})

        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "failing on purpose"):
            run_experiment(pipe, cache=self.cache, cores=2)
        self.assertLess(time.monotonic() - start, 15.0)
        started = [json.loads(line) for line in self.log.with_suffix(".log.start").read_text().splitlines()]
        self.assertEqual({s["tag"] for s in started}, {"slow", "bad"})  # no new stage after the failure
        for s in started:
            with self.assertRaises(ProcessLookupError):
                os.kill(s["pid"], 0)
        children = [s["child"] for s in started if s["child"]]
        self.assertEqual(len(children), 1)
        self.assertTrue(_wait_until(lambda: all(_is_dead(c) for c in children)), "the stage's child survived")
        self.assertEqual(self.cache.inspect(), [])  # the killed stage published nothing
        self.assert_clean()


class SchedulerUnitTests(unittest.TestCase):
    def _sched(self, cores: int) -> Any:
        return driver_module._LocalScheduler(
            cores, resources={}, stage_resources={}, exclusive_stages=frozenset(), executor=LocalExecutor())

    @staticmethod
    def _queued(sched: Any, n: int) -> bool:
        with sched._pool:
            return len(sched._queue) == n

    def test_waiting_big_request_is_not_overtaken(self) -> None:
        sched = self._sched(4)
        order: list[str] = []
        names = ("a", "big", "small")
        entered = {name: threading.Event() for name in names}
        release = {name: threading.Event() for name in names}

        def take(name: str, n: int) -> None:
            with sched.cores_held(n):
                order.append(name)
                entered[name].set()
                release[name].wait(10)

        threads = {name: threading.Thread(target=take, args=(name, n)) for name, n in (("a", 1), ("big", 4), ("small", 1))}
        threads["a"].start()
        self.assertTrue(entered["a"].wait(10))
        threads["big"].start()
        self.assertTrue(_wait_until(lambda: self._queued(sched, 1)))
        threads["small"].start()
        self.assertTrue(_wait_until(lambda: self._queued(sched, 2)))
        self.assertEqual(order, ["a"])  # 3 cores are free, but the 4-core request came first
        release["a"].set()
        self.assertTrue(entered["big"].wait(10))
        self.assertEqual(order, ["a", "big"])
        self.assertFalse(entered["small"].is_set())  # the 4-core request holds every core
        release["big"].set()
        self.assertTrue(entered["small"].wait(10))
        release["small"].set()
        for t in threads.values():
            t.join(10)
        self.assertEqual(order, ["a", "big", "small"])

    def test_cancel_wakes_core_waiters(self) -> None:
        sched = self._sched(1)
        raised: list[BaseException] = []
        with sched.cores_held(1):
            def wait() -> None:
                try:
                    with sched.cores_held(1):
                        pass
                except BaseException as exc:  # noqa: BLE001
                    raised.append(exc)

            t = threading.Thread(target=wait)
            t.start()
            self.assertTrue(_wait_until(lambda: self._queued(sched, 1)))
            sched.cancel()
            t.join(5)
        self.assertEqual([type(e) for e in raised], [driver_module._Cancelled])

    def test_executor_cancel_kills_only_its_token(self) -> None:
        executor = LocalExecutor()
        mine, other = threading.Event(), threading.Event()
        results: dict[str, Any] = {}

        def run(name: str, token: threading.Event) -> None:
            try:
                results[name] = executor.run(ExecutionRequest(
                    stage_name=name, command=[sys.executable, "-c", "import time; time.sleep(1.5)"], cancel_token=token))
            except BaseException as exc:  # noqa: BLE001
                results[name] = exc

        threads = [threading.Thread(target=run, args=(n, t)) for n, t in (("mine", mine), ("other", other))]
        for t in threads:
            t.start()
        self.assertTrue(_wait_until(lambda: len(executor._active) == 2))
        mine.set()
        executor.cancel(mine)
        for t in threads:
            t.join(10)
        self.assertIsInstance(results["mine"], StageCancelled)
        self.assertEqual(results["other"].exit_code, 0)
        with self.assertRaises(StageCancelled):  # a set token starts nothing
            executor.run(ExecutionRequest(stage_name="late", command=["true"], cancel_token=mine))


class CacheThreadSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_parallel_cache_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_concurrent_store_lookup_and_pin(self) -> None:
        for lock_mode in ("flock", "lockfile", "none"):
            with self.subTest(lock_mode=lock_mode):
                cache = Cache(self.tmp / lock_mode, lock_mode=lock_mode)
                errors: list[BaseException] = []
                with cache.lease() as lease:
                    def worker(w: int) -> None:
                        try:
                            for i in range(15):
                                out = cache.make_temp_path(".txt")
                                out.write_text(f"{w}-{i}")
                                key = f"{w:02d}{i:02d}".ljust(64, "0")
                                cache.store(request_key=key, stage_name="s", code_hash="c", inputs_spec={},
                                            settings={}, temp_outputs={"out": (out, ".txt")}, info={},
                                            execution={}, lease=lease)
                                hit = cache.lookup(key, expected_outputs={"out": ".txt"}, lease=lease)
                                assert hit is not None and hit["out"].path.read_text() == f"{w}-{i}"
                                lease.pin([f"extra-{w}-{i}"])
                                cache.hash_file(hit["out"].path)
                        except BaseException as exc:  # noqa: BLE001
                            errors.append(exc)

                    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.assertEqual(errors, [])
                    names = lease.object_names
                    self.assertEqual(len([n for n in names if n.startswith("extra-")]), 8 * 15)
                    self.assertEqual(len([n for n in names if n.endswith(".txt")]), 8 * 15)
                    if lock_mode != "none":
                        on_disk = json.loads(next(cache.leases_dir.glob("*.lease")).read_text())
                        self.assertEqual(set(on_disk["objects"]), set(names))
                self.assertEqual(len(cache.inspect()), 8 * 15)
                self.assertEqual(list(cache.leases_dir.iterdir()), [])

    def test_pin_returns_only_once_the_name_is_in_the_lease_file(self) -> None:
        cache = Cache(self.tmp / "pin", lock_mode="flock")
        real_locked = cache._locked
        reached = {"A": threading.Event(), "B": threading.Event()}  # asked for the cache lock, or returned
        on_disk: dict[str, list[str]] = {}

        @contextlib.contextmanager
        def locked() -> Any:
            event = reached.get(threading.current_thread().name)
            if event is not None:
                event.set()
            with real_locked():
                yield

        with cache.lease() as lease:
            def pin(name: str) -> None:
                lease.pin(["obj"])
                on_disk[name] = json.loads(lease._path.read_text())["objects"]
                reached[name].set()

            cache._locked = locked  # type: ignore[method-assign]
            try:
                with real_locked():  # hold the cache lock: the first pin can't write the file yet
                    a = threading.Thread(target=pin, args=("A",), name="A")
                    a.start()
                    self.assertTrue(reached["A"].wait(10))
                    b = threading.Thread(target=pin, args=("B",), name="B")
                    b.start()
                    self.assertTrue(reached["B"].wait(10))
                a.join(10)
                b.join(10)
            finally:
                del cache._locked
        self.assertEqual(on_disk, {"A": ["obj"], "B": ["obj"]})

    def test_closed_lease_ignores_late_pins(self) -> None:
        cache = Cache(self.tmp / "c")
        lease = cache.lease()
        with lease:
            lease.pin(["a"])
        lease.pin(["b"])  # after close: no file to write, no error
        self.assertEqual(list(cache.leases_dir.iterdir()), [])


class ExperimentMainCoresTests(ParallelTestBase):
    def _pipe(self):
        def pipe(run: RunContext) -> dict[str, Any]:
            k = compare("k", ["a", "b"])
            return {"solution": run.run_stage(self.solve, settings={"tag": k})}

        return pipe

    def _main(self, *argv: str, **defaults: Any) -> tuple[int, str, str, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        real = experiment_cli.run_experiment

        def spy(fn: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return real(fn, **kwargs)

        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(experiment_cli, "run_experiment", spy), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(self._pipe(), argv=list(argv), cache=self.cache, **defaults)
        return code, out.getvalue(), err.getvalue(), calls

    def test_script_default_and_flag(self) -> None:
        defaults = dict(cores=3, resources={"cpus": 2, "time": "00:10:00"}, stage_resources={"solve": {"cpus": 1}},
                        exclusive_stages=["mesh"])
        code, _, err, calls = self._main(**defaults)
        self.assertEqual(code, 0, err)
        self.assertEqual(calls[-1]["cores"], 3)
        self.assertEqual(calls[-1]["resources"], {"cpus": 2, "time": "00:10:00"})
        self.assertEqual(calls[-1]["stage_resources"], {"solve": {"cpus": 1}})
        self.assertEqual(calls[-1]["exclusive_stages"], frozenset({"mesh"}))
        self.assertEqual({r["omp"] for r in self.records("solve")}, {"1"})

        code, _, err, calls = self._main("--cores", "2", "--out", str(self.tmp / "res"), **defaults)
        self.assertEqual(code, 0, err)
        self.assertEqual(calls[-1]["cores"], 2)
        code, _, err, calls = self._main()
        self.assertEqual((code, calls[-1]["cores"]), (0, 1))

    def test_cluster_modes_ignore_script_cores(self) -> None:
        for argv in (["--count"], ["--warmup"], ["--shard", "0/2"], ["--collect", "--allow-recompute", "--out", str(self.tmp / "r")]):
            with self.subTest(argv=argv):
                code, _, err, calls = self._main(*argv, cores=4)
                self.assertIn(code, (0,), err)
                self.assertTrue(all(c.get("cores", 1) == 1 for c in calls), calls)
        with _Env(SLURM_JOB_ID="123"):
            code, _, err, calls = self._main(cores=4, stage_resources={"solve": {"cpus": 4}})
            self.assertEqual(code, 0, err)
            self.assertEqual(calls[-1]["cores"], 1)
        with _Env(SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_TASK_COUNT="2"):
            code, _, err, calls = self._main(cores=4)
            self.assertEqual(code, 0, err)
            self.assertNotIn("cores", calls[-1])
            self.assertEqual(calls[-1]["shard"], "slurm")

    def test_cores_flag_refused_with_cluster_modes(self) -> None:
        plan = str(self.tmp / "plan.json")
        for argv in (["--count"], ["--shard", "0/2"], ["--warmup"], ["--collect", "--out", "x"], ["--dag-plan", plan],
                     ["--dag-plan", plan, "--dag-task", "0/0"], ["--show-settings"]):
            with self.subTest(argv=argv):
                code, _, err, calls = self._main("--cores", "2", *argv)
                self.assertEqual(code, 2)
                self.assertIn("--cores", err)
                self.assertEqual(calls, [])
        with _Env(SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_TASK_COUNT="2"):
            self.assertEqual(self._main("--cores", "2")[0], 2)
        with _Env(SLURM_JOB_ID="1"):  # inside any Slurm job, even an interactive one
            code, _, err, calls = self._main("--cores", "2")
            self.assertEqual((code, calls), (2, []))
            self.assertIn("Slurm job", err)
            code, _, err, calls = self._main("--cores", "1")
            self.assertEqual((code, calls[-1]["cores"]), (0, 1), err)
        for bad in ("0", "-1", "two"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self._main("--cores", bad)

    def test_show_settings_and_validation(self) -> None:
        code, out, _, _ = self._main("--show-settings", cores=8, exclusive_stages=["solve"])
        self.assertEqual(code, 0)
        self.assertNotIn("cores", out)
        self.assertNotIn("exclusive", out)
        self.assertEqual(sorted(line.split("=")[0] for line in out.splitlines()),
                         ["setting cpus", "setting mem_per_cpu", "setting mode", "setting shard_order", "setting time"])
        for bad in (dict(cores=0), dict(cores=True), dict(exclusive_stages="solve"), dict(exclusive_stages=[""])):
            with self.subTest(**{k: repr(v) for k, v in bad.items()}), self.assertRaisesRegex(ValueError, "experiment_main"):
                experiment_main(self._pipe(), argv=["--show-settings"], **bad)


class InterruptTests(ParallelTestBase):
    """A parallel run stopped by SIGTERM or SIGINT (to the runner only) kills its stages and the
    processes they started, and leaves no staging files, leases or locks behind."""

    def _start(self) -> subprocess.Popen[bytes]:
        script = self.tmp / "sweep.py"
        script.write_text(textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT / "python")!r})
            from poisson_runner import Cache, Stage, compare, run_experiment
            solve = Stage.from_cli("solve", binary={str(self.tmp / "solve_bin")!r}, input_flags={{"inp": "--in"}},
                                   setting_flags={{"tag": "--tag", "sleep": "--sleep", "spawn": "--spawn"}},
                                   output_flags={{"out": "--out"}},
                                   output_specs={{"out": ".txt"}}, info_flag="--info-out")

            def pipe(run):
                k = compare("k", ["a", "b", "c", "d"])
                run.run_stage(solve, settings={{"tag": k, "sleep": 60, "spawn": True}},
                              cache="off" if k == "b" else "use")

            run_experiment(pipe, cache=Cache({str(self.cache.root)!r}), cores=2)
        """))
        return subprocess.Popen([sys.executable, str(script)], env=dict(os.environ), stderr=subprocess.PIPE)

    def _wait_started(self, n: int) -> list[dict[str, Any]]:
        start_log = self.log.with_suffix(".log.start")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if start_log.exists():
                lines = [json.loads(x) for x in start_log.read_text().splitlines() if x.strip()]
                if len(lines) >= n:
                    return lines
            time.sleep(0.05)
        self.fail("stages did not start")

    def _check(self, sig: int, expect_code: int | None) -> None:
        proc = self._start()
        started = self._wait_started(2)
        os.kill(proc.pid, sig)
        try:
            _, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("runner did not stop")
        if expect_code is not None:
            self.assertEqual(proc.returncode, expect_code, stderr.decode())
        else:
            self.assertNotEqual(proc.returncode, 0)
        self.assertEqual({s["tag"] for s in started}, {"a", "b"})
        for s in started:
            with self.assertRaises(ProcessLookupError):
                os.kill(s["pid"], 0)
        children = [s["child"] for s in started]
        self.assertTrue(all(children), started)
        self.assertTrue(_wait_until(lambda: all(_is_dead(c) for c in children)), "a stage's child survived")
        self.assertEqual(self.cache.inspect(), [])
        self.assert_clean()
        self.assertFalse((self.cache.root / ".cache.lock.excl").exists())

    def test_sigterm(self) -> None:
        self._check(signal.SIGTERM, 128 + signal.SIGTERM)

    def test_sigint(self) -> None:
        self._check(signal.SIGINT, None)  # KeyboardInterrupt


if __name__ == "__main__":
    unittest.main()
