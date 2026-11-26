"""Tests for running on a SLURM cluster: lock settings, cross-node cleanup, SIGTERM handling, experiment CLI."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import (  # noqa: E402
    Artifact,
    Cache,
    LockMode,
    RunContext,
    Stage,
    collect_host_environment,
    compare,
    experiment_main,
    run_experiment,
)
from poisson_runner import cache as cache_module  # noqa: E402


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ClusterTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_cluster_test_")
        self.tmp = Path(self._tmp.name)
        self._clean_env = _env(SLURM_JOB_ID=None, SLURM_ARRAY_TASK_ID=None, POISSON_CACHE_LOCK_MODE=None)
        self._clean_env.__enter__()

    def tearDown(self) -> None:
        self._clean_env.__exit__(None, None, None)
        self._tmp.cleanup()

    def _stage(self, name: str, *, fail_tag: str | None = None) -> tuple[Stage, Path]:
        """A stage that logs each execution and writes its tag; exits 1 when `--tag` == `fail_tag`."""
        log = self.tmp / f"{name}.log"
        binary = _write_executable(
            self.tmp / f"{name}_bin",
            f"""
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log)!r}, "a") as f:
    f.write(a.tag + "\\n")
if a.tag == {fail_tag!r}:
    sys.exit(1)
pathlib.Path(a.out).write_text(a.tag)
pathlib.Path(a.info_out).write_text("{{}}")
""",
        )
        stage = Stage.from_cli(
            name,
            binary=binary,
            input_flags={"inp": "--in"},
            setting_flags={"tag": "--tag"},
            output_flags={"out": "--out"},
            output_specs={"out": ".txt"},
            info_flag="--info-out",
        )
        return stage, log

    @staticmethod
    def _runs(log: Path) -> int:
        return len(log.read_text().splitlines()) if log.exists() else 0

    # ------------------------------------------------------------------ lock settings

    def test_lock_timeout_must_exceed_stale_age(self) -> None:
        c = Cache(self.tmp / "defaults")
        self.assertGreater(c.lock_timeout_s, c.lock_stale_s)
        with self.assertRaises(ValueError):
            Cache(self.tmp / "bad", lock_timeout_s=60.0, lock_stale_s=300.0)
        with self.assertRaises(ValueError):
            Cache(self.tmp / "equal", lock_timeout_s=30.0, lock_stale_s=30.0)

    def test_waiter_reclaims_lock_left_by_killed_process_on_another_host(self) -> None:
        c = Cache(self.tmp / "lf", lock_mode="lockfile", lock_stale_s=0.3, lock_timeout_s=5.0)
        c._excl_lock_path.write_text(json.dumps({"host": "other-node", "pid": 1, "token": "dead"}), encoding="utf-8")
        start = time.monotonic()
        with c._locked():
            pass
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertFalse(c._excl_lock_path.exists())

    def test_fenced_stale_lock_recovery_does_not_remove_live_owner(self) -> None:
        c = Cache(self.tmp / "fenced_lf", lock_mode="lockfile", lock_stale_s=0.1, lock_timeout_s=5.0)
        c._excl_lock_path.write_text(json.dumps({"host": "other-node", "pid": 1, "token": "dead0"}), encoding="utf-8")
        old = time.time() - 10.0
        os.utime(c._excl_lock_path, (old, old))

        # Simulate Waiter B inspecting the stale lock, pausing while Waiter A reclaims it and
        # acquires a fresh live lock, then Waiter B attempting to finish reclamation.
        orig_is_stale = c._is_lockfile_stale
        injected = False
        live_lock_ctx = None

        def _pause_and_let_waiter_a_acquire(path: Path, st: os.stat_result, meta):
            nonlocal injected, live_lock_ctx
            res = orig_is_stale(path, st, meta)
            if not injected and path == c._excl_lock_path and res:
                injected = True
                # Waiter A reclaims the stale lock and enters the critical section with a fresh lock.
                c2 = Cache(c.root, lock_mode="lockfile", lock_stale_s=0.1, lock_timeout_s=5.0)
                live_lock_ctx = c2._locked()
                live_lock_ctx.__enter__()
            return res

        try:
            with mock.patch.object(c, "_is_lockfile_stale", side_effect=_pause_and_let_waiter_a_acquire):
                self.assertFalse(c._try_break_stale_lockfile())
            self.assertTrue(injected)
            # Waiter A's live lockfile must still be intact with a fresh token != "dead0".
            self.assertTrue(c._excl_lock_path.exists())
            live_meta = json.loads(c._excl_lock_path.read_text(encoding="utf-8"))
            self.assertNotEqual(live_meta.get("token"), "dead0")
        finally:
            if live_lock_ctx is not None:
                live_lock_ctx.__exit__(None, None, None)
        self.assertFalse(c._excl_lock_path.exists())

    def test_lock_mode_immutable_on_existing_cache(self) -> None:
        # 1. Brand-new cache inside SLURM defaults to lockfile and persists .lock_mode.
        slurm_root = self.tmp / "slurm_new"
        with _env(SLURM_JOB_ID="123"):
            self.assertEqual(Cache(slurm_root).lock_mode, LockMode.LOCKFILE)
            self.assertEqual(Cache(self.tmp / "explicit_flock", lock_mode="flock").lock_mode, LockMode.FLOCK)
            with _env(POISSON_CACHE_LOCK_MODE="none"):
                self.assertEqual(Cache(self.tmp / "env_none").lock_mode, LockMode.NONE)
        # Later process outside SLURM (e.g. login node) adopts lockfile from the marker.
        self.assertEqual(Cache(slurm_root).lock_mode, LockMode.LOCKFILE)

        # 2. Existing flock cache refuses silent switch inside SLURM or explicit conflicting mode.
        flock_root = self.tmp / "existing_flock"
        self.assertEqual(Cache(flock_root).lock_mode, LockMode.FLOCK)
        with _env(SLURM_JOB_ID="456"):
            with self.assertRaises(ValueError):
                Cache(flock_root)
        with self.assertRaises(ValueError):
            Cache(flock_root, lock_mode="lockfile")
        with _env(POISSON_CACHE_LOCK_MODE="lockfile"):
            with self.assertRaises(ValueError):
                Cache(flock_root)
        # Marker must remain unchanged.
        self.assertEqual((flock_root / ".lock_mode").read_text(encoding="utf-8").strip(), "flock")

        # 3. Existing lockfile cache refuses explicit flock override.
        with self.assertRaises(ValueError):
            Cache(slurm_root, lock_mode="flock")

    # ------------------------------------------------------------------ cross-node cleanup

    def test_ephemeral_cleanup_spares_young_files_of_other_hosts(self) -> None:
        c = Cache(self.tmp / "eph")
        own = c.make_ephemeral_path(".h5")
        self.assertTrue(own.name.startswith(socket.gethostname().replace("/", "_") + "_"))
        own.write_text("mine")
        # Another node's file whose PID happens not to exist here.
        other = c.ephemeral_dir / f"other-node_999999_{'0' * 32}.h5"
        other.write_text("theirs")
        # A file left by a dead process on this host.
        dead = c.ephemeral_dir / f"{socket.gethostname().replace('/', '_')}_999999_{'1' * 32}.h5"
        dead.write_text("dead")
        with mock.patch.object(cache_module, "_pid_is_alive", side_effect=lambda pid: pid == os.getpid()):
            removed = c._clean_ephemeral_unlocked()
        self.assertEqual(removed, 1)
        self.assertTrue(own.exists())
        self.assertTrue(other.exists())
        self.assertFalse(dead.exists())
        # Once it is an hour old with no live lease, the other host's file counts as abandoned.
        old = time.time() - 7200
        os.utime(other, (old, old))
        with mock.patch.object(cache_module, "_pid_is_alive", side_effect=lambda pid: pid == os.getpid()):
            self.assertEqual(c._clean_ephemeral_unlocked(), 1)
        self.assertFalse(other.exists())

    def test_remote_cleanup_spares_hour_old_files_with_live_lease(self) -> None:
        c = Cache(self.tmp / "remote_live", lock_mode="lockfile", lock_stale_s=1.0, lock_timeout_s=5.0)
        old = time.time() - 7200
        tmp_dir = c.tmp_dir / f"other-node_4242_{'a' * 32}"
        tmp_dir.mkdir()
        tmp_file = tmp_dir / "0_out.h5"
        tmp_file.write_text("in-progress")
        eph_dir = c.ephemeral_dir / f"other-node_4242_{'b' * 32}"
        eph_dir.mkdir()
        eph_file = eph_dir / "0_out.h5"
        eph_file.write_text("ephemeral")
        for p in (tmp_file, tmp_dir, eph_file, eph_dir):
            os.utime(p, (old, old))

        # Active remote lease (fresh mtime) for (other-node, 4242).
        remote_lease = c.leases_dir / f"other-node_4242_{'c' * 32}.lease"
        remote_lease.write_text(
            json.dumps({"host": "other-node", "pid": 4242, "mode": "lockfile", "objects": []}),
            encoding="utf-8",
        )
        res = c.clean()
        self.assertEqual((res["removed_tmp"], res["removed_ephemeral"]), (0, 0))
        self.assertTrue(tmp_file.exists())
        self.assertTrue(eph_file.exists())

        # Once the remote lease expires, routine clean removes the lease and the old files.
        stale_t = time.time() - 10.0
        os.utime(remote_lease, (stale_t, stale_t))
        res2 = c.clean()
        self.assertEqual((res2["removed_tmp"], res2["removed_ephemeral"]), (1, 1))
        self.assertFalse(remote_lease.exists())
        self.assertFalse(tmp_dir.exists())
        self.assertFalse(eph_dir.exists())

    # ------------------------------------------------------------------ sharding

    def test_sharded_discovery_uses_short_lived_bounded_lease(self) -> None:
        stage, _ = self._stage("leaf")
        c = Cache(self.tmp / "shard_cache", lock_mode="lockfile")

        # Pre-populate all 6 branches in the cache.
        def pipe_all(run: RunContext) -> Artifact:
            a = compare("a", [1, 2, 3])
            b = compare("b", ["x", "y"])
            return run.run_stage(stage, settings={"tag": f"{a}{b}"})

        full_report = run_experiment(pipe_all, cache=c)
        all_obj_names = [b.result.path.name for b in full_report.branches]
        self.assertEqual(len(set(all_obj_names)), 6)

        observed_exec_pins: list[set[str]] = []
        max_discovery_pins = 0

        def pipe_check(run: RunContext) -> Artifact:
            nonlocal max_discovery_pins
            a = compare("a", [1, 2, 3])
            b = compare("b", ["x", "y"])
            art = run.run_stage(stage, settings={"tag": f"{a}{b}"})
            if run.dry_run:
                max_discovery_pins = max(max_discovery_pins, len(run._lease.object_names))
            else:
                observed_exec_pins.append(set(run._lease.object_names))
            return art

        with mock.patch.object(Cache, "lease", autospec=True, side_effect=Cache.lease) as lease:
            report = run_experiment(pipe_check, cache=c, shard=(1, 2))

        # 2 leases total (1 short-lived discovery lease + 1 execution lease), NOT 1 per branch.
        self.assertEqual(lease.call_count, 2)
        self.assertEqual([b.index for b in report.branches], [1, 3, 5])
        # Discovery never accumulated more than one branch's object.
        self.assertLessEqual(max_discovery_pins, 1)
        # Shard (1, 2)'s execution lease must never pin branches 0, 2, or 4.
        other_shard_objs = {all_obj_names[0], all_obj_names[2], all_obj_names[4]}
        for pinned in observed_exec_pins:
            self.assertTrue(pinned.isdisjoint(other_shard_objs))

    # ------------------------------------------------------------------ SIGTERM

    def test_sigterm_cleans_up_lock_leases_child_and_sibling_staging(self) -> None:
        cache_root = self.tmp / "term_cache"
        pid_file = self.tmp / "child.pid"
        sleeper = _write_executable(
            self.tmp / "sleeper",
            f"""
import argparse, os, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
a = p.parse_args()
# Mimic poisson_adapt creating a sibling staging file next to --out before being killed.
pathlib.Path(a.out + ".tmp.out." + str(os.getpid())).write_text("partial")
pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
time.sleep(60)
""",
        )
        script = self.tmp / "term_experiment.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(REPO_ROOT / "python")!r})
                from poisson_runner import Cache, Stage, run_experiment
                stage = Stage(name="sleep", executable={str(sleeper)!r},
                              build_command=lambda call: [str(call.executable), "--out", str(call.outputs["out"])],
                              output_specs={{"out": ".txt"}})
                run_experiment(lambda run: run.run_stage(stage), cache=Cache({str(cache_root)!r}, lock_mode="lockfile"))
                """
            ),
            encoding="utf-8",
        )
        proc = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "POISSON_BUILD_DIR": os.environ.get("POISSON_BUILD_DIR", "")})
        try:
            deadline = time.monotonic() + 30
            while not pid_file.exists() and time.monotonic() < deadline:
                if proc.poll() is not None:
                    self.fail(f"experiment exited early with {proc.returncode}")
                time.sleep(0.05)
            self.assertTrue(pid_file.exists(), "stage never started")
            time.sleep(0.2)
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=30), 128 + signal.SIGTERM)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        child = int(pid_file.read_text())
        self.assertFalse(cache_module._pid_is_alive(child), "stage process kept running")
        self.assertEqual(list((cache_root / "leases").iterdir()), [])
        self.assertFalse((cache_root / ".cache.lock.excl").exists())
        self.assertEqual(list((cache_root / "tmp").iterdir()), [])

    def test_sigterm_during_lockfile_acquisition_leaves_no_lock(self) -> None:
        cache_root = self.tmp / "sigterm_lock_cache"
        script = self.tmp / "sigterm_lock.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import os, signal, sys
                sys.path.insert(0, {str(REPO_ROOT / "python")!r})
                from poisson_runner import Cache
                from poisson_runner.driver import _sigterm_as_exit
                c = Cache({str(cache_root)!r}, lock_mode="lockfile")
                orig_link = os.link
                def _link_then_term(src, dst):
                    orig_link(src, dst)
                    if str(dst).endswith(".cache.lock.excl"):
                        os.kill(os.getpid(), signal.SIGTERM)
                os.link = _link_then_term
                with _sigterm_as_exit():
                    with c._locked():
                        pass
                """
            ),
            encoding="utf-8",
        )
        res = subprocess.run([sys.executable, str(script)], check=False)
        self.assertEqual(res.returncode, 128 + signal.SIGTERM)
        self.assertFalse((cache_root / ".cache.lock.excl").exists())
        self.assertEqual(list(cache_root.glob(".cache.lock.tmp.*")), [])

    # ------------------------------------------------------------------ host metadata

    def test_cpu_count_is_the_allocation(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity")
        self.assertEqual(collect_host_environment()["cpu_count"], len(os.sched_getaffinity(0)))

    # ------------------------------------------------------------------ experiment CLI

    def _main(self, pipe, *argv: str, cache: Cache) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=list(argv), cache=cache)
        return code, out.getvalue(), err.getvalue()

    def test_shard_order_block_keeps_siblings_together(self) -> None:
        mesh, mesh_log = self._stage("mesh")
        leaf, _ = self._stage("leaf")
        c = Cache(self.tmp / "order_cache")

        def pipe(run: RunContext) -> dict[str, Artifact]:
            m = compare("mesh", ["coarse", "fine"])
            msh = run.run_stage(mesh, settings={"tag": m})
            s = compare("solver", ["cg", "gmres", "direct"])
            return {"out": run.run_stage(leaf, inputs={"inp": msh}, settings={"tag": f"{m}-{s}"})}

        stride = [[b.index for b in run_experiment(pipe, cache=c, shard=(i, 2), dry_run=True).branches] for i in range(2)]
        block = [
            [b.index for b in run_experiment(pipe, cache=c, shard=(i, 2), shard_order="block", dry_run=True).branches]
            for i in range(2)
        ]
        self.assertEqual(stride, [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(block, [[0, 1, 2], [3, 4, 5]])
        uneven = [
            [b.index for b in run_experiment(pipe, cache=c, shard=(i, 4), shard_order="block", dry_run=True).branches]
            for i in range(4)
        ]
        self.assertEqual(sorted(i for part in uneven for i in part), list(range(6)))
        self.assertTrue(all(1 <= len(part) <= 2 for part in uneven))

        # Each block shard computes its one mesh once (here sequentially, so stride would hit the cache too).
        for i in range(2):
            run_experiment(pipe, cache=c, shard=(i, 2), shard_order="block")
        self.assertEqual(self._runs(mesh_log), 2)
        with self.assertRaises(ValueError):
            run_experiment(pipe, cache=c, shard=(0, 2), shard_order="random")  # type: ignore[arg-type]

    def test_select_filters_branches_and_keeps_global_indices(self) -> None:
        leaf, leaf_log = self._stage("leaf")
        c = Cache(self.tmp / "select_cache")

        def pipe(run: RunContext) -> dict[str, Artifact]:
            m = compare("mesh", ["coarse", "fine"])
            if m == "fine":
                compare("order", [1, 2])
            s = compare("solver", {"cg": "cg", "lu": "direct"})
            return {"out": run.run_stage(leaf, settings={"tag": f"{m}-{s}-{run.choices.get('order')}"})}

        def indices(**kw):
            return [b.index for b in run_experiment(pipe, cache=c, dry_run=True, **kw).branches]

        self.assertEqual(indices(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(indices(select={"mesh": "fine"}), [2, 3, 4, 5])
        self.assertEqual(indices(select={"mesh": "fine", "solver": "lu"}), [3, 5])
        self.assertEqual(indices(select={"solver": ["cg", "lu"], "mesh": "coarse"}), [0, 1])
        self.assertEqual(indices(select={"order": "2"}), [4, 5])  # coarse branches never reach `order`
        self.assertEqual(indices(select={"mesh": "fine"}, shard=(1, 2)), [3, 5])
        self.assertEqual(indices(select={"mesh": "fine"}, shard=(1, 2), shard_order="block"), [4, 5])
        for bad in ({"mesh": "medium"}, {"meshes": "fine"}, {"mesh": []}):
            with self.assertRaises(ValueError):
                indices(select=bad)

        # The CLI: --count and --warmup honour --select; a selected run saves only as a partial run.
        code, out, _ = self._main(pipe, "--count", "--select", "mesh=fine", "--select", "solver=lu", cache=c)
        self.assertEqual(code, 0)
        self.assertIn("branches: 2", out.splitlines())
        code, _, _ = self._main(pipe, "--warmup", "--select", "mesh=fine", cache=c)
        self.assertEqual(code, 0)
        self.assertEqual(leaf_log.read_text().splitlines(), ["fine-cg-1"])
        code, _, _ = self._main(pipe, "--select", "mesh=coarse", "--out", str(self.tmp / "coarse"), cache=c)
        self.assertEqual(code, 0)
        saved = json.loads((self.tmp / "coarse" / "report.json").read_text())
        self.assertEqual(saved["selection"], {"mesh": ["coarse"]})
        self.assertEqual([b["index"] for b in saved["branches"]], [0, 1])
        with self.assertRaises(SystemExit):
            self._main(pipe, "--select", "mesh", cache=c)
        with self.assertRaises(TypeError):
            experiment_main(pipe, argv=["--count"], cache=c, select={"mesh": "fine"})

    def test_experiment_cli_count_pieces_and_collect(self) -> None:
        prefix, prefix_log = self._stage("prefix")
        leaf, leaf_log = self._stage("leaf")
        c = Cache(self.tmp / "cli_cache")

        def pipe(run: RunContext) -> dict[str, Artifact]:
            shared = run.run_stage(prefix, settings={"tag": "shared"})
            k = compare("k", [1, 2, 3, 4])
            return {"out": run.run_stage(leaf, inputs={"inp": shared}, settings={"tag": f"leaf{k}"})}

        code, out, _ = self._main(pipe, "--count", cache=c)
        self.assertEqual(code, 0)
        self.assertIn("branches: 4", out.splitlines())
        self.assertEqual(self._runs(leaf_log), 0)

        # Collecting before the pieces ran is refused and computes nothing.
        code, _, err = self._main(pipe, "--collect", "--out", str(self.tmp / "early"), cache=c)
        self.assertEqual(code, 2)
        self.assertIn("not in the cache", err)
        self.assertEqual(self._runs(leaf_log), 0)
        self.assertFalse((self.tmp / "early").exists())

        code, _, _ = self._main(pipe, "--warmup", cache=c)
        self.assertEqual((code, self._runs(prefix_log), self._runs(leaf_log)), (0, 1, 1))

        for i in range(2):
            code, out, _ = self._main(pipe, "--shard", f"{i}/2", cache=c)
            self.assertEqual(code, 0)
            self.assertIn(f"[piece {i}/2] branches: 2", out)
        self.assertEqual((self._runs(prefix_log), self._runs(leaf_log)), (1, 4))

        results = self.tmp / "results"
        code, out, _ = self._main(pipe, "--collect", "--out", str(results), cache=c)
        self.assertEqual(code, 0, out)
        self.assertTrue((results / "report.json").is_file())
        self.assertEqual((self._runs(prefix_log), self._runs(leaf_log)), (1, 4))  # nothing recomputed

        with self.assertRaises(SystemExit):
            self._main(pipe, "--shard", "2/2", cache=c)

    def test_experiment_cli_failed_branch_blocks_collect(self) -> None:
        leaf, _ = self._stage("leaf", fail_tag="leaf2")
        c = Cache(self.tmp / "fail_cache")

        def pipe(run: RunContext) -> dict[str, Artifact]:
            k = compare("k", [1, 2, 3])
            return {"out": run.run_stage(leaf, settings={"tag": f"leaf{k}"})}

        with _env(SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_TASK_MIN="0", SLURM_ARRAY_TASK_MAX="0", SLURM_ARRAY_TASK_COUNT="1"):
            code, _, err = self._main(pipe, cache=c)  # array task without --shard: shard="slurm", keeps going
        self.assertEqual(code, 1)
        self.assertIn("failed", err)
        code, _, err = self._main(pipe, "--collect", "--out", str(self.tmp / "res"), cache=c)
        self.assertEqual(code, 2)
        self.assertIn("branch 1: leaf", err)

    def test_collect_refuses_mid_run_eviction_without_executing(self) -> None:
        leaf, leaf_log = self._stage("leaf")
        c = Cache(self.tmp / "collect_race_cache")

        def pipe(run: RunContext) -> dict[str, Artifact]:
            k = compare("k", [1, 2])
            return {"out": run.run_stage(leaf, settings={"tag": f"leaf{k}"})}

        run_experiment(pipe, cache=c)
        self.assertEqual(self._runs(leaf_log), 2)

        # Simulate another process evicting the second entry right after the first lookup succeeds.
        orig_lookup = c.lookup
        calls = 0

        def _evict_on_second_lookup(request_key: str, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return None
            return orig_lookup(request_key, **kwargs)

        out_dir = self.tmp / "evicted_collect"
        with mock.patch.object(c, "lookup", side_effect=_evict_on_second_lookup):
            code, _, err = self._main(pipe, "--collect", "--out", str(out_dir), cache=c)
        self.assertEqual(code, 2)
        self.assertIn("Refusing to collect", err)
        self.assertEqual(self._runs(leaf_log), 2)  # must NOT execute the missing stage
        self.assertFalse(out_dir.exists())

    def test_submit_sh_defaults_extra_validation_and_parsable_id(self) -> None:
        fake_bin = self.tmp / "fake_bin"
        fake_bin.mkdir()
        sbatch_log = self.tmp / "sbatch.log"
        fake_sbatch = _write_executable(
            fake_bin / "sbatch",
            f"""
import pathlib, sys
with open({str(sbatch_log)!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
count = len(pathlib.Path({str(sbatch_log)!r}).read_text().splitlines())
print(f"{{1000 + count}};delftblue")
""",
        )
        _ = fake_sbatch
        exp = self.tmp / "exp.py"
        exp.write_text(
            textwrap.dedent(
                """
                from poisson_runner import RunContext, compare, experiment_main
                def pipe(run: RunContext):
                    return {"k": compare("k", [1, 2, 3])}
                if __name__ == "__main__":
                    raise SystemExit(experiment_main(pipe))
                """
            ),
            encoding="utf-8",
        )
        base_env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "POISSON_CACHE_DIR": str(self.tmp / "sub_cache"),
            "POISSON_SHARD_TOTAL": "99",  # inherited value must NOT override --shard-total
        }
        submit_sh = str(REPO_ROOT / "scripts/delftblue/submit.sh")

        # Reserved sbatch options must be rejected.
        for bad in ("--export=NONE", "--dependency=afterany:1", "--array=0-1", "--parsable"):
            r = subprocess.run(
                ["bash", submit_sh, str(exp), "--", bad],
                env=base_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn("reserved", r.stderr)

        # Normal submission strips `;cluster` from --parsable IDs, uses compute-p1 / 3900M, and passes --shard-total.
        r = subprocess.run(
            ["bash", submit_sh, str(exp), "--warmup"],
            env=base_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = sbatch_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("--partition=compute-p1", lines[0])
        self.assertIn("--mem-per-cpu=3900M", lines[0])
        self.assertIn("--dependency=afterok:1001", lines[1])
        self.assertNotIn("1001;delftblue", lines[1])
        self.assertIn("--shard-total 3", lines[1])
        self.assertIn("--dependency=afterok:1002", lines[2])
        self.assertNotIn("1002;delftblue", lines[2])
        self.assertIn("/results/exp/1002", lines[2])


    def test_shared_cached_input_touched_by_another_process_mid_stage(self) -> None:
        # Shard A runs a slow stage on a cached object while shard B gets a cache hit on the
        # stage that produced it (which touches the object for LRU) and re-publishes the same
        # content. Neither may make A believe its input changed.
        cache_root = self.tmp / "shared_cache"
        started = self.tmp / "started"
        producer = _write_executable(
            self.tmp / "producer",
            """
import sys
open(sys.argv[sys.argv.index("--out") + 1], "w").write("shared")
""",
        )
        slow = _write_executable(
            self.tmp / "slow",
            f"""
import pathlib, sys, time
args = sys.argv
pathlib.Path({str(started)!r}).write_text("x")
time.sleep(1.0)
open(args[args.index("--out") + 1], "w").write(open(args[args.index("--in") + 1]).read() + "!")
""",
        )
        script = self.tmp / "shared_experiment.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(REPO_ROOT / "python")!r})
                from poisson_runner import Cache, Stage, run_experiment
                produce = Stage(name="produce", executable={str(producer)!r},
                                build_command=lambda call: [str(call.executable), "--out", str(call.outputs["out"])],
                                output_specs={{"out": ".txt"}})
                consume = Stage(name="consume", executable={str(slow)!r},
                                build_command=lambda call: [str(call.executable), "--in", str(call.inputs["x"].path),
                                                            "--out", str(call.outputs["out"])],
                                output_specs={{"out": ".txt"}})
                cache = Cache({str(cache_root)!r}, lock_mode="lockfile")
                if sys.argv[1] == "consume":
                    report = run_experiment(lambda run: run.run_stage(consume, inputs={{"x": run.run_stage(produce)}}), cache=cache)
                else:
                    mode = "refresh" if sys.argv[1] == "refresh" else None
                    report = run_experiment(lambda run: run.run_stage(produce), cache=cache, cache_mode=mode)
                sys.exit(0 if all(b.ok for b in report.branches) else 1)
                """
            ),
            encoding="utf-8",
        )
        env = {**os.environ, "POISSON_BUILD_DIR": os.environ.get("POISSON_BUILD_DIR", "")}
        consumer = subprocess.Popen([sys.executable, str(script), "consume"], env=env, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 30
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(started.exists(), "the slow stage never started")
            for mode in ("hit", "refresh"):   # a lookup touching the object; a re-publication of it
                other = subprocess.run([sys.executable, str(script), mode], env=env, capture_output=True, text=True)
                self.assertEqual(other.returncode, 0, other.stderr)
            self.assertEqual(consumer.wait(timeout=30), 0, consumer.stderr.read() if consumer.stderr else "")
        finally:
            if consumer.poll() is None:
                consumer.kill()
                consumer.wait()

if __name__ == "__main__":
    unittest.main()
