"""End-to-end tests for BLAKE3 hashing, poisson_import CLI, content-addressed cache, and Compare()."""

from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import (  # noqa: E402
    Artifact,
    Cache,
    Compare,
    LocalExecutor,
    RunContext,
    Stage,
    StageCall,
    StageTrace,
    blake3_bytes,
    blake3_file,
    blake3_version,
    compare,
    run_experiment,
)
from poisson_runner.cli import main as cli_main  # noqa: E402

SMALL_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 7 "0V"
2 9 "D0"
3 5 "siliconnitride"
3 2 "vacuum"
$EndPhysicalNames
$Nodes
5
10 0 0 0
20 1 0 0
30 0 1 0
40 0 0 1
50 0 0 -1
$EndNodes
$Elements
4
1 2 2 7 1 10 20 40
2 2 2 9 1 10 30 50
3 4 2 5 1 10 20 30 40
4 4 2 2 2 10 20 30 50
$EndElements
"""

SMALL_MATERIAL = """name: siliconnitride
dielectric:
    epsilon_r: 7.5
"""


def write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class RunnerTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_runner_test_")
        self.tmp = Path(self._tmp.name)
        self.cache_dir = self.tmp / "cache"
        self.mesh_file = self.tmp / "small.msh"
        self.mesh_file.write_text(SMALL_MESH, encoding="utf-8")
        self.mat_file = self.tmp / "siliconnitride.yaml"
        self.mat_file.write_text(SMALL_MATERIAL, encoding="utf-8")
        self.cha_file = self.tmp / "small.cha"
        self.cha_file.write_bytes(struct.pack("<fffi", 0.1, 0.1, 0.1, 1) + struct.pack("<fffi", 0.1, 0.1, -0.5, -1))
        self.pri_file = self.tmp / "small.pri"
        self.pri_file.write_bytes(struct.pack("<fffffffii", 1.0, 2.0, 35.0, 0.0, 0.0, -1.0, 500.0, 3, 4))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_blake3_vectors(self) -> None:
        self.assertTrue(blake3_version().startswith("1."))
        vectors = {
            0: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            1: "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
            64: "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
            1024: "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
            1025: "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
            102400: "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085",
        }
        for length, expected in vectors.items():
            payload = bytes(i % 251 for i in range(length))
            self.assertEqual(blake3_bytes(payload), expected)
            f = self.tmp / f"vec_{length}.bin"
            f.write_bytes(payload)
            self.assertEqual(blake3_file(f), expected)

    def test_real_import_mesh_and_free_charges_caching(self) -> None:
        cache = Cache(self.cache_dir)

        def pipe(run: RunContext) -> tuple[Artifact, Artifact]:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file], charges=self.cha_file)
            primaries = run.import_free_charges(self.pri_file)
            return mesh, primaries

        r1 = run_experiment(pipe, cache=cache)
        self.assertEqual(len(r1.branches), 1)
        self.assertEqual(r1.cache_misses, 2)
        self.assertEqual(r1.cache_hits, 0)

        mesh_art, pri_art = r1.branches[0].result
        self.assertFalse(mesh_art.cached)
        self.assertGreater(mesh_art.info["time"], 0.0)
        self.assertEqual(len(mesh_art.info["warnings"]), 1)
        self.assertIsNotNone(mesh_art.execution["max_rss_kb"])

        # Inspect deterministic HDF5 directly from Python via h5py.
        with h5py.File(mesh_art.path, "r") as h5:
            self.assertEqual(h5.attrs["schema"].decode("utf-8") if isinstance(h5.attrs["schema"], bytes) else h5.attrs["schema"], "MeshProperties")
            self.assertEqual(h5["nodes"].shape, (5, 3))
            self.assertEqual(h5["elements/node"].shape, (2, 4))
            self.assertEqual(h5["charges/value"].shape, (2,))
        with h5py.File(pri_art.path, "r") as h5:
            self.assertEqual(h5["electrons/energy"].shape, (1,))
            self.assertAlmostEqual(float(h5["electrons/energy"][0]), 500.0)

        with h5py.File(mesh_art.path, "r") as h5:
            self.assertEqual(int(h5.attrs["screened"]), 1)   # screening is on by default

        # Turning screening off is a different request with its own result.
        unscreened = run_experiment(
            lambda run: run.import_mesh(self.mesh_file, materials=[self.mat_file], charges=self.cha_file, screening=False),
            cache=cache,
        ).branches[0].result
        self.assertNotEqual(unscreened.content_id, mesh_art.content_id)
        with h5py.File(unscreened.path, "r") as h5:
            self.assertEqual(int(h5.attrs["screened"]), 0)

        # Second run hits cache immediately without re-executing poisson_import.
        r2 = run_experiment(pipe, cache=cache)
        self.assertEqual(r2.cache_hits, 2)
        self.assertEqual(r2.cache_misses, 0)
        self.assertEqual(r2.branches[0].result[0].content_id, mesh_art.content_id)

        # Modifying the input mesh invalidates import_mesh automatically, while free_charges stays cached!
        self.mesh_file.write_text(SMALL_MESH.replace("50 0 0 -1", "50 0 0 -2"), encoding="utf-8")
        r3 = run_experiment(pipe, cache=cache)
        self.assertEqual(r3.cache_misses, 1)
        self.assertEqual(r3.cache_hits, 1)
        self.assertNotEqual(r3.branches[0].result[0].content_id, mesh_art.content_id)
        self.assertEqual(r3.branches[0].result[1].content_id, pri_art.content_id)

    def test_compare_branching_nested_if_and_multi_input_stages(self) -> None:
        """Verifies loop-free Compare(), conditional Compare inside `if`, multi-input stages, and HIT counts."""
        log_file = self.tmp / "executions.log"

        asm_bin = write_executable(
            self.tmp / "mock_assemble",
            f"""
import argparse, json, pathlib, h5py
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--mode", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log_file)!r}, "a") as f: f.write(f"assemble:{{a.mode}}\\n")
with h5py.File(a.inp, "r") as src:
    n_nodes = src["nodes"].shape[0]
payload = json.dumps({{"stage": "assemble", "mode": a.mode, "n_nodes": n_nodes}}, sort_keys=True)
pathlib.Path(a.out).write_text(payload)
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.12, "nnz": n_nodes * 4}}))
""",
        )

        sol_bin = write_executable(
            self.tmp / "mock_solve",
            f"""
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--method", required=True)
p.add_argument("--smoother", default="none")
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log_file)!r}, "a") as f: f.write(f"solve:{{a.method}}:{{a.smoother}}\\n")
asm_data = json.loads(pathlib.Path(a.inp).read_text())
payload = json.dumps({{"asm": asm_data, "method": a.method, "smoother": a.smoother}}, sort_keys=True)
pathlib.Path(a.out).write_text(payload)
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.25, "iterations": 10, "residual": 1e-9}}))
""",
        )

        sb_bin = write_executable(
            self.tmp / "mock_solution_builder",
            f"""
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--assembled", required=True)
p.add_argument("--solved", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log_file)!r}, "a") as f: f.write("solution_builder\\n")
d1 = json.loads(pathlib.Path(a.assembled).read_text())
d2 = json.loads(pathlib.Path(a.solved).read_text())
pathlib.Path(a.out).write_text(json.dumps({{"assembled": d1, "solved": d2}}, sort_keys=True))
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.03}}))
""",
        )

        assemble_stage = Stage(
            name="assemble",
            executable=asm_bin,
            output_specs={"out": ".json"},
            build_command=lambda c: [
                str(c.executable),
                "--in", str(c.inputs["mesh"].path),  # type: ignore[union-attr]
                "--mode", str(c.settings["mode"]),
                "--out", str(c.outputs["out"]),
                "--info-out", str(c.info_path),
            ],
        )
        solve_stage = Stage(
            name="solve",
            executable=sol_bin,
            output_specs={"out": ".json"},
            build_command=lambda c: [
                str(c.executable),
                "--in", str(c.inputs["problem"].path),  # type: ignore[union-attr]
                "--method", str(c.settings["method"]),
                "--smoother", str(c.settings.get("smoother", "none")),
                "--out", str(c.outputs["out"]),
                "--info-out", str(c.info_path),
            ],
        )
        sb_stage = Stage(
            name="solution_builder",
            executable=sb_bin,
            output_specs={"out": ".json"},
            build_command=lambda c: [
                str(c.executable),
                "--assembled", str(c.inputs["assembled"].path),  # type: ignore[union-attr]
                "--solved", str(c.inputs["solved"].path),  # type: ignore[union-attr]
                "--out", str(c.outputs["out"]),
                "--info-out", str(c.info_path),
            ],
        )

        def pipeline(run: RunContext) -> Artifact:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            asm_mode = compare("assembly", ["P1", "P2"])
            problem = run.run_stage(assemble_stage, inputs={"mesh": mesh}, settings={"mode": asm_mode})
            sol_mode = compare(Compare(["CG", "AMG"], name="solver"))
            if sol_mode == "AMG":
                smoother = compare("smoother", ["jacobi", "chebyshev"])
                solved = run.run_stage(
                    solve_stage,
                    inputs={"problem": problem},
                    settings={"method": sol_mode, "smoother": smoother},
                )
            else:
                solved = run.run_stage(
                    solve_stage,
                    inputs={"problem": problem},
                    settings={"method": sol_mode},
                )
            # Multi-input stage combining AssembledProblem + SolverResult!
            solution = run.run_stage(
                sb_stage,
                inputs={"assembled": problem, "solved": solved},
            )
            assert isinstance(solution, Artifact)
            return solution

        cache = Cache(self.cache_dir)
        report = run_experiment(pipeline, cache=cache)

        # 2 assembly modes * (1 CG + 2 AMG smoothers) = 6 total branches!
        self.assertEqual(len(report.branches), 6)
        executed = log_file.read_text(encoding="utf-8").splitlines()
        # Import runs 1x, Assemble runs 2x, Solve runs 6x, SolutionBuilder runs 6x.
        self.assertEqual(executed.count("assemble:P1"), 1)
        self.assertEqual(executed.count("assemble:P2"), 1)
        self.assertEqual(sum(1 for line in executed if line.startswith("solve:")), 6)
        self.assertEqual(executed.count("solution_builder"), 6)
        # Across 6 branches * 4 stages = 24 stage calls: 1+2+6+6 = 15 misses, 9 hits!
        self.assertEqual(report.cache_misses, 15)
        self.assertEqual(report.cache_hits, 9)

        # Export reports cleanly.
        table = report.format_table()
        self.assertIn("assembly", table)
        self.assertIn("smoother", table)
        json_out = self.tmp / "report.json"
        csv_out = self.tmp / "report.csv"
        report.save_json(json_out)
        report.save_csv(csv_out)
        self.assertEqual(json.loads(json_out.read_text())["branch_count"], 6)
        self.assertIn("solve_status", csv_out.read_text())

    def test_content_addressed_dedup_and_binary_hash_invalidation(self) -> None:
        """Proves:
        1. Editing stage A's binary invalidates stage A (auto binary hash).
        2. If stage A still writes identical output bytes, stage B gets a CACHE HIT (content addressing)!
        3. Only one copy of the identical output object is stored in cache/objects/.
        """
        counter = self.tmp / "counts.txt"
        stage_a_bin = write_executable(
            self.tmp / "stage_a",
            f"""
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
with open({str(counter)!r}, "a") as f: f.write("A\\n")
# v1 comment
pathlib.Path(a.out).write_bytes(b"IDENTICAL_OUTPUT_PAYLOAD")
""",
        )
        stage_b_bin = write_executable(
            self.tmp / "stage_b",
            f"""
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
with open({str(counter)!r}, "a") as f: f.write("B\\n")
data = pathlib.Path(a.inp).read_bytes()
pathlib.Path(a.out).write_bytes(b"DOWNSTREAM:" + data)
""",
        )

        stage_a = Stage(
            name="stage_a",
            executable=stage_a_bin,
            output_specs={"out": ".bin"},
            build_command=lambda c: [str(c.executable), "--in", str(c.inputs["raw"].path), "--out", str(c.outputs["out"])],  # type: ignore[union-attr]
        )
        stage_b = Stage(
            name="stage_b",
            executable=stage_b_bin,
            output_specs={"out": ".bin"},
            build_command=lambda c: [str(c.executable), "--in", str(c.inputs["a"].path), "--out", str(c.outputs["out"])],  # type: ignore[union-attr]
        )

        def pipe(run: RunContext) -> Artifact:
            a_out = run.run_stage(stage_a, inputs={"raw": self.mesh_file})
            b_out = run.run_stage(stage_b, inputs={"a": a_out})
            assert isinstance(b_out, Artifact)
            return b_out

        cache = Cache(self.cache_dir)
        r1 = run_experiment(pipe, cache=cache)
        self.assertEqual(r1.cache_misses, 2)
        self.assertEqual(counter.read_text().splitlines(), ["A", "B"])

        # Modify stage_a's binary (simulating a refactor/recompile) while keeping identical output bytes.
        time.sleep(0.02)
        write_executable(
            stage_a_bin,
            f"""
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
with open({str(counter)!r}, "a") as f: f.write("A\\n")
# v2 refactored implementation producing identical bytes!
pathlib.Path(a.out).write_bytes(b"IDENTICAL_OUTPUT_PAYLOAD")
""",
        )

        r2 = run_experiment(pipe, cache=cache)
        # Stage A MUST re-run (binary hash changed), Stage B MUST HIT (A's output content_id is identical)!
        self.assertEqual(r2.cache_misses, 1)
        self.assertEqual(r2.cache_hits, 1)
        self.assertEqual(counter.read_text().splitlines(), ["A", "B", "A"])
        # Only 2 objects in cache/objects/ (deduplicated A output + B output).
        self.assertEqual(cache.stats().object_count, 2)

    def test_atomic_failure_and_lru_eviction_and_cli(self) -> None:
        # 1. Failing stage leaves zero partial artifacts or temp files.
        fail_bin = write_executable(
            self.tmp / "fail_stage",
            """
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
a = p.parse_args()
pathlib.Path(a.out).write_bytes(b"PARTIAL_CORRUPT_DATA")
sys.stderr.write("boom\\n")
sys.exit(2)
""",
        )
        fail_stage = Stage(
            name="failing",
            executable=fail_bin,
            output_specs={"out": ".bin"},
            build_command=lambda c: [str(c.executable), "--out", str(c.outputs["out"])],
        )
        cache = Cache(self.cache_dir)
        report = run_experiment(
            lambda run: run.run_stage(fail_stage),
            cache=cache,
            continue_on_error=True,
        )
        self.assertIsNotNone(report.branches[0].error)
        self.assertEqual(cache.stats().object_count, 0)
        self.assertEqual(cache.stats().request_count, 0)
        self.assertEqual(cache.stats().temp_count, 0)

        # 2. LRU eviction respects max_bytes.
        gen_bin = write_executable(
            self.tmp / "gen_blob",
            """
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
pathlib.Path(a.out).write_bytes((a.tag.encode() * 1024)[:1024])
""",
        )
        gen_stage = Stage(
            name="gen",
            executable=gen_bin,
            output_specs={"out": ".bin"},
            build_command=lambda c: [str(c.executable), "--tag", str(c.settings["tag"]), "--out", str(c.outputs["out"])],
        )
        small_cache = Cache(self.tmp / "lru_cache", max_bytes=2048)
        for tag in ("A", "B", "C"):
            run_experiment(lambda run, t=tag: run.run_stage(gen_stage, settings={"tag": t}), cache=small_cache)
            time.sleep(0.02)
        st = small_cache.stats()
        self.assertLessEqual(st.object_bytes, 2048)
        self.assertEqual(st.object_count, 2)

        # 3. CLI commands work cleanly.
        self.assertEqual(cli_main(["hash", str(self.mesh_file)]), 0)
        self.assertEqual(cli_main(["cache", "--cache-dir", str(small_cache.root), "status"]), 0)
        self.assertEqual(cli_main(["cache", "--cache-dir", str(small_cache.root), "inspect"]), 0)
        self.assertEqual(cli_main(["cache", "--cache-dir", str(small_cache.root), "clean", "--all"]), 0)
        self.assertEqual(small_cache.stats().object_count, 0)

    def test_review_hardening_regressions(self) -> None:
        from poisson_runner.stages import resolve_binary

        # 1. CLI rejects --out aliasing input or --info-out aliasing --out.
        imp_bin = resolve_binary("poisson_import")
        res1 = subprocess.run(
            [str(imp_bin), "--mesh", str(self.mesh_file), "--out", str(self.mesh_file)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res1.returncode, 0)
        self.assertIn("aliases input", res1.stderr)

        out_h5 = self.tmp / "same_out.h5"
        res2 = subprocess.run(
            [str(imp_bin), "--mesh", str(self.mesh_file), "--out", str(out_h5), "--info-out", str(out_h5)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res2.returncode, 0)
        self.assertIn("must not alias --out", res2.stderr)

        # 2. Identical bytes with .yaml vs .mat must NOT collide in cache (parser depends on extension).
        cache = Cache(self.cache_dir)
        fake_mat = self.tmp / "siliconnitride.mat"
        fake_mat.write_bytes(self.mat_file.read_bytes())
        r_yaml = run_experiment(lambda run: run.import_mesh(self.mesh_file, materials=[self.mat_file]), cache=cache)
        self.assertEqual(r_yaml.cache_misses, 1)
        with self.assertRaises(RuntimeError) as ctx_err:
            run_experiment(lambda run: run.import_mesh(self.mesh_file, materials=[fake_mat]), cache=cache)
        self.assertIn("not a Nebula material file", str(ctx_err.exception))

        # 3. Changing output_specs changes request_key (no wrong-shape cache hit).
        out_bin = write_executable(
            self.tmp / "multi_out",
            """
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", default=None)
a = p.parse_args()
pathlib.Path(a.a).write_bytes(b"AAA")
if a.b:
    pathlib.Path(a.b).write_bytes(b"BBB")
""",
        )
        s_one = Stage(
            name="mstage",
            executable=out_bin,
            output_specs={"out": ".bin"},
            build_command=lambda c: [str(c.executable), "--a", str(c.outputs["out"])],
        )
        s_two = Stage(
            name="mstage",
            executable=out_bin,
            output_specs={"first": ".bin", "second": ".bin"},
            build_command=lambda c: [
                str(c.executable), "--a", str(c.outputs["first"]), "--b", str(c.outputs["second"])
            ],
        )
        r_one = run_experiment(lambda run: run.run_stage(s_one), cache=cache)
        r_two = run_experiment(lambda run: run.run_stage(s_two), cache=cache)
        self.assertEqual(r_one.cache_misses, 1)
        self.assertEqual(r_two.cache_misses, 1)
        self.assertIsInstance(r_two.branches[0].result, dict)
        self.assertEqual(set(r_two.branches[0].result.keys()), {"first", "second"})

        # 4. Published cached objects are read-only; truncated cached object is rejected on lookup and recomputed.
        art_one: Artifact = r_one.branches[0].result
        self.assertEqual(stat.S_IMODE(art_one.path.stat().st_mode), 0o444)
        art_one.path.chmod(0o644)
        art_one.path.write_bytes(b"A")   # corrupt/truncate from 3 bytes to 1 byte
        r_rebuilt = run_experiment(lambda run: run.run_stage(s_one), cache=cache)
        self.assertEqual(r_rebuilt.cache_misses, 1)
        self.assertEqual(r_rebuilt.branches[0].result.path.read_bytes(), b"AAA")
        self.assertEqual(stat.S_IMODE(r_rebuilt.branches[0].result.path.stat().st_mode), 0o444)

        # 4b. Same-size corruption is healed by cache="refresh" (store always os.replaces).
        healed_path = r_rebuilt.branches[0].result.path
        healed_path.chmod(0o644)
        healed_path.write_bytes(b"ZZZ")
        r_refreshed = run_experiment(lambda run: run.run_stage(s_one, cache="refresh"), cache=cache)
        self.assertEqual(r_refreshed.branches[0].result.path.read_bytes(), b"AAA")
        self.assertEqual(stat.S_IMODE(r_refreshed.branches[0].result.path.stat().st_mode), 0o444)

        # 5. Live CacheLease protects active artifacts from LRU eviction, and releasing allows eviction.
        lease_cache = Cache(self.tmp / "lease_cache", max_bytes=1024)
        with lease_cache.lease() as active_lease:
            r_pin = run_experiment(lambda run: run.run_stage(s_one), cache=lease_cache)
            pinned_art: Artifact = r_pin.branches[0].result
            active_lease.pin([pinned_art.path.name])
            self.assertEqual(lease_cache.evict_lru(max_bytes=0), 0)
            self.assertTrue(pinned_art.path.is_file())
        # After lease closes, eviction reclaims it.
        self.assertEqual(lease_cache.evict_lru(max_bytes=0), 1)
        self.assertFalse(pinned_art.path.is_file())

        # 6. Compare catches duplicate names, non-deterministic option reordering, and unconsumed replay paths.
        with self.assertRaises(ValueError):
            run_experiment(lambda run: (compare("x", [1, 2]), compare("x", [3, 4])), cache=cache)

        calls = {"n": 0}
        def nondet_reorder(run: RunContext) -> str:
            calls["n"] += 1
            opts = ["a", "b"] if calls["n"] == 1 else ["b", "a"]
            return compare("order", opts)

        with self.assertRaises(RuntimeError):
            run_experiment(nondet_reorder, cache=cache)

        calls2 = {"n": 0}
        def early_return(run: RunContext) -> str:
            calls2["n"] += 1
            if calls2["n"] == 2:
                return "early"
            return compare("late", ["x", "y"])

        with self.assertRaises(RuntimeError):
            run_experiment(early_return, cache=cache)

        # 7. Non-string dict keys in settings are rejected to avoid hash collisions.
        with self.assertRaises(TypeError):
            run_experiment(lambda run: run.run_stage(s_one, settings={1: "a"}), cache=cache)  # type: ignore[dict-item]

    def test_four_cache_modes_and_layered_precedence(self) -> None:
        """Verifies all 4 cache modes (use, read_only, refresh, off) and 4-layer precedence."""
        counter_file = self.tmp / "mode_count.txt"
        stage_bin = write_executable(
            self.tmp / "counter_stage",
            f"""
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
cfile = pathlib.Path({str(counter_file)!r})
n = int(cfile.read_text()) + 1 if cfile.exists() else 1
cfile.write_text(str(n))
pathlib.Path(a.out).write_text(f"{{a.tag}}:run={{n}}")
""",
        )

        def make_stage(name: str, default_cache: str = "use") -> Stage:
            return Stage(
                name=name,
                executable=stage_bin,
                output_specs={"out": ".txt"},
                default_cache=default_cache,
                build_command=lambda c: [
                    str(c.executable), "--tag", str(c.settings["tag"]), "--out", str(c.outputs["out"])
                ],
            )

        cache = Cache(self.tmp / "modes_cache")
        s_use = make_stage("s_use", default_cache="use")
        s_off = make_stage("s_off", default_cache="off")

        # 1. read_only on empty cache: MISS, runs, does NOT write to cache.
        r_ro_miss = run_experiment(
            lambda run: run.run_stage(s_use, settings={"tag": "t1"}, cache="read_only"),
            cache=cache,
        )
        self.assertEqual(r_ro_miss.cache_misses, 1)
        self.assertEqual(cache.stats().object_count, 0)
        self.assertEqual(r_ro_miss.branches[0].result.path.read_text(), "t1:run=1")

        # 2. use populates cache (run=2), then read_only gets a HIT returning run=2!
        r_pop = run_experiment(lambda run: run.run_stage(s_use, settings={"tag": "t1"}), cache=cache)
        self.assertEqual(r_pop.cache_misses, 1)
        self.assertEqual(cache.stats().object_count, 1)
        r_ro_hit = run_experiment(
            lambda run: run.run_stage(s_use, settings={"tag": "t1"}, cache="read_only"),
            cache=cache,
        )
        self.assertEqual(r_ro_hit.cache_hits, 1)
        self.assertEqual(r_ro_hit.branches[0].result.path.read_text(), "t1:run=2")

        # 3. refresh ignores cached run=2, re-runs (run=3), and overwrites cache so next `use` hits run=3!
        r_ref = run_experiment(
            lambda run: run.run_stage(s_use, settings={"tag": "t1"}, cache="refresh"),
            cache=cache,
        )
        self.assertEqual(r_ref.cache_misses, 1)
        self.assertEqual(r_ref.branches[0].result.path.read_text(), "t1:run=3")
        r_after_ref = run_experiment(lambda run: run.run_stage(s_use, settings={"tag": "t1"}), cache=cache)
        self.assertEqual(r_after_ref.cache_hits, 1)
        self.assertEqual(r_after_ref.branches[0].result.path.read_text(), "t1:run=3")

        # 4. off ignores cache and does not overwrite it (runs run=4, cache still serves run=3).
        r_off = run_experiment(
            lambda run: run.run_stage(s_use, settings={"tag": "t1"}, cache="off"),
            cache=cache,
        )
        self.assertEqual(r_off.cache_misses, 1)
        self.assertEqual(r_off.branches[0].result.path.read_text(), "t1:run=4")
        r_still_3 = run_experiment(lambda run: run.run_stage(s_use, settings={"tag": "t1"}), cache=cache)
        self.assertEqual(r_still_3.cache_hits, 1)
        self.assertEqual(r_still_3.branches[0].result.path.read_text(), "t1:run=3")

        # 5. Layered precedence:
        #    Stage default ("off") < experiment cache_mode ("use") < stage_cache_modes ("refresh") < per-call ("read_only").
        #    a) Stage default "off" never caches on its own:
        r_def_off_1 = run_experiment(lambda run: run.run_stage(s_off, settings={"tag": "p"}), cache=cache)
        r_def_off_2 = run_experiment(lambda run: run.run_stage(s_off, settings={"tag": "p"}), cache=cache)
        self.assertEqual(r_def_off_1.cache_misses, 1)
        self.assertEqual(r_def_off_2.cache_misses, 1)

        #    b) Experiment cache_mode="use" overrides Stage default "off":
        r_exp_1 = run_experiment(lambda run: run.run_stage(s_off, settings={"tag": "p"}), cache=cache, cache_mode="use")
        r_exp_2 = run_experiment(lambda run: run.run_stage(s_off, settings={"tag": "p"}), cache=cache, cache_mode="use")
        self.assertEqual(r_exp_1.cache_misses, 1)
        self.assertEqual(r_exp_2.cache_hits, 1)

        #    c) Experiment stage_cache_modes overrides experiment cache_mode:
        r_stage_override = run_experiment(
            lambda run: run.run_stage(s_off, settings={"tag": "p"}),
            cache=cache,
            cache_mode="use",
            stage_cache_modes={"s_off": "refresh"},
        )
        self.assertEqual(r_stage_override.cache_misses, 1)

        #    d) Per-call cache="read_only" overrides stage_cache_modes={"s_off": "refresh"}:
        r_call_override = run_experiment(
            lambda run: run.run_stage(s_off, settings={"tag": "p"}, cache="read_only"),
            cache=cache,
            cache_mode="use",
            stage_cache_modes={"s_off": "refresh"},
        )
        self.assertEqual(r_call_override.cache_hits, 1)

        # 6. Invalid cache mode raises ValueError.
        with self.assertRaises(ValueError):
            run_experiment(lambda run: run.run_stage(s_use, settings={"tag": "t1"}, cache="bogus"), cache=cache)

    def test_repeats_os_metrics_from_cli_config_tidy_csv_and_dry_run(self) -> None:
        """Verifies #1 (OS metrics), #2 (repeats/warmup stats), #5 (Stage.from_cli + --config),
        #6 (tidy/loop-friendly CSV), and #7 (dry_run preview)."""
        import csv

        exec_log = self.tmp / "cfg_exec.log"
        cfg_bin = write_executable(
            self.tmp / "cfg_stage",
            f"""
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
cfg = json.loads(pathlib.Path(a.config).read_text())
with open({str(exec_log)!r}, "a") as f:
    f.write(json.dumps(cfg, sort_keys=True) + "\\n")
inp_bytes = pathlib.Path(a.inp).read_bytes()
pathlib.Path(a.out).write_text(json.dumps({{"len": len(inp_bytes), "cfg": cfg}}, sort_keys=True))
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.015}}))
""",
        )

        # #5: Declarative Stage.from_cli with --config for nested settings!
        cfg_stage = Stage.from_cli(
            "adapt",
            binary=cfg_bin,
            input_flags={"mesh": "--in"},
            config_flag="--config",
            output_flags={"out": "--out"},
            output_specs={"out": ".json"},
            info_flag="--info-out",
        )

        cache = Cache(self.tmp / "metrics_cache")

        def loop_pipeline(run: RunContext) -> Artifact:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            estimator = compare("estimator", ["kelly", "residual"])
            cur = mesh
            for cycle in range(2):
                cur = run.run_stage(
                    cfg_stage,
                    inputs={"mesh": cur},
                    settings={
                        "cycle": cycle,
                        "estimator": {"type": estimator, "quadrature": 4},
                        "marker": {"type": "dorfler", "theta": 0.5},
                    },
                )  # type: ignore[assignment]
            return cur

        # #7: dry_run on cold cache executes NOTHING: 5 WOULD_RUN stage calls, plus the shared
        # import on the 2nd branch which is REUSED from the 1st branch's planned run.
        dry_cold = run_experiment(loop_pipeline, cache=cache, dry_run=True)
        self.assertTrue(dry_cold.dry_run)
        self.assertEqual(len(dry_cold.branches), 2)
        self.assertEqual(dry_cold.cache_hits, 0)
        self.assertEqual(dry_cold.cache_misses, 5)
        self.assertEqual(dry_cold.reused, 1)
        self.assertEqual(dry_cold.branches[1].stages[0].status, "REUSED")
        self.assertFalse(exec_log.exists())
        self.assertIn("WOULD_RUN", dry_cold.format_table())

        # #1 + #2: Real run with warmup=1, repeats=3 records full stats + OS metrics.
        real = run_experiment(loop_pipeline, cache=cache, repeats=3, warmup=1)
        self.assertEqual(real.cache_misses, 5)   # 1 import + 2 kelly cycles + 2 residual cycles
        self.assertEqual(real.cache_hits, 1)     # 2nd branch hits import!
        # 4 adapt stage misses * (1 warmup + 3 measured) = 16 executions of cfg_bin.
        self.assertEqual(len(exec_log.read_text().splitlines()), 16)

        adapt_trace = real.branches[0].stages[1]
        self.assertIsNotNone(adapt_trace.execution["cpu_user_s"])
        self.assertIsNotNone(adapt_trace.execution["cpu_sys_s"])
        self.assertIsNotNone(adapt_trace.execution["cpu_efficiency"])
        self.assertIsNotNone(adapt_trace.execution["major_page_faults"])
        bench = adapt_trace.execution["benchmark"]
        self.assertEqual(bench["repeats"], 3)
        self.assertEqual(bench["warmup"], 1)
        for stat_key in ("min", "median", "mean", "max", "std", "samples"):
            self.assertIn(stat_key, bench["wall_time_s"])
            self.assertIn(stat_key, bench["info_time_s"])
        self.assertEqual(len(bench["wall_time_s"]["samples"]), 3)
        self.assertAlmostEqual(bench["info_time_s"]["median"], 0.015)

        # #7: dry_run on warm cache reports 100% HITs (6 hits, 0 would-run) without executing anything.
        dry_warm = run_experiment(loop_pipeline, cache=cache, repeats=3, warmup=1, dry_run=True)
        self.assertEqual(dry_warm.cache_hits, 6)
        self.assertEqual(dry_warm.cache_misses, 0)
        self.assertEqual(len(exec_log.read_text().splitlines()), 16)

        # #6: Both long (tidy) and wide (disambiguated loop stages) CSV exports work cleanly.
        long_csv = self.tmp / "long.csv"
        wide_csv = self.tmp / "wide.csv"
        real.save_csv(long_csv, layout="long")
        real.save_csv(wide_csv, layout="wide")

        with long_csv.open(encoding="utf-8") as fp:
            long_rows = list(csv.DictReader(fp))
        self.assertEqual(len(long_rows), 6)   # 2 branches * 3 stage calls each
        self.assertEqual(long_rows[1]["repeats"], "3")
        self.assertNotEqual(long_rows[1]["wall_median_s"], "")
        self.assertNotEqual(long_rows[1]["cpu_efficiency"], "")

        with wide_csv.open(encoding="utf-8") as fp:
            wide_reader = csv.DictReader(fp)
            wide_headers = wide_reader.fieldnames or []
            wide_rows = list(wide_reader)
        self.assertEqual(len(wide_rows), 2)
        self.assertIn("adapt#0_status", wide_headers)
        self.assertIn("adapt#1_status", wide_headers)

    def test_real_poisson_adapt_cli_and_compare(self) -> None:
        """End-to-end test: real poisson_import -> real poisson_adapt CLI comparing estimators & refiners,
        plus 2-cycle red-green refinement with persisted RefinementHistory."""
        import numpy as np

        cache = Cache(self.tmp / "adapt_cache")

        # Write a deterministic NodalField HDF5 for the 5-node small mesh: u(x,y,z) = x^2 + y^2 + z^2.
        def write_nodal_field_h5(path: Path, nodes: np.ndarray) -> Path:
            u = np.sum(nodes**2, axis=1).astype("<f8")
            with h5py.File(path, "w") as h5:
                h5.attrs.create("schema", b"NodalField", dtype="S10")
                h5.attrs.create("schema_version", np.uint32(1), dtype="<u4")
                h5.create_dataset("u", data=u, dtype="<f8")
                g = h5.create_group("point_sources")
                g.create_dataset("position", data=np.zeros((0, 3), dtype="<f8"), dtype="<f8")
                g.create_dataset("value", data=np.zeros((0,), dtype="<f8"), dtype="<f8")
                g.create_dataset("element", data=np.zeros((0,), dtype="<u4"), dtype="<u4")
            return path

        # Each material of the small mesh is one tet, so per-material recovery is exactly 0 there;
        # averaging across the interface gives it something to measure.
        config_for = lambda est, ref, **adapter: {  # noqa: E731
            "estimator": {"type": est, **({"per_material": False} if est == "recovery" else {})},
            "marker": {"type": "dorfler", "theta": 0.5},
            "refiner": {"type": ref},
            **({"adapter": adapter} if adapter else {}),
        }

        # 1. Run 2x2 Compare across estimators & refiners on real poisson_import + poisson_adapt!
        def amr_compare(run: RunContext) -> dict[str, Artifact]:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            with h5py.File(mesh.path, "r") as h5:
                nodes = h5["nodes"][:]
            sol_path = write_nodal_field_h5(self.tmp / "u_init.h5", nodes)
            est = compare("estimator", ["kelly", "recovery"])
            ref = compare("refiner", ["red_green", "longest_edge_bisection"])
            return run.adapt_mesh(mesh, sol_path, config=config_for(est, ref))

        report = run_experiment(amr_compare, cache=cache)
        self.assertEqual(len(report.branches), 4)
        self.assertEqual(report.cache_misses, 5)   # 1 import + 4 adapt combinations
        self.assertEqual(report.cache_hits, 3)     # import cached on branches 1..3
        for br in report.branches:
            res: dict[str, Artifact] = br.result
            self.assertEqual(set(res), {"mesh", "parents", "state", "eta", "marked"})
            info = res["mesh"].info
            self.assertEqual(info["status"], "adapted")
            self.assertGreater(info["elements_after"], info["elements_before"])
            self.assertEqual(len(info["stages"]), 3)
            self.assertNotIn("error_estimate", info)   # the error is the estimate stage's "total"
            with h5py.File(res["eta"].path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], b"ErrorIndicators")
                eta = h5["eta"][:]
            self.assertEqual(len(eta), info["elements_before"])   # per tet of the INPUT mesh
            with h5py.File(res["marked"].path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], b"MarkedElements")
                marked = h5["marked"][:]
            self.assertEqual(len(marked), int(info["stages"][1]["metrics"]["marked"]))
            self.assertEqual(marked.tolist(), sorted(set(marked.tolist())))
            self.assertTrue(all(0 <= int(k) < info["elements_before"] for k in marked))
            total = info["stages"][0]["metrics"]["total"]   # the estimate stage's error estimate
            self.assertAlmostEqual(float(np.sqrt(np.sum(eta**2))), total, places=12)
            with h5py.File(res["parents"].path, "r") as h5:
                self.assertEqual(len(h5["elements"][:]), info["elements_after"])
            with h5py.File(res["state"].path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], b"AdaptationState")
                self.assertAlmostEqual(float(h5["initial_error"][0]), total, places=12)
                if br.choice_labels["refiner"] == "red_green":
                    self.assertEqual(len(h5["history/family"][:]), info["elements_after"])

        # 2. Multi-cycle red-green chained through `state` across separate CLI processes.
        def cycles(run: RunContext, n: int, **adapter: Any) -> list[dict[str, Artifact]]:
            mesh: Artifact = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            state: Artifact | None = None
            results: list[dict[str, Artifact]] = []
            for cycle in range(n):
                with h5py.File(mesh.path, "r") as h5:
                    nodes = h5["nodes"][:]
                sol_path = write_nodal_field_h5(self.tmp / f"u_c{cycle}_{nodes.shape[0]}.h5", nodes)
                res = run.adapt_mesh(mesh, sol_path, state=state, config=config_for("kelly", "red_green", **adapter))
                results.append(res)
                mesh, state = res["mesh"], res["state"]
            return results

        r_cyc = run_experiment(lambda run: cycles(run, 2), cache=cache)
        two = r_cyc.branches[0].result
        self.assertEqual([r["mesh"].info["status"] for r in two], ["adapted", "adapted"])
        self.assertGreater(two[1]["mesh"].info["elements_after"], two[0]["mesh"].info["elements_after"])
        est_metrics = two[1]["mesh"].info["stages"][0]["metrics"]
        self.assertIn("total", est_metrics)
        self.assertIn("point_sources", est_metrics)
        self.assertNotIn("elements", est_metrics)   # already Info's elements_before
        ref_metrics = two[1]["mesh"].info["stages"][2]["metrics"]
        self.assertIn("nodes_before", ref_metrics)
        self.assertIn("nodes_after", ref_metrics)
        for repeated in ("elements_before", "elements_after", "marked"):   # already in Info / mark stage
            self.assertNotIn(repeated, ref_metrics)
        with h5py.File(two[1]["state"].path, "r") as h5:
            # The baseline stays the FIRST cycle's error, not the second's.
            self.assertAlmostEqual(float(h5["initial_error"][0]), two[0]["mesh"].info["stages"][0]["metrics"]["total"],
                                   places=12)

        # 3. relative_tolerance works across processes via `state`: a huge tolerance converges on
        #    cycle 2 (cycle 1 sets the baseline), and converged cycles pass the mesh through
        #    unchanged with identity parents.
        r_rel = run_experiment(lambda run: cycles(run, 2, relative_tolerance=0.999999), cache=cache)
        first, second = r_rel.branches[0].result
        self.assertEqual(first["mesh"].info["status"], "adapted")
        self.assertEqual(second["mesh"].info["status"], "converged")
        with h5py.File(second["marked"].path, "r") as h5:
            self.assertEqual(h5.attrs["schema"], b"MarkedElements")
            self.assertEqual(len(h5["marked"][:]), 0)
        self.assertEqual(second["mesh"].content_id, first["mesh"].content_id)   # same mesh passed through
        with h5py.File(second["parents"].path, "r") as h5:
            n_el = second["mesh"].info["elements_before"]
            self.assertEqual(h5["elements"][:].tolist(), list(range(n_el)))
            node_pairs = h5["nodes"][:]
            self.assertTrue(np.array_equal(node_pairs[:, 0], node_pairs[:, 1]))
        self.assertEqual(second["state"].content_id, first["state"].content_id)   # state unchanged too

        # 4. CLI validation checks: invalid settings rejected even with --state-out, and
        #    simple_json rejects out-of-range ints cleanly (no UB).
        from poisson_runner.stages import resolve_binary

        adapt_bin = resolve_binary("poisson_adapt")
        mesh_art = run_experiment(lambda run: run.import_mesh(self.mesh_file, materials=[self.mat_file]), cache=cache).branches[0].result
        with h5py.File(mesh_art.path, "r") as h5:
            nodes = h5["nodes"][:]
        sol_path = write_nodal_field_h5(self.tmp / "u_cli_checks.h5", nodes)

        def run_adapt_raw(cfg: dict, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
            cfg_p = self.tmp / "check_cfg.json"
            cfg_p.write_text(json.dumps(cfg), encoding="utf-8")
            cmd = [
                str(adapt_bin),
                "--mesh", str(mesh_art.path),
                "--solution", str(sol_path),
                "--config", str(cfg_p),
                "--out", str(self.tmp / "check_out.h5"),
                *(extra or []),
            ]
            return subprocess.run(cmd, capture_output=True, text=True)

        bad_max = run_adapt_raw(
            {
                "estimator": {"type": "kelly"},
                "marker": {"type": "dorfler"},
                "refiner": {"type": "red_green"},
                "adapter": {"max_elements": 0},
            },
            ["--state-out", str(self.tmp / "bad_state.h5")],
        )
        self.assertNotEqual(bad_max.returncode, 0)
        self.assertIn("max_elements", bad_max.stderr)

        bad_int = run_adapt_raw(
            {
                "estimator": {"type": "residual", "tet_degree": 1e10},
                "marker": {"type": "dorfler"},
                "refiner": {"type": "red_green"},
            }
        )
        self.assertNotEqual(bad_int.returncode, 0)
        self.assertIn("32-bit integer", bad_int.stderr)

    def test_consensus_runner_fixes(self) -> None:
        """Verifies experiment-wide lease, negative max_bytes rejection, ephemeral cleanup, and spec_hash+version."""
        blob_bin = write_executable(
            self.tmp / "blob_stage",
            """
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--extra", default="a")
p.add_argument("--out", required=True)
a = p.parse_args()
pathlib.Path(a.out).write_bytes((f"{a.tag}:{a.extra}:".encode() * 200)[:1024])
""",
        )

        # 1. Experiment-wide lease prevents later branches from evicting earlier branches' outputs.
        tight_cache = Cache(self.tmp / "exp_lease_cache", max_bytes=1500)
        stage_blob = Stage.from_cli(
            "blob",
            binary=blob_bin,
            setting_flags={"tag": "--tag"},
            output_specs={"out": ".bin"},
            info_flag=None,
            version="blob_v1",
        )
        rep = run_experiment(
            lambda run: run.run_stage(stage_blob, settings={"tag": compare("tag", ["t1", "t2", "t3"])}),
            cache=tight_cache,
        )
        for br in rep.branches:
            self.assertTrue(br.result.path.is_file(), f"branch {br.index} artifact was evicted mid-experiment")

        # 2. Negative max_bytes is rejected in API and CLI.
        with self.assertRaises(ValueError):
            tight_cache.evict_lru(max_bytes=-1)
        with self.assertRaises(ValueError):
            tight_cache.clean(max_bytes=-1)
        self.assertNotEqual(cli_main(["cache", "--cache-dir", str(tight_cache.root), "clean", "--max-gb", "-1"]), 0)

        # 3. Ephemeral outputs are tracked in stats() and removed by clean(all_entries=True).
        run_experiment(
            lambda run: run.run_stage(stage_blob, settings={"tag": "eph"}, cache="off"),
            cache=tight_cache,
        )
        st = tight_cache.stats()
        self.assertEqual(st.ephemeral_count, 1)
        self.assertGreater(st.ephemeral_bytes, 0)
        cleaned = tight_cache.clean(all_entries=True)
        self.assertEqual(cleaned["removed_ephemeral"], 1)
        self.assertEqual(tight_cache.stats().ephemeral_count, 0)

        # 4. Stage.from_cli combines manual `version` with the flag/extra_args spec hash.
        s_a = Stage.from_cli(
            "blob_spec",
            binary=blob_bin,
            setting_flags={"tag": "--tag"},
            extra_args=["--extra", "A"],
            output_specs={"out": ".bin"},
            info_flag=None,
            version="same_v1",
        )
        s_b = Stage.from_cli(
            "blob_spec",
            binary=blob_bin,
            setting_flags={"tag": "--tag"},
            extra_args=["--extra", "B"],
            output_specs={"out": ".bin"},
            info_flag=None,
            version="same_v1",
        )
        self.assertNotEqual(s_a.version, s_b.version)
        spec_cache = Cache(self.tmp / "spec_cache")
        r_a = run_experiment(lambda run: run.run_stage(s_a, settings={"tag": "x"}), cache=spec_cache)
        r_b = run_experiment(lambda run: run.run_stage(s_b, settings={"tag": "x"}), cache=spec_cache)
        self.assertEqual(r_a.cache_misses, 1)
        self.assertEqual(r_b.cache_misses, 1)
        self.assertNotEqual(r_a.branches[0].result.content_id, r_b.branches[0].result.content_id)

    def test_hpc_lock_modes_slurm_sharding_and_env_metadata(self) -> None:
        """Verifies Lustre-safe lock modes (lockfile/none), SLURM job-array sharding,
        per-stage env overrides with content deduplication, and HPC execution metadata."""
        import csv
        from poisson_runner import LockMode, resolve_slurm_shard

        # 1. lock_mode="none" rejects max_bytes auto-eviction, works cleanly with max_bytes=None.
        with self.assertRaises(ValueError):
            Cache(self.tmp / "bad_none_cache", max_bytes=1024, lock_mode="none")

        none_cache = Cache(self.tmp / "none_cache", lock_mode="none")
        self.assertEqual(none_cache.lock_mode, LockMode.NONE)

        # 2. lock_mode="lockfile" handles atomic O_CREAT|O_EXCL locking, stale-lock recovery, and leases.
        lf_cache = Cache(self.tmp / "lf_cache", max_bytes=4096, lock_mode="lockfile", lock_stale_s=0.05)
        self.assertEqual(lf_cache.lock_mode, LockMode.LOCKFILE)
        # Plant a dead-PID lockfile and verify automatic stale recovery.
        lf_cache._excl_lock_path.write_text(
            json.dumps({"host": "other-node", "pid": 999999, "token": "stale"}),
            encoding="utf-8",
        )
        old_time = time.time() - 10.0
        os.utime(lf_cache._excl_lock_path, (old_time, old_time))

        # Stage that prints its OMP_NUM_THREADS to info, but writes deterministic bytes!
        env_log = self.tmp / "env_exec.log"
        env_bin = write_executable(
            self.tmp / "env_stage",
            f"""
import argparse, json, os, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
threads = os.environ.get("OMP_NUM_THREADS", "unset")
with open({str(env_log)!r}, "a") as f:
    f.write(f"{{a.tag}}:{{threads}}\\n")
# Deterministic output independent of thread count!
pathlib.Path(a.out).write_text(f"payload:{{a.tag}}")
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.01, "threads": threads}}))
""",
        )
        env_stage = Stage.from_cli(
            "compute",
            binary=env_bin,
            setting_flags={"tag": "--tag"},
            output_flags={"out": "--out"},
            output_specs={"out": ".txt"},
            info_flag="--info-out",
        )

        # Verify stale lockfile is recovered and lease pins objects in lockfile mode.
        with lf_cache.lease() as lease:
            r_lf = run_experiment(
                lambda run: run.run_stage(env_stage, settings={"tag": "a"}, env={"OMP_NUM_THREADS": "4"}),
                cache=lf_cache,
            )
            self.assertEqual(r_lf.cache_misses, 1)
            art_a: Artifact = r_lf.branches[0].result
            lease.pin([art_a.path.name])
            self.assertEqual(lf_cache.evict_lru(max_bytes=0), 0)
            self.assertTrue(art_a.path.is_file())
        self.assertEqual(lf_cache.evict_lru(max_bytes=0), 1)
        self.assertFalse(art_a.path.is_file())

        # 3. Thread scaling with env={"OMP_NUM_THREADS": ...}:
        #    Separate request keys (both run), deduplicated single object in cache/objects/,
        #    and HPC metadata recorded in execution + long CSV!
        old_slurm = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "424242"
        try:
            def scaling_pipe(run: RunContext) -> Artifact:
                t = compare("threads", [1, 4])
                res = run.run_stage(env_stage, settings={"tag": "same"}, env={"OMP_NUM_THREADS": str(t)})
                assert isinstance(res, Artifact)
                return res

            r_scale = run_experiment(scaling_pipe, cache=none_cache)
            self.assertEqual(r_scale.cache_misses, 2)
            # Identical output bytes -> only 1 deduplicated object on disk, 2 request receipts!
            self.assertEqual(none_cache.stats().object_count, 1)
            self.assertEqual(none_cache.stats().request_count, 2)

            exec_env = r_scale.branches[1].stages[0].execution["environment"]
            self.assertIsNotNone(exec_env["cpu_count"])
            self.assertEqual(exec_env["threading"]["OMP_NUM_THREADS"], "4")
            self.assertEqual(exec_env["slurm"]["SLURM_JOB_ID"], "424242")

            long_csv = self.tmp / "hpc_long.csv"
            r_scale.save_csv(long_csv, layout="long")
            with long_csv.open(encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(rows[0]["omp_num_threads"], "1")
            self.assertEqual(rows[1]["omp_num_threads"], "4")
            self.assertEqual(rows[1]["slurm_job_id"], "424242")
        finally:
            if old_slurm is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = old_slurm

        # 4. SLURM job-array sharding across 3 shards + read_only aggregation!
        self.assertEqual(
            resolve_slurm_shard({"SLURM_ARRAY_TASK_ID": "2", "SLURM_ARRAY_TASK_COUNT": "4"}),
            (2, 4),
        )
        self.assertEqual(
            resolve_slurm_shard(
                {
                    "SLURM_ARRAY_TASK_ID": "3",
                    "SLURM_ARRAY_TASK_MIN": "1",
                    "SLURM_ARRAY_TASK_MAX": "5",
                    "SLURM_ARRAY_TASK_STEP": "2",
                }
            ),
            (1, 3),
        )
        self.assertIsNone(resolve_slurm_shard({}))

        shard_cache = Cache(self.tmp / "shard_cache", lock_mode="none")

        def six_branch_pipe(run: RunContext) -> Artifact:
            a = compare("a", ["x", "y"])
            b = compare("b", [1, 2, 3])
            res = run.run_stage(env_stage, settings={"tag": f"{a}_{b}"})
            assert isinstance(res, Artifact)
            return res

        seen_indices: list[int] = []
        for shard_idx in range(3):
            r_sh = run_experiment(six_branch_pipe, cache=shard_cache, shard=(shard_idx, 3))
            self.assertEqual(r_sh.shard, (shard_idx, 3))
            self.assertEqual(len(r_sh.branches), 2)
            self.assertEqual(r_sh.cache_misses, 2)
            seen_indices.extend(b.index for b in r_sh.branches)
            self.assertIn(f"[shard {shard_idx}/3]", r_sh.format_table())

        self.assertEqual(sorted(seen_indices), [0, 1, 2, 3, 4, 5])

        # Final read_only aggregation pass hits 100% cache across all 6 branches!
        r_agg = run_experiment(six_branch_pipe, cache=shard_cache, cache_mode="read_only")
        self.assertEqual(len(r_agg.branches), 6)
        self.assertEqual(r_agg.cache_hits, 6)
        self.assertEqual(r_agg.cache_misses, 0)

        # 5. Review regressions:
        #    a) Cache persists `.lock_mode` so a default Cache(root) / CLI adopts `lockfile` automatically.
        reopened = Cache(lf_cache.root)
        self.assertEqual(reopened.lock_mode, LockMode.LOCKFILE)

        #    b) Sparse SLURM arrays (e.g. --array=1,3,8) are rejected clearly instead of silently mis-sharding.
        with self.assertRaises(ValueError):
            resolve_slurm_shard(
                {
                    "SLURM_ARRAY_TASK_ID": "3",
                    "SLURM_ARRAY_TASK_MIN": "1",
                    "SLURM_ARRAY_TASK_MAX": "8",
                    "SLURM_ARRAY_TASK_STEP": "1",
                    "SLURM_ARRAY_TASK_COUNT": "3",
                }
            )

        #    c) Sharded run fails loudly if replay discovers unplanned compare() choices (e.g. gated on Artifact.info).
        def info_gated_pipe(run: RunContext) -> Artifact:
            top = compare("top", ["a", "b"])
            res = run.run_stage(env_stage, settings={"tag": f"gated_{top}"})
            assert isinstance(res, Artifact)
            if res.info.get("threads"):  # Empty in dry-run discovery, populated on real run!
                _ = compare("late_choice", [1, 2])
            return res

        with self.assertRaises(RuntimeError):
            run_experiment(info_gated_pipe, cache=Cache(self.tmp / "gated_cache", lock_mode="none"), shard=(0, 2))

        #    d) Choice named `omp_num_threads` does not collide with or overwrite the long CSV metadata column.
        def colliding_name_pipe(run: RunContext) -> Artifact:
            t = compare("omp_num_threads", ["fast", "slow"])
            res = run.run_stage(env_stage, settings={"tag": f"col_{t}"}, env={"OMP_NUM_THREADS": "2"})
            assert isinstance(res, Artifact)
            return res

        r_col = run_experiment(colliding_name_pipe, cache=none_cache)
        col_csv = self.tmp / "colliding_long.csv"
        r_col.save_csv(col_csv, layout="long")
        with col_csv.open(encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            self.assertIn("choice_omp_num_threads", reader.fieldnames or [])
            self.assertIn("omp_num_threads", reader.fieldnames or [])
            col_rows = list(reader)
        self.assertEqual(col_rows[0]["choice_omp_num_threads"], "fast")
        self.assertEqual(col_rows[0]["omp_num_threads"], "2")

    def _counting_stage(self, name: str, *, random_output: bool = False, info_time: float | None = 0.25) -> tuple[Stage, Path]:
        """A stage that appends one line per execution to a log and optionally writes random bytes."""
        log = self.tmp / f"{name}.log"
        info_line = (
            f"pathlib.Path(a.info_out).write_text(json.dumps({{'time': {info_time!r}}}))"
            if info_time is not None
            else "pathlib.Path(a.info_out).write_text('{}')"
        )
        payload = "os.urandom(16).hex()" if random_output else "a.tag"
        stage_bin = write_executable(
            self.tmp / f"{name}_bin",
            f"""
import argparse, json, os, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log)!r}, "a") as f:
    f.write(a.tag + "\\n")
pathlib.Path(a.out).write_text(str({payload}))
{info_line}
""",
        )
        stage = Stage.from_cli(
            name,
            binary=stage_bin,
            input_flags={"inp": "--in"},
            setting_flags={"tag": "--tag"},
            output_flags={"out": "--out"},
            output_specs={"out": ".txt"},
            info_flag="--info-out",
        )
        return stage, log

    @staticmethod
    def _count_lines(log: Path) -> int:
        return len(log.read_text().splitlines()) if log.exists() else 0

    def test_shared_prefix_runs_once_per_experiment_under_refresh_and_off(self) -> None:
        prefix, prefix_log = self._counting_stage("prefix")
        leaf, leaf_log = self._counting_stage("leaf")
        cache = Cache(self.tmp / "reuse_cache")

        def pipe(run: RunContext) -> Artifact:
            shared = run.run_stage(prefix, settings={"tag": "shared"})
            theta = compare("theta", [0.3, 0.5, 0.7, 0.9])
            out = run.run_stage(leaf, inputs={"inp": shared}, settings={"tag": f"leaf_{theta}"})
            assert isinstance(out, Artifact)
            return out

        # read_only on an empty cache: the miss runs once and is not written, later branches reuse it.
        ro_cache = Cache(self.tmp / "reuse_ro_cache")
        r_ro = run_experiment(pipe, cache=ro_cache, stage_cache_modes={"prefix": "read_only"})
        self.assertEqual(self._count_lines(prefix_log), 1)
        self.assertEqual([b.stages[0].status for b in r_ro.branches], ["MISS"] + ["REUSED"] * 3)
        self.assertIsNone(r_ro.branches[1].stages[0].request_key)  # never published to the cache

        for mode in ("refresh", "off"):
            before = self._count_lines(prefix_log)
            report = run_experiment(pipe, cache=cache, stage_cache_modes={"prefix": mode})
            self.assertEqual(self._count_lines(prefix_log) - before, 1, f"prefix must run exactly once under {mode}")
            self.assertEqual([b.stages[0].status for b in report.branches], ["MISS"] + ["REUSED"] * 3)
            self.assertEqual(report.reused, 3)
            self.assertIn("3 reused in this run", report.format_table())
            # Every branch sees the same upstream content.
            self.assertEqual(len({b.stages[0].content_ids["out"] for b in report.branches}), 1)

        # Leaves were cached on the first `use`-mode experiment and are plain HITs afterwards.
        self.assertEqual(self._count_lines(leaf_log), 8)  # 4 in ro_cache + 4 in cache

        # A new run_experiment call starts fresh: refresh re-runs the prefix exactly once again.
        before = self._count_lines(prefix_log)
        run_experiment(pipe, cache=cache, stage_cache_modes={"prefix": "refresh"})
        self.assertEqual(self._count_lines(prefix_log) - before, 1)

    def test_repeats_determinism_check(self) -> None:
        det, _ = self._counting_stage("det")
        rnd, _ = self._counting_stage("rnd", random_output=True)
        cache = Cache(self.tmp / "det_cache")

        r_det = run_experiment(lambda run: run.run_stage(det, settings={"tag": "a"}), cache=cache, repeats=3)
        st = r_det.branches[0].stages[0]
        self.assertEqual(st.warnings, [])
        self.assertEqual(st.execution["nondeterministic_outputs"], [])

        r_rnd = run_experiment(lambda run: run.run_stage(rnd, settings={"tag": "a"}), cache=cache, repeats=2, warmup=1)
        st = r_rnd.branches[0].stages[0]
        self.assertEqual(st.execution["nondeterministic_outputs"], ["out"])
        self.assertEqual(len(st.warnings), 1)
        self.assertIn("different output bytes", st.warnings[0])
        self.assertIn("Warning:", r_rnd.format_table())
        self.assertIn("!", r_rnd.format_table())

        # The warning is persisted with the cache entry and survives a later HIT.
        r_hit = run_experiment(lambda run: run.run_stage(rnd, settings={"tag": "a"}), cache=cache)
        self.assertTrue(r_hit.branches[0].stages[0].cached)
        self.assertEqual(len(r_hit.branches[0].stages[0].warnings), 1)

        # A single run cannot be checked, so no determinism field is recorded.
        r_one = run_experiment(lambda run: run.run_stage(rnd, settings={"tag": "single"}), cache=cache)
        self.assertNotIn("nondeterministic_outputs", r_one.branches[0].stages[0].execution)

    def test_per_stage_repeats_precedence_and_hits_ignore_repeats(self) -> None:
        a, a_log = self._counting_stage("stage_a")
        b, b_log = self._counting_stage("stage_b")
        cache = Cache(self.tmp / "rep_cache")

        def bench(trace: StageTrace) -> tuple[int, int]:
            info = trace.execution["benchmark"]
            return info["repeats"], info["warmup"]

        def pipe(run: RunContext) -> None:
            run.run_stage(a, settings={"tag": "x"})
            run.run_stage(b, settings={"tag": "x"})
            run.run_stage(b, settings={"tag": "y"}, repeats=2, warmup=0)  # per-call beats per-stage

        r = run_experiment(pipe, cache=cache, repeats=3, warmup=1, stage_repeats={"stage_b": 5}, stage_warmup={"stage_b": 2})
        s_a, s_bx, s_by = r.branches[0].stages
        self.assertEqual(bench(s_a), (3, 1))   # experiment-wide default
        self.assertEqual(bench(s_bx), (5, 2))  # per-stage beats experiment-wide
        self.assertEqual(bench(s_by), (2, 0))  # per-call beats per-stage
        self.assertEqual(self._count_lines(a_log), 4)
        self.assertEqual(self._count_lines(b_log), 7 + 2)

        # Asking for more repeats later is still a cache HIT: nothing re-runs.
        r_more = run_experiment(pipe, cache=cache, repeats=10, stage_repeats={"stage_b": 20})
        self.assertEqual(r_more.cache_hits, 3)
        self.assertEqual(self._count_lines(a_log), 4)
        self.assertEqual(bench(r_more.branches[0].stages[0]), (3, 1))  # trace reports what was measured

        # Re-measuring requires an explicit refresh for that stage only.
        r_ref = run_experiment(pipe, cache=cache, repeats=10, stage_cache_modes={"stage_a": "refresh"})
        self.assertEqual(self._count_lines(a_log), 4 + 10)
        self.assertEqual(bench(r_ref.branches[0].stages[0]), (10, 0))
        self.assertTrue(r_ref.branches[0].stages[1].cached)

        with self.assertRaises(ValueError):
            run_experiment(pipe, cache=cache, stage_repeats={"stage_a": 0})
        with self.assertRaises(ValueError):
            run_experiment(pipe, cache=cache, stage_warmup={"stage_a": -1})

    def test_info_time_is_headline_timing(self) -> None:
        timed, _ = self._counting_stage("timed", info_time=0.25)
        untimed, _ = self._counting_stage("untimed", info_time=None)
        cache = Cache(self.tmp / "time_cache")

        def pipe(run: RunContext) -> None:
            run.run_stage(timed, settings={"tag": "t"})
            run.run_stage(untimed, settings={"tag": "t"})

        r = run_experiment(pipe, cache=cache, repeats=2)
        s_timed, s_untimed = r.branches[0].stages
        self.assertAlmostEqual(s_timed.time_s or 0.0, 0.25)
        self.assertIsNone(s_untimed.time_s)
        table = r.format_table()
        self.assertIn("timed(MISS 0.250s)", table)
        self.assertRegex(table, r"untimed\(MISS \d+\.\d{3}s wall\)")

        # Cached hits keep the headline time.
        r_hit = run_experiment(pipe, cache=cache)
        self.assertAlmostEqual(r_hit.branches[0].stages[0].time_s or 0.0, 0.25)

        import csv

        long_csv = self.tmp / "time_long.csv"
        r.save_csv(long_csv, layout="long")
        with long_csv.open(encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            cols = reader.fieldnames or []
            rows = list(reader)
        self.assertLess(cols.index("info_time_s"), cols.index("wall_s"))
        self.assertEqual(rows[0]["info_time_s"], "0.250000")
        wide_csv = self.tmp / "time_wide.csv"
        r.save_csv(wide_csv, layout="wide")
        with wide_csv.open(encoding="utf-8") as fp:
            wcols = csv.DictReader(fp).fieldnames or []
        self.assertLess(wcols.index("timed_info_time_s"), wcols.index("timed_wall_s"))

    def test_integral_float_settings_share_cache_key(self) -> None:
        key_int = Cache.compute_request_key("s", "c", {}, {"cycles": 1, "nested": [2, {"x": 3}]})
        key_float = Cache.compute_request_key("s", "c", {}, {"cycles": 1.0, "nested": [2.0, {"x": 3.0}]})
        self.assertEqual(key_int, key_float)
        self.assertNotEqual(key_int, Cache.compute_request_key("s", "c", {}, {"cycles": 1.5, "nested": [2, {"x": 3}]}))
        self.assertNotEqual(key_int, Cache.compute_request_key("s", "c", {}, {"cycles": True, "nested": [2, {"x": 3}]}))

        # Only the key is normalized: the binary still receives the float as written.
        cfg_seen = self.tmp / "cfg_seen.json"
        cfg_bin = write_executable(
            self.tmp / "cfg_echo",
            f"""
import argparse, pathlib, shutil
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
shutil.copy(a.config, {str(cfg_seen)!r})
pathlib.Path(a.out).write_text("ok")
""",
        )
        stage = Stage.from_cli(
            "cfg_echo", binary=cfg_bin, config_flag="--config", output_specs={"out": ".txt"}, info_flag=None
        )
        cache = Cache(self.tmp / "float_cache")
        r1 = run_experiment(lambda run: run.run_stage(stage, settings={"theta": 1.0}), cache=cache)
        self.assertIn("1.0", cfg_seen.read_text())
        r2 = run_experiment(lambda run: run.run_stage(stage, settings={"theta": 1}), cache=cache)
        self.assertEqual(r1.cache_misses, 1)
        self.assertEqual(r2.cache_hits, 1)

    def test_shared_library_tracking(self) -> None:
        from poisson_runner import make_import_mesh_stage, shared_library_deps

        stage = make_import_mesh_stage()
        libs = dict(stage.tracked_shared_libs())
        if not libs:
            self.skipTest("ldd unavailable or poisson_import is statically linked")
        self.assertTrue(any(name.startswith("libhdf5") for name in libs), libs)
        # OS base libraries (libc, libm, ld-linux, ...) are skipped by default (issue #3).
        self.assertFalse(any(name.startswith(("libc.so", "libm.so", "ld-linux")) for name in libs), libs)
        all_libs = dict(shared_library_deps(stage.executable, include_os_libs=True))
        self.assertTrue(any(name.startswith("libc.so") for name in all_libs), all_libs)
        # Only direct NEEDED dependencies are tracked, not the whole transitive closure.
        self.assertFalse(any(name.startswith("libcurl") for name in libs), libs)
        for path in libs.values():
            self.assertTrue(path.is_file())

        cache = Cache(self.tmp / "libs_cache")
        with_libs = stage.compute_code_hash(cache)
        stage.track_shared_libs = False
        self.assertNotEqual(with_libs, stage.compute_code_hash(cache))

        script, _ = self._counting_stage("script_stage")
        self.assertEqual(script.tracked_shared_libs(), [])
        self.assertEqual(shared_library_deps(self.mesh_file), [])


if __name__ == "__main__":
    unittest.main()
