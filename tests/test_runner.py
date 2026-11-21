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

        # 4. Truncated cached object is rejected on lookup and recomputed cleanly.
        art_one: Artifact = r_one.branches[0].result
        art_one.path.write_bytes(b"A")   # corrupt/truncate from 3 bytes to 1 byte
        r_rebuilt = run_experiment(lambda run: run.run_stage(s_one), cache=cache)
        self.assertEqual(r_rebuilt.cache_misses, 1)
        self.assertEqual(r_rebuilt.branches[0].result.path.read_bytes(), b"AAA")

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

        # #7: dry_run on cold cache executes NOTHING and reports WOULD_RUN for all 6 stage calls.
        dry_cold = run_experiment(loop_pipeline, cache=cache, dry_run=True)
        self.assertTrue(dry_cold.dry_run)
        self.assertEqual(len(dry_cold.branches), 2)
        self.assertEqual(dry_cold.cache_hits, 0)
        self.assertEqual(dry_cold.cache_misses, 6)
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

        # 1. Run 2x2 Compare across estimators & refiners on real poisson_import + poisson_adapt!
        def amr_compare(run: RunContext) -> Artifact:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            with h5py.File(mesh.path, "r") as h5:
                nodes = h5["nodes"][:]
            sol_path = write_nodal_field_h5(self.tmp / "u_init.h5", nodes)
            est = compare("estimator", ["kelly", "recovery"])
            ref = compare("refiner", ["red_green", "longest_edge_bisection"])
            adapted = run.adapt_mesh(
                mesh,
                sol_path,
                config={
                    "estimator": {"type": est},
                    "marker": {"type": "dorfler", "theta": 0.5},
                    "refiner": {"type": ref},
                },
            )
            assert isinstance(adapted, Artifact)
            return adapted

        report = run_experiment(amr_compare, cache=cache)
        self.assertEqual(len(report.branches), 4)
        self.assertEqual(report.cache_misses, 5)   # 1 import + 4 adapt combinations
        self.assertEqual(report.cache_hits, 3)     # import cached on branches 1..3
        for br in report.branches:
            adapted_art: Artifact = br.result
            self.assertEqual(adapted_art.info["status"], "adapted")
            self.assertGreater(adapted_art.info["elements_after"], adapted_art.info["elements_before"])
            self.assertEqual(len(adapted_art.info["stages"]), 3)

        # 2. 2-cycle RedGreenRefinement passing RefinementHistory + Parents across CLI invocations.
        def two_cycle_rg(run: RunContext) -> dict[str, Artifact]:
            mesh = run.import_mesh(self.mesh_file, materials=[self.mat_file])
            hist: Artifact | None = None
            out: dict[str, Artifact] = {}
            for cycle in range(2):
                with h5py.File(mesh.path, "r") as h5:
                    nodes = h5["nodes"][:]
                sol_path = write_nodal_field_h5(self.tmp / f"u_c{cycle}.h5", nodes)
                res = run.adapt_mesh(
                    mesh,
                    sol_path,
                    history=hist,
                    with_parents=True,
                    with_history=True,
                    config={
                        "estimator": {"type": "kelly"},
                        "marker": {"type": "dorfler", "theta": 0.5},
                        "refiner": {"type": "red_green"},
                    },
                )
                assert isinstance(res, dict)
                mesh = res["out"]
                hist = res["history"]
                out = res
            return out

        r_cyc = run_experiment(two_cycle_rg, cache=cache)
        self.assertEqual(len(r_cyc.branches), 1)
        final_dict = r_cyc.branches[0].result
        self.assertEqual(final_dict["out"].info["status"], "adapted")
        self.assertGreater(final_dict["out"].info["elements_after"], 2)
        self.assertTrue(final_dict["parents"].path.is_file())
        self.assertTrue(final_dict["history"].path.is_file())


if __name__ == "__main__":
    unittest.main()
