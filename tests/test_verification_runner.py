"""End-to-end tests of the verification stages: case -> gmsh -> import -> exact field -> verify / sample,
inside an experiment and as a separate evaluation of a saved run."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import (  # noqa: E402
    Cache,
    Case,
    compare,
    dirichlet,
    evaluate,
    gmsh_mesh,
    load_run,
    resolve_binary,
    robin,
    run_experiment,
)

# A dielectric, a floating metal plate, and a screened dielectric between u = 0 and a Robin face.
PLANAR = Case.of(
    "planar_layers",
    layers=[
        {"name": "oxide", "thickness": 1, "eps_r": 3.9},
        {"name": "plate", "thickness": 0.5, "metal": True},
        {"name": "si", "thickness": 1.5, "eps_r": 11.7},
    ],
    left=dirichlet(0),
    right=robin(0.5, 0.1),
)
SCREENED = Case.of(
    "planar_layers",
    layers=[{"name": "doped", "thickness": 3, "eps_r": 11.7, "screening_length": 0.5}],
    left=dirichlet(0.3),
    right=dirichlet(1.7),
)
BALL = Case.of(
    "spherical_layers",
    layers=[{"name": "core", "thickness": 2, "eps_r": 3.9, "screening_length": 2}],
    outer={"type": "dirichlet", "value": 0},
    charge=-1,
)


@unittest.skipIf(shutil.which("gmsh") is None, "gmsh is not installed")
class VerificationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_verification_test_")
        self.tmp = Path(self._tmp.name)
        self.cache = Cache(self.tmp / "cache")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exact_cases_and_convergence(self) -> None:
        def experiment(run):
            case = compare("case", {"planar": PLANAR, "screened": SCREENED})
            h = compare("h", [0.2, 0.1])
            mesh = run.case_mesh(case, h_max=h)
            exact = run.exact_field(case, mesh)
            errors = run.verify(case, mesh, exact)
            line = run.sample_field(mesh, exact, line=([0.05, 0.5, 0.5], [2.95, 0.5, 0.5], 20), case=case)
            return {"mesh": mesh, "exact": exact, "errors": errors, "line": line, "case": case}

        report = run_experiment(experiment, cache=self.cache)
        self.assertEqual(len(report.branches), 4)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        by = {(b.choice_labels["case"], b.choice_labels["h"]): b for b in report.branches}

        # Piecewise linear (with a floating metal): the interpolant is exact on any mesh.
        for h in ("0.2", "0.1"):
            info = by[("planar", h)]["errors"].info
            self.assertLess(info["l2_error"], 1e-12)
            self.assertLess(info["metal_potential_error"], 1e-13)
            self.assertGreater(info["metal_elements"], 0)
            self.assertEqual(by[("planar", h)]["mesh"].info["warnings"], [])
        # Screened: second order in L2 between h = 0.2 and 0.1.
        coarse = by[("screened", "0.2")]["errors"].info["l2_error"]
        fine = by[("screened", "0.1")]["errors"].info["l2_error"]
        self.assertGreater(math.log2(coarse / fine), 1.6)

        # The exact field matches the exact solution at sample points of a linear case.
        with h5py.File(by[("planar", "0.1")]["line"].path, "r") as f:
            self.assertEqual(int(f["found"][:].sum()), 20)
            self.assertLess(float(abs(f["u"][:] - f["u_exact"][:]).max()), 1e-12)
            self.assertEqual(f["points"].shape, (20, 3))

        # The norms are CSV columns.
        csv_path = self.tmp / "report.csv"
        report.save_csv(csv_path, layout="wide")
        with csv_path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self.assertIn("verify.l2_error", header)
        self.assertIn("verify.h1_error", header)

        # Re-running is all cache hits.
        again = run_experiment(experiment, cache=self.cache)
        self.assertEqual(again.cache_misses, 0)

    def test_saved_run_is_verified_afterwards(self) -> None:
        # The experiment only builds meshes and a "solution"; verification happens later.
        def experiment(run):
            h = compare("h", [0.8, 0.4])
            mesh = run.case_mesh(BALL, h_max=h)
            return {"mesh": mesh, "solution": run.exact_field(BALL, mesh), "case": BALL}

        report = run_experiment(experiment, cache=self.cache)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        mesh_info = report.branches[0]["mesh"].info
        self.assertEqual(mesh_info["warnings"], [])
        report.save(self.tmp / "saved")

        saved = load_run(self.tmp / "saved")
        self.assertEqual(saved.branches[0]["case"], BALL)   # the case comes back as a Case

        def analysis(run, b):
            far = run.verify(b["case"], b["mesh"], b["solution"], exclusion_radius=1.5)
            return {"l2": far.info["l2_error"], "excluded": far.info["excluded_points"]}

        ana = evaluate(saved, analysis, cache=self.cache)
        self.assertTrue(all(b.ok for b in ana.branches), [b.error for b in ana.branches])
        l2 = {b.choice_labels["h"]: b.scalars["l2"] for b in ana.branches}
        self.assertGreater(math.log2(l2["0.8"] / l2["0.4"]), 1.6)
        self.assertTrue(all(b.scalars["excluded"] > 0 for b in ana.branches))

    def test_mismatched_mesh_is_rejected(self) -> None:
        other = Case.of("planar_layers", layers=[{"name": "oxide", "thickness": 1, "eps_r": 3.9}])

        def experiment(run):
            mesh = run.case_mesh(PLANAR, h_max=0.5)
            run.verify(other, mesh, run.exact_field(PLANAR, mesh))

        report = run_experiment(experiment, cache=self.cache, continue_on_error=True)
        self.assertFalse(report.branches[0].ok)
        self.assertIn("does not match case", str(report.branches[0].error))

    def test_refinement_near_charge_and_options(self) -> None:
        image = Case.of("grounded_ball_image", radius=5, eps_r=3.9, charge=-1, position=[1, 0.5, -0.5])

        def experiment(run):
            refined = compare("refined", [False, True])
            options = {"h_charge": 0.2, "r_fine": 0.5, "r_coarse": 2} if refined else {}
            mesh = run.case_mesh(image, h_max=1.0, algorithm="del3d", optimize=True, **options)
            with h5py.File(mesh.path, "r") as f:
                return {"nodes": int(f["nodes"].shape[0])}

        report = run_experiment(experiment, cache=self.cache)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        nodes = {b.choice_labels["refined"]: b.scalars["nodes"] for b in report.branches}
        self.assertGreater(nodes["True"], nodes["False"])

        def bad(run):
            run.case_mesh(image, h_max=1.0, h_charge=0.2, r_fine=2, r_coarse=1)

        with self.assertRaises(ValueError):
            run_experiment(bad, cache=self.cache)

    def test_dry_run_of_a_cached_charge_free_case_runs_nothing(self) -> None:
        def experiment(run):
            mesh = run.case_mesh(PLANAR, h_max=0.5)
            return {"errors": run.verify(PLANAR, mesh, run.exact_field(PLANAR, mesh))}

        run_experiment(experiment, cache=self.cache)
        dry = run_experiment(experiment, cache=self.cache, dry_run=True)
        self.assertEqual(dry.cache_misses, 0, [(s.stage, s.status) for s in dry.branches[0].stages])

    def test_geo_must_be_self_contained(self) -> None:
        box = self.tmp / "box.geo"
        box.write_text('SetFactory("OpenCASCADE");\nBox(1) = {0, 0, 0, 1, 1, 1};\n'
                       'Physical Volume("oxide") = {1};\nPhysical Surface("0V") = {1:6};\n', encoding="utf-8")
        main = self.tmp / "main.geo"
        main.write_text('Include "box.geo";\n', encoding="utf-8")
        commented = self.tmp / "commented.geo"
        commented.write_text(box.read_text(encoding="utf-8") + '// Include "other.geo";\n/* Merge "x.msh"; */\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Include"):
            run_experiment(lambda run: gmsh_mesh(run, main, h_max=0.5), cache=self.cache)
        # A "//" inside a string is not a comment.
        url = self.tmp / "url.geo"
        url.write_text('Printf("http://example.invalid"); Include "box.geo";\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Include"):
            run_experiment(lambda run: gmsh_mesh(run, url, h_max=0.5), cache=self.cache)
        report = run_experiment(lambda run: gmsh_mesh(run, commented, h_max=0.5), cache=self.cache)
        self.assertTrue(report.branches[0].ok)

    def test_cli_does_not_publish_part_of_its_outputs(self) -> None:
        config = self.tmp / "case.json"
        config.write_text(json.dumps({"case": PLANAR.spec()}), encoding="utf-8")
        geo = self.tmp / "case.geo"
        geo.write_text("old", encoding="utf-8")
        info_dir = self.tmp / "info"
        info_dir.mkdir()
        proc = subprocess.run(
            [str(resolve_binary("poisson_case")), "--config", str(config), "--geo-out", str(geo),
             "--materials-out", str(self.tmp / "case.yaml"), "--charges-out", str(self.tmp / "case.cha"),
             "--info-out", str(info_dir)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("is a directory", proc.stderr)
        self.assertEqual(geo.read_text(encoding="utf-8"), "old")
        self.assertFalse((self.tmp / "case.yaml").exists())

    def test_case_record(self) -> None:
        self.assertEqual(Case.of("x", a=(1, 2)).spec()["params"], {"a": [1, 2]})
        # Parameters are read-only, so a case cannot change its hash inside a set.
        layered = Case.of("x", layers=[{"thickness": 1}])
        members = {layered}
        with self.assertRaises(TypeError):
            layered.params["layers"][0]["thickness"] = 2   # type: ignore[index]
        self.assertIn(layered, members)
        self.assertEqual(Case.from_dict(layered.to_dict()), layered)
        self.assertEqual(hash(Case.of("x", a=1, b=2)), hash(Case("x", {"b": 2, "a": 1})))
        self.assertFalse(Case.of("x", screened=False).screened)
        with self.assertRaises(TypeError):
            Case.of("x", a=object())
        with self.assertRaises(ValueError):
            Case("", {})
        # Equal cases hash equally; 1 and 1.0 are the same number, true is not 1.
        self.assertEqual(Case.of("x", t=1), Case.of("x", t=1.0))
        self.assertEqual(hash(Case.of("x", t=1)), hash(Case.of("x", t=1.0)))
        self.assertEqual(hash(Case.of("x", t=-0.0)), hash(Case.of("x", t=0)))
        self.assertNotEqual(Case.of("x", t=True), Case.of("x", t=1))
        self.assertEqual(len({Case.of("x", t=1), Case.of("x", t=1.0)}), 1)
        self.assertTrue(Case.of("x", screened=None).screened)   # null means the default, as in C++
        import poisson_runner
        for name in ("Case", "dirichlet", "case_mesh", "gmsh_mesh", "verify", "make_verify_stage"):
            self.assertIn(name, poisson_runner.__all__)


if __name__ == "__main__":
    unittest.main()
