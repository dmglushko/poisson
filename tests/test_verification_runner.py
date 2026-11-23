"""End-to-end tests of the verification stages: case -> gmsh -> import, caching, saved runs, and the
safety checks of the gmsh stage and the poisson_case CLI."""

from __future__ import annotations

import json
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

    def test_case_meshes_and_caching(self) -> None:
        def experiment(run):
            case = compare("case", {"planar": PLANAR, "screened": SCREENED, "ball": BALL})
            h = compare("h", [0.4, 0.2])
            return {"mesh": run.case_mesh(case, h_max=h), "case": case}

        report = run_experiment(experiment, cache=self.cache)
        self.assertEqual(len(report.branches), 6)
        self.assertTrue(all(b.ok for b in report.branches), [b.error for b in report.branches])
        by = {(b.choice_labels["case"], b.choice_labels["h"]): b for b in report.branches}
        for b in report.branches:
            self.assertEqual(b["mesh"].info["warnings"], [])

        with h5py.File(by[("planar", "0.2")]["mesh"].path, "r") as f:
            names = f["materials/name_bytes"][:].tobytes().decode()
            self.assertTrue(all(n in names for n in ("oxide", "plate", "si")))
            self.assertEqual(int(f.attrs["screened"]), 1)
            self.assertEqual(f["charges/value"].shape, (0,))
        with h5py.File(by[("ball", "0.2")]["mesh"].path, "r") as f:
            self.assertEqual(list(f["charges/value"][:]), [-1])
        # Finer h, more tets.
        with h5py.File(by[("screened", "0.4")]["mesh"].path, "r") as a, h5py.File(by[("screened", "0.2")]["mesh"].path, "r") as b:
            self.assertGreater(b["elements/node"].shape[0], a["elements/node"].shape[0])

        # Re-running is all cache hits.
        self.assertEqual(run_experiment(experiment, cache=self.cache).cache_misses, 0)

    def test_saved_run_keeps_the_case(self) -> None:
        def experiment(run):
            return {"mesh": run.case_mesh(BALL, h_max=0.8), "case": BALL}

        run_experiment(experiment, cache=self.cache).save(self.tmp / "saved")
        saved = load_run(self.tmp / "saved")
        self.assertEqual(saved.branches[0]["case"], BALL)   # the case comes back as a Case

        unscreened = Case.of("planar_layers", layers=[{"name": "oxide", "thickness": 1, "eps_r": 3.9}], screened=False)
        mesh = run_experiment(lambda run: run.case_mesh(unscreened, h_max=0.5), cache=self.cache).branches[0].result
        with h5py.File(mesh.path, "r") as f:
            self.assertEqual(int(f.attrs["screened"]), 0)   # imported as the case says

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
            return {"mesh": run.case_mesh(PLANAR, h_max=0.5)}

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
        for name in ("Case", "dirichlet", "case_mesh", "gmsh_mesh", "case_inputs", "make_gmsh_stage"):
            self.assertIn(name, poisson_runner.__all__)


if __name__ == "__main__":
    unittest.main()
