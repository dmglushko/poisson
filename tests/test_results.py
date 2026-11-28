"""Tests for named branch outputs, info-scalar CSV columns, saved runs, and post-hoc evaluation."""

from __future__ import annotations

import csv
import dataclasses
import fcntl
import io
import json
import math
import shutil
import stat
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import (  # noqa: E402
    SOURCE_BRANCH,
    Artifact,
    Cache,
    RecordError,
    RunContext,
    Stage,
    UnresolvedRecord,
    Unserializable,
    compare,
    evaluate,
    info_scalars,
    load_run,
    register_record,
    run_experiment,
)


@register_record("test_ball_case")
@dataclass(frozen=True)
class BallCase:
    radius: float
    eps: float
    charge: int = -1


@register_record("test_mesh_case")
@dataclass(frozen=True)
class MeshCase:
    geo: Path
    label: str
    shape: tuple = (1, 2)


class _Opaque:
    pass


AMR_LIKE_INFO = {
    "type": "AdaptationResult::Info",
    "status": "adapted",
    "adapter": "error_based",
    "time": 1.84,
    "elements_before": 120,
    "elements_after": 310,
    "optional_metric": None,
    "warnings": ["w"],
    "details": {"passes": 3, "nested": {"skip": 1}},
    "history": [1, 2, 3],
    "stages": [
        {"stage": "estimate", "method": "residual", "time": 0.61, "metrics": {"total": 0.0042, "jump_part": 0.0027}, "warnings": []},
        {"stage": "refine", "method": "leb", "time": 1.1, "metrics": {"marked": 9}},
        {"stage": "refine", "method": "leb", "time": 0.2, "metrics": {"marked": 4}},
    ],
}


def write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class ResultsTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_results_test_")
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "exec.log"
        gen_bin = write_executable(
            self.tmp / "gen_bin",
            f"""
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(self.log)!r}, "a") as f: f.write("gen:" + a.tag + "\\n")
pathlib.Path(a.out).write_text("x" * len(a.tag))
info = {{"type": "Gen::Info", "time": 0.5, "length": len(a.tag), "status": "ok", "warnings": [],
        "stages": [{{"stage": "inner", "time": 0.1, "metrics": {{"cells": len(a.tag) * 10}}}}]}}
pathlib.Path(a.info_out).write_text(json.dumps(info))
""",
        )
        diff_bin = write_executable(
            self.tmp / "diff_bin",
            f"""
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(self.log)!r}, "a") as f: f.write("diff\\n")
la = len(pathlib.Path(a.a).read_text()); lb = len(pathlib.Path(a.b).read_text())
pathlib.Path(a.out).write_text(json.dumps({{"abs_diff": abs(la - lb)}}))
pathlib.Path(a.info_out).write_text(json.dumps({{"time": 0.01, "abs_diff": abs(la - lb)}}))
""",
        )
        self.gen = Stage.from_cli(
            "gen", binary=gen_bin, setting_flags={"tag": "--tag"}, output_specs={"out": ".txt"}, track_shared_libs=False
        )
        self.diff = Stage.from_cli(
            "diff",
            binary=diff_bin,
            input_flags={"a": "--a", "b": "--b"},
            output_specs={"out": ".json"},
            track_shared_libs=False,
        )
        self.cache = Cache(self.tmp / "cache")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def executions(self, prefix: str) -> int:
        if not self.log.exists():
            return 0
        return sum(1 for line in self.log.read_text().splitlines() if line.startswith(prefix))

    def sweep(self, run: RunContext) -> dict[str, object]:
        solver = compare("solver", ["direct", "cg"])
        h = compare("h", [0.5, 0.25])
        out = run.run_stage(self.gen, settings={"tag": f"{solver}-{h}"})
        assert isinstance(out, Artifact)
        return {
            "solution": out,
            "dofs": int(1 / h) * 100,
            "case": BallCase(radius=10.0, eps=3.9),
            "note": Path("/tmp/x"),
            "tags": [solver, {"$weird": 1}],
        }

    # ------------------------------------------------------------------ info flattening

    def test_info_scalars_flattening(self) -> None:
        flat = info_scalars(AMR_LIKE_INFO)
        self.assertEqual(flat["status"], "adapted")
        self.assertEqual(flat["elements_after"], 310)
        for skipped in ("type", "time", "warnings", "optional_metric", "history", "details.nested"):
            self.assertNotIn(skipped, flat)
        self.assertEqual(flat["details.passes"], 3)
        self.assertEqual(flat["estimate.method"], "residual")
        self.assertEqual(flat["estimate.time"], 0.61)
        self.assertEqual(flat["estimate.jump_part"], 0.0027)
        self.assertNotIn("estimate.warnings", flat)
        self.assertEqual(flat["refine#0.marked"], 9)
        self.assertEqual(flat["refine#1.marked"], 4)
        self.assertEqual(info_scalars(None), {})

    # ------------------------------------------------------------------ CSV columns

    def test_csv_has_metric_and_result_columns_and_reserved_names_are_safe(self) -> None:
        def pipe(run: RunContext) -> dict[str, object]:
            compare("branch", ["p", "q"])  # collides with the fixed `branch` column
            run.run_stage(self.gen, settings={"tag": "ab"})
            run.run_stage(self.gen, settings={"tag": "abcd"})
            return {"value": 0.1 + 0.2, "flag": True}

        r = run_experiment(pipe, cache=self.cache)
        wide = self.tmp / "wide.csv"
        r.save_csv(wide, layout="wide")
        with wide.open(encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            cols = reader.fieldnames or []
            rows = list(reader)
        self.assertEqual(cols.count("branch"), 1)
        self.assertIn("choice_branch", cols)
        self.assertEqual([row["branch"] for row in rows], ["0", "1"])
        self.assertEqual([row["choice_branch"] for row in rows], ["p", "q"])
        self.assertEqual(rows[0]["res.value"], repr(0.1 + 0.2))  # full precision
        self.assertEqual(float(rows[0]["res.value"]), 0.1 + 0.2)
        self.assertEqual(rows[0]["res.flag"], "True")
        self.assertEqual(rows[0]["gen#0.length"], "2")
        self.assertEqual(rows[0]["gen#1.length"], "4")
        self.assertEqual(rows[0]["gen#1.inner.cells"], "40")
        self.assertEqual(rows[0]["gen#0.status"], "ok")
        self.assertIn("gen#0_status", cols)  # existing per-stage columns are kept
        self.assertNotIn("gen#0.time", cols)

        long = self.tmp / "long.csv"
        r.save_csv(long, layout="long")
        with long.open(encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            lcols = reader.fieldnames or []
            lrows = list(reader)
        self.assertIn("choice_branch", lcols)
        self.assertEqual([row["gen.length"] for row in lrows], ["2", "4", "2", "4"])
        self.assertEqual([row["step_index"] for row in lrows], ["0", "1", "0", "1"])
        self.assertEqual(lrows[0]["res.value"], repr(0.1 + 0.2))

    def test_missing_metrics_are_empty_cells(self) -> None:
        def pipe(run: RunContext) -> dict[str, object]:
            if compare("mode", ["with", "without"]) == "with":
                return {"extra": 1}
            return {}

        r = run_experiment(pipe, cache=self.cache)
        out = self.tmp / "m.csv"
        r.save_csv(out)
        with out.open(encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        self.assertEqual([row["res.extra"] for row in rows], ["1", ""])

    # ------------------------------------------------------------------ save / load

    def test_save_and_load_round_trip(self) -> None:
        r = run_experiment(self.sweep, cache=self.cache)
        saved_dir = r.save(self.tmp / "results" / "sweep")
        self.assertTrue((saved_dir / "report.csv").is_file())
        self.assertTrue((saved_dir / "report_long.csv").is_file())
        with self.assertRaises(FileExistsError):
            r.save(saved_dir)
        r.save(saved_dir, overwrite=True)

        saved = load_run(saved_dir)
        self.assertEqual(len(saved), 4)
        b = saved.one(solver="cg", h=0.25)
        self.assertEqual(b.choices["h"], 0.25)
        self.assertIsInstance(b.choices["h"], float)
        sol = b["solution"]
        self.assertIsInstance(sol, Artifact)
        self.assertEqual(sol.path.read_text(), "x" * len("cg-0.25"))
        # Default save(bundle="link") hard-links outputs into <saved_dir>/objects/.
        self.assertEqual(sol.path.parent, (saved_dir / "objects").resolve())
        self.assertEqual(sol.path.stat().st_ino, (self.cache.objects_dir / sol.path.name).stat().st_ino)
        self.assertEqual(b["dofs"], 400)
        self.assertEqual(b["case"], BallCase(radius=10.0, eps=3.9))
        self.assertEqual(b["note"], Path("/tmp/x"))
        self.assertEqual(b["tags"], ["cg", {"$weird": 1}])
        self.assertEqual(b.scalars, {"dofs": 400})
        self.assertEqual(b.stages[0].outputs["out"].content_id, sol.content_id)
        self.assertEqual(b.stages[0].metrics["length"], len("cg-0.25"))
        with self.assertRaises(KeyError):
            b["missing"]

        # Reloaded CSV equals the original one.
        again = self.tmp / "again.csv"
        saved.save_csv(again)
        self.assertEqual(again.read_text(), (saved_dir / "report.csv").read_text())

    def test_load_resolves_objects_under_another_cache_mount(self) -> None:
        r = run_experiment(self.sweep, cache=self.cache)
        saved_dir = r.save(self.tmp / "sweep", bundle="none")
        moved = self.tmp / "moved_cache"
        shutil.copytree(self.cache.root, moved)
        shutil.rmtree(self.cache.root)
        saved = load_run(saved_dir, cache=Cache(moved))
        sol = saved.one(solver="direct", h=0.5)["solution"]
        self.assertEqual(sol.path.parent, (moved / "objects").resolve())
        self.assertTrue(sol.path.is_file())
        with self.assertRaises(FileNotFoundError):
            load_run(saved_dir).one(solver="direct", h=0.5)["solution"]

    def test_unregistered_records_and_unserializable_values(self) -> None:
        def pipe(run: RunContext) -> dict[str, object]:
            compare("fn", {"square": lambda x: x * x})
            return {"case": BallCase(1.0, 2.0)}

        r = run_experiment(pipe, cache=self.cache)
        saved_dir = r.save(self.tmp / "s")
        raw = json.loads((saved_dir / "report.json").read_text())
        self.assertEqual(raw["branches"][0]["outputs"]["case"]["$record"], "test_ball_case")

        from poisson_runner import records

        cls = records._REGISTRY_BY_NAME.pop("test_ball_case")
        try:
            saved = load_run(saved_dir)
        finally:
            records._REGISTRY_BY_NAME["test_ball_case"] = cls
        b = saved.branches[0]
        self.assertIsInstance(b["case"], UnresolvedRecord)
        self.assertEqual(b["case"].data, {"radius": 1.0, "eps": 2.0, "charge": -1})
        self.assertIsInstance(b.choices["fn"], Unserializable)
        self.assertEqual(b.choice_labels["fn"], "square")

        # Outputs are strict: an unsavable returned value fails save(), not silently.
        bad = run_experiment(lambda: {"thing": _Opaque()}, cache=self.cache)
        with self.assertRaises(RecordError):
            bad.save(self.tmp / "bad")
        self.assertFalse((self.tmp / "bad" / "report.json").exists())
        bad.save_json(self.tmp / "bad.json")  # the lenient JSON export still works

    def test_non_finite_and_non_dict_results(self) -> None:
        r = run_experiment(lambda: [float("nan"), float("inf")], cache=self.cache)
        saved = load_run(r.save(self.tmp / "nf"))
        vals = saved.branches[0]["result"]
        self.assertTrue(math.isnan(vals[0]))
        self.assertEqual(vals[1], float("inf"))
        none = load_run(run_experiment(lambda: None, cache=self.cache).save(self.tmp / "none"))
        self.assertEqual(none.branches[0].outputs, {})

    def test_save_rejects_dry_run_and_partial_shards(self) -> None:
        dry = run_experiment(self.sweep, cache=self.cache, dry_run=True)
        with self.assertRaises(ValueError):
            dry.save(self.tmp / "dry")
        shard = run_experiment(self.sweep, cache=self.cache, shard=(0, 2))
        with self.assertRaises(ValueError):
            shard.save(self.tmp / "shard")
        shard.save(self.tmp / "shard", allow_partial=True)
        self.assertEqual(load_run(self.tmp / "shard").report.shard, (0, 2))

    def test_load_rejects_plain_json_export(self) -> None:
        legacy = self.tmp / "legacy.json"
        legacy.write_text(json.dumps({"branches": []}))
        with self.assertRaises(ValueError):
            load_run(legacy)

    # ------------------------------------------------------------------ select / match

    def test_select_one_and_match(self) -> None:
        def pipe(run: RunContext) -> dict[str, object]:
            solver = compare("solver", ["direct", "cg"])
            if solver == "cg":
                tol = compare("tol", ["1e-6", "1e-10"])
                if tol == "1e-10":
                    raise RuntimeError("diverged")
            compare("h", [0.5, 0.25])
            return {}

        r = run_experiment(pipe, cache=self.cache, continue_on_error=True)
        saved = load_run(r.save(self.tmp / "m"))
        self.assertEqual(len(saved.select(solver="cg")), 3)  # failed branch included
        self.assertEqual(len(saved.successful), 4)
        b = saved.one(solver="cg", tol="1e-6", h=0.25)
        ref = saved.match(b, solver="direct", tol=None)
        self.assertEqual(ref.choice_labels, {"solver": "direct", "h": "0.25"})
        with self.assertRaises(LookupError):  # tol=1e-6 does not exist for direct
            saved.match(b, solver="direct")
        with self.assertRaises(LookupError):  # two branches match once h is dropped
            saved.match(b, solver="direct", tol=None, h=None)
        with self.assertRaises(LookupError):  # the only match failed
            saved.match(b, tol="1e-10")
        with self.assertRaises(KeyError):
            saved.select(solvr="cg")

    # ------------------------------------------------------------------ evaluate

    def test_evaluate_is_a_separate_cached_experiment(self) -> None:
        def sweep(run: RunContext) -> dict[str, object]:
            solver = compare("solver", ["direct", "cg"])
            h = compare("h", [0.5, 0.25])
            if solver == "cg" and h == 0.25:
                raise RuntimeError("diverged")
            out = run.run_stage(self.gen, settings={"tag": f"{solver}-{h}"})
            return {"solution": out}

        r = run_experiment(sweep, cache=self.cache, continue_on_error=True)
        saved = load_run(r.save(self.tmp / "sweep"))
        gen_runs = self.executions("gen:")

        def analysis(run: RunContext, b):
            ref = saved.match(b, solver="direct")
            norm = compare("h", ["l2", "max"])  # clashes with the saved `h` choice
            d = run.run_stage(self.diff, inputs={"a": b["solution"], "b": ref["solution"]})
            assert isinstance(d, Artifact)
            return {"abs_diff": json.loads(d.path.read_text())["abs_diff"], "norm": norm}

        ana = evaluate(saved, analysis)  # cache defaults to the saved run's cache
        self.assertEqual(self.executions("gen:"), gen_runs)  # the sweep never re-runs
        self.assertEqual(len(ana.branches), 3 * 2)  # failed source branch skipped
        first = ana.branches[0]
        self.assertEqual(list(first.choice_labels), [SOURCE_BRANCH, "solver", "h", "eval.h"])
        self.assertEqual(first.choices["h"], 0.5)
        by = {(b.choice_labels["solver"], b.choice_labels["h"]): b for b in ana.branches if b.choice_labels["eval.h"] == "l2"}
        self.assertEqual(by[("direct", "0.5")]["abs_diff"], 0)
        self.assertEqual(by[("cg", "0.5")]["abs_diff"], len("direct-0.5") - len("cg-0.5"))
        diff_runs = self.executions("diff")
        self.assertEqual(diff_runs, 3)  # l2/max branches share one diff request each

        again = evaluate(self.tmp / "sweep", analysis)
        self.assertEqual(self.executions("diff"), diff_runs)  # all hits
        self.assertTrue(all(s.cached for b in again.branches for s in b.stages))

        out = self.tmp / "ana.csv"
        ana.save_csv(out)
        with out.open(encoding="utf-8") as fp:
            cols = csv.DictReader(fp).fieldnames or []
        for col in (SOURCE_BRANCH, "solver", "h", "eval.h", "res.abs_diff", "res.norm", "diff.abs_diff"):
            self.assertIn(col, cols)
        ana.save(self.tmp / "ana")  # an evaluation is itself a saveable run

    def test_evaluate_requires_successful_branches(self) -> None:
        def broken(run: RunContext) -> None:
            raise RuntimeError("no")

        r = run_experiment(broken, cache=self.cache, continue_on_error=True)
        saved = load_run(r.save(self.tmp / "broken"))
        with self.assertRaises(ValueError):
            evaluate(saved, lambda run, b: None)

    def test_to_dataframe_has_typed_columns(self) -> None:
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        r = run_experiment(self.sweep, cache=self.cache)
        df = r.to_dataframe()
        self.assertEqual(list(df["res.dofs"]), [200, 400, 200, 400])
        self.assertEqual(df["gen.length"].dtype.kind, "i")

    # ------------------------------------------------------------------ review regressions

    def test_failed_overwrite_keeps_previous_bundle_and_paths_in_settings_save(self) -> None:
        def with_path(run: RunContext) -> dict[str, object]:
            out = run.run_stage(self.gen, settings={"tag": "ab", "geo": Path("/tmp/a.geo")}, cache="off")
            return {"solution": out}

        self.gen = Stage.from_cli(
            "gen", binary=self.tmp / "gen_bin", setting_flags={"tag": "--tag"}, output_specs={"out": ".txt"},
            track_shared_libs=False,
        )
        good = run_experiment(with_path, cache=self.cache)
        d = good.save(self.tmp / "bundle")
        loaded = load_run(d)
        self.assertEqual(loaded.branches[0].stages[0].settings["geo"], Path("/tmp/a.geo"))
        before = {n: (d / n).read_bytes() for n in ("report.json", "report.csv", "report_long.csv")}

        bad = run_experiment(lambda: {"thing": _Opaque()}, cache=self.cache)
        with self.assertRaises(RecordError):
            bad.save(d, overwrite=True)
        after = {n: (d / n).read_bytes() for n in before}
        self.assertEqual(before, after)

        # A failure while staging the files (here: the long CSV) publishes nothing either.
        late = run_experiment(with_path, cache=self.cache)
        real_save_csv = late.save_csv

        def failing_save_csv(path, *, layout="wide"):
            if layout == "long":
                raise OSError("disk full")
            real_save_csv(path, layout=layout)

        late.save_csv = failing_save_csv  # type: ignore[method-assign]
        with self.assertRaises(OSError):
            late.save(d, overwrite=True)
        self.assertEqual(before, {n: (d / n).read_bytes() for n in before})
        self.assertEqual(sorted(p.name for p in d.iterdir() if not p.name.startswith(".report.lock")),
                         ["objects", "report.csv", "report.json", "report_long.csv"])

        with (d / ".report.lock").open("a") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            with self.assertRaises(RuntimeError):
                good.save(d, overwrite=True)

    def test_evaluate_never_overwrites_source_choice_names(self) -> None:
        def sweep(run: RunContext) -> dict[str, object]:
            compare("x", ["a"])
            compare("eval.x", ["b"])
            return {}

        saved = load_run(run_experiment(sweep, cache=self.cache).save(self.tmp / "s"))

        def analysis(run: RunContext, b):
            compare("x", ["c"])
            return {}

        ana = evaluate(saved, analysis)
        self.assertEqual(
            ana.branches[0].choice_labels, {SOURCE_BRANCH: "0", "x": "a", "eval.x": "b", "eval.x~2": "c"}
        )

    def test_match_requires_the_same_set_of_choices(self) -> None:
        def sweep(run: RunContext) -> dict[str, object]:
            solver = compare("solver", ["direct", "cg"])
            if solver == "cg":
                compare("precond", ["ilu"])
            compare("h", [0.5])
            return {}

        saved = load_run(run_experiment(sweep, cache=self.cache).save(self.tmp / "s"))
        direct = saved.one(solver="direct")
        cg = saved.one(solver="cg")
        with self.assertRaises(LookupError):  # direct has no precond=ilu
            saved.match(cg, solver="direct")
        with self.assertRaises(LookupError):  # cg has an extra choice
            saved.match(direct, solver="cg")
        self.assertIs(saved.match(direct, solver="cg", precond=None), cg)
        self.assertIs(saved.match(cg, solver="direct", precond=None), direct)

    def test_colliding_column_names_keep_every_value(self) -> None:
        self.assertEqual(
            info_scalars({"details.count": 1, "details": {"count": 2}}), {"details.count": 1, "details.count~2": 2}
        )
        res_stage = Stage.from_cli(
            "res", binary=self.tmp / "gen_bin", setting_flags={"tag": "--tag"}, output_specs={"out": ".txt"},
            track_shared_libs=False,
        )

        def pipe(run: RunContext) -> dict[str, object]:
            run.run_stage(res_stage, settings={"tag": "abc"})  # reports length=3 -> `res.length`
            return {"length": 10}

        r = run_experiment(pipe, cache=self.cache)
        for layout in ("wide", "long"):
            out = self.tmp / f"{layout}.csv"
            r.save_csv(out, layout=layout)
            with out.open(encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                cols = reader.fieldnames or []
                row = next(reader)
            self.assertEqual(len(cols), len(set(cols)), layout)
            self.assertEqual(row["res.length"], "10", layout)
            self.assertEqual(row["res.length~2"], "3", layout)

    def test_evaluate_from_directory_with_a_separate_evaluation_cache(self) -> None:
        r = run_experiment(self.sweep, cache=self.cache)
        d = r.save(self.tmp / "sweep")
        eval_cache = Cache(self.tmp / "eval_cache")

        def analysis(run: RunContext, b):
            out = run.run_stage(self.diff, inputs={"a": b["solution"], "b": b["solution"]})
            return {"diff": out}

        ana = evaluate(d, analysis, cache=eval_cache)
        self.assertEqual(len(ana.branches), 4)
        self.assertTrue(all(b.ok for b in ana.branches))
        self.assertEqual(ana.cache_root, eval_cache.root)
        self.assertEqual(ana.branches[0]["diff"].path.parent, eval_cache.objects_dir)

    def test_json_exports_and_dry_runs_are_not_saved_runs(self) -> None:
        dry = run_experiment(self.sweep, cache=self.cache, dry_run=True)
        dry.save_json(self.tmp / "dry.json")
        with self.assertRaises(ValueError):
            load_run(self.tmp / "dry.json")
        from poisson_runner import SavedRun

        with self.assertRaises(ValueError):
            evaluate(SavedRun(self.tmp, dry), lambda run, b: None)

    def test_codec_preserves_values_exactly(self) -> None:
        with self.assertRaises(RecordError):
            run_experiment(lambda: {1: 12, "1": 43}, cache=self.cache).save(self.tmp / "keys")

        def pipe(run: RunContext) -> dict[str, object]:
            mesh = compare("mesh", {"coarse": ("coarse.msh", 1e-4)})
            return {"pair": mesh, "case": MeshCase(Path("/tmp/b.geo"), "$path")}

        d = run_experiment(pipe, cache=self.cache).save(self.tmp / "tuples")
        b = load_run(d).branches[0]
        self.assertEqual(b.choices["mesh"], ("coarse.msh", 1e-4))
        self.assertEqual(b["pair"], ("coarse.msh", 1e-4))
        self.assertEqual(b["case"], MeshCase(Path("/tmp/b.geo"), "$path", (1, 2)))

        # An unresolved record survives a save/load cycle without the type being registered.
        from poisson_runner import records

        cls = records._REGISTRY_BY_NAME.pop("test_mesh_case")
        try:
            unresolved = load_run(d)
            self.assertIsInstance(unresolved.branches[0]["case"], UnresolvedRecord)
            d2 = unresolved.report.save(self.tmp / "resaved")
        finally:
            records._REGISTRY_BY_NAME["test_mesh_case"] = cls
        self.assertEqual(load_run(d2).branches[0]["case"], MeshCase(Path("/tmp/b.geo"), "$path", (1, 2)))

    def test_record_registration_rejects_ambiguous_or_unloadable_types(self) -> None:
        twin = dataclasses.make_dataclass("BallCase", [("radius", float)], module="some_other_module")
        with self.assertRaises(ValueError):
            register_record("test_ball_case")(twin)

        @dataclass
        class Derived:
            a: int
            b: int = dataclasses.field(init=False, default=0)

        with self.assertRaises(TypeError):
            register_record("test_derived")(Derived)

    def test_artifact_execution_round_trips_and_headers_are_stable(self) -> None:
        cold = run_experiment(self.sweep, cache=self.cache)
        warm = run_experiment(self.sweep, cache=self.cache)
        self.assertTrue(all(s.cached for b in warm.branches for s in b.stages))
        for layout in ("wide", "long"):
            a, b = self.tmp / f"cold_{layout}.csv", self.tmp / f"warm_{layout}.csv"
            cold.save_csv(a, layout=layout)
            warm.save_csv(b, layout=layout)
            self.assertEqual(a.read_text().splitlines()[0], b.read_text().splitlines()[0], layout)
        saved = load_run(cold.save(self.tmp / "exec"))
        self.assertIn("wall_time_s", saved.branches[0]["solution"].execution)

    # ------------------------------------------------------------------ bundling (#1 & #4)

    def test_save_link_survives_cache_clean_and_lru_ignores_shared(self) -> None:
        r = run_experiment(self.sweep, cache=self.cache)
        saved_dir = r.save(self.tmp / "linked_run")  # default: bundle="link"

        st = self.cache.stats()
        self.assertGreater(st.shared_count, 0)
        self.assertEqual(st.shared_count, st.object_count)
        self.assertEqual(st.shared_bytes, st.object_bytes)

        # LRU eviction skips objects with st_nlink > 1 because unlinking frees 0 bytes.
        evicted = self.cache.evict_lru(max_bytes=0)
        self.assertEqual(evicted, 0)
        self.assertEqual(self.cache.stats().object_count, st.object_count)

        # cache clean --all removes cache names, but the saved run's hard links keep data intact.
        res = self.cache.clean(all_entries=True)
        self.assertGreater(res["removed_objects"], 0)
        self.assertEqual(self.cache.stats().object_count, 0)

        saved = load_run(saved_dir)
        ana = evaluate(saved, lambda run, b: {"len": len(b["solution"].path.read_text())}, cache=self.cache)
        self.assertEqual(len(ana.branches), 4)
        self.assertTrue(all(b.ok for b in ana.branches))

    def test_exdev_fallback_link_vs_link_or_copy_and_cli_bundle_status(self) -> None:
        import errno
        import warnings
        from unittest import mock
        from poisson_runner import cli, saved_run_status

        r = run_experiment(self.sweep, cache=self.cache)

        def fail_exdev(*_args, **_kwargs):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        # 1. Default bundle="link" on EXDEV warns once and leaves references in cache.
        link_dir = self.tmp / "exdev_link"
        with mock.patch("os.link", side_effect=fail_exdev), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r.save(link_dir)
        self.assertEqual(len(caught), 1)
        self.assertIn("run bundle", str(caught[0].message))
        st = saved_run_status(link_dir)
        self.assertEqual(st.bundled, 0)
        self.assertEqual(st.cache_only, st.total)
        self.assertEqual(st.missing, 0)

        # 2. CLI `run bundle` copies across EXDEV and verifies BLAKE3; `run status --verify` passes.
        with mock.patch("os.link", side_effect=fail_exdev):
            code = cli.main(["run", "bundle", str(link_dir)])
        self.assertEqual(code, 0)
        self.assertEqual(cli.main(["run", "status", str(link_dir), "--verify"]), 0)
        st2 = saved_run_status(link_dir, verify=True)
        self.assertEqual(st2.bundled, st2.total)
        self.assertEqual(st2.cache_only, 0)
        self.assertEqual(st2.corrupted, ())

        # 3. Direct bundle="link_or_copy" on EXDEV copies during save().
        copy_dir = self.tmp / "exdev_copy"
        with mock.patch("os.link", side_effect=fail_exdev):
            r.save(copy_dir, bundle="link_or_copy")
        self.cache.clean(all_entries=True)
        loaded = load_run(copy_dir)
        self.assertEqual(loaded.one(solver="cg", h=0.25)["solution"].path.read_text(), "x" * len("cg-0.25"))

    def test_ephemeral_and_allow_recompute_outputs_always_materialized(self) -> None:
        from poisson_runner import experiment_main

        # cache="off" writes to ephemeral/; even bundle="none" must materialize ephemeral outputs.
        r = run_experiment(self.sweep, cache=self.cache, cache_mode="off")
        saved_dir = r.save(self.tmp / "ephemeral_run", bundle="none")
        self.cache.clean(all_entries=True)
        loaded = load_run(saved_dir)
        self.assertEqual(loaded.one(solver="direct", h=0.5)["solution"].path.read_text(), "x" * len("direct-0.5"))

        # --collect --allow-recompute writes recomputed stages to ephemeral/; save must keep them alive.
        out_dir = self.tmp / "collect_recompute"
        code = experiment_main(
            self.sweep,
            argv=["--collect", "--allow-recompute", "--out", str(out_dir)],
            cache=self.cache,
        )
        self.assertEqual(code, 0)
        self.cache.clean(all_entries=True)
        loaded2 = load_run(out_dir)
        self.assertEqual(loaded2.one(solver="cg", h=0.5)["solution"].path.read_text(), "x" * len("cg-0.5"))

    def test_partial_bundle_overwrite_pruning_old_v1_and_missing_eviction(self) -> None:
        from poisson_runner import bundle_saved_run, cli, saved_run_status

        r = run_experiment(self.sweep, cache=self.cache)
        d = r.save(self.tmp / "partial_run", bundle="link")
        objs = sorted((d / "objects").iterdir())
        self.assertGreaterEqual(len(objs), 2)
        # Remove one bundled object: that artifact falls back to cache, others stay bundled.
        removed_name = objs[0].name
        objs[0].unlink()
        loaded = load_run(d)
        parents = {b["solution"].path.parent for b in loaded.branches}
        self.assertIn((d / "objects").resolve(), parents)
        self.assertIn(self.cache.objects_dir, parents)
        st = saved_run_status(d)
        self.assertEqual(st.cache_only, 1)
        self.assertEqual(st.bundled, st.total - 1)

        # Overwrite with a smaller run prunes unreferenced objects from <dir>/objects/.
        small = run_experiment(lambda run: {"solution": run.run_stage(self.gen, settings={"tag": "only-one"})}, cache=self.cache)
        small.save(d, overwrite=True)
        remaining = [p.name for p in (d / "objects").iterdir()]
        self.assertEqual(len(remaining), 1)
        self.assertNotIn(removed_name, remaining)

        # Old format_version=1 report (without location/ext) still loads cleanly.
        raw = json.loads((d / "report.json").read_text())
        raw["format_version"] = 1
        for b in raw["branches"]:
            b["outputs"]["solution"]["$artifact"].pop("location", None)
            b["outputs"]["solution"]["$artifact"].pop("ext", None)
        (d / "report.json").write_text(json.dumps(raw))
        self.assertEqual(load_run(d).branches[0]["solution"].path.read_text(), "x" * len("only-one"))

        # Evicting an object before bundling reports it clearly and exits non-zero.
        d2 = r.save(self.tmp / "evicted_run", bundle="none")
        self.cache.clean(all_entries=True)
        summary = bundle_saved_run(d2)
        self.assertGreater(summary.missing, 0)
        self.assertEqual(cli.main(["run", "bundle", str(d2)]), 1)

    # ------------------------------------------------------------------ issue #2: sharded evaluate reads outputs

    def test_sharded_evaluate_reading_stage_outputs_on_cold_cache(self) -> None:
        from poisson_runner import DryRunPlaceholderError, experiment_main

        r = run_experiment(self.sweep, cache=self.cache)
        saved_dir = r.save(self.tmp / "eval_src")
        saved = load_run(saved_dir)
        eval_cache = Cache(self.tmp / "eval_cold_cache")

        def analysis(run: RunContext, b):
            ref = saved.match(b, solver="direct")
            d = run.run_stage(self.diff, inputs={"a": b["solution"], "b": ref["solution"]})
            # Exercise read_text, open(), with_suffix(), info[], info.get(), execution[] without `if not run.dry_run:`.
            text = d.path.read_text()
            with open(d.path, encoding="utf-8") as fp:
                text2 = fp.read()
            assert text == text2
            delta = d.info["abs_diff"]
            delta2 = d.info.get("abs_diff")
            wall = d.execution["wall_time_s"]
            return {"diff": d, "delta": delta, "delta2": delta2, "wall": wall}

        # Direct placeholder checks in dry_run: path, derived path, info, execution all raise DryRunPlaceholderError.
        seen_errors: list[type] = []

        def probe(run: RunContext) -> None:
            out = run.run_stage(self.gen, settings={"tag": "probe-cold"})
            for fn in (
                lambda: out.path.read_text(),
                lambda: open(out.path),
                lambda: out.path.exists(),
                lambda: out.path.with_suffix(".info.json").read_text(),
                lambda: out.info["length"],
                lambda: out.info.get("length"),
                lambda: out.execution["wall_time_s"],
            ):
                try:
                    fn()
                except DryRunPlaceholderError as exc:
                    seen_errors.append(type(exc))
                    self.assertEqual(exc.stage, "gen")
            self.assertEqual(out.path.suffix, ".txt")
            self.assertEqual(out.path.name, "out.txt")

        run_experiment(probe, cache=eval_cache, dry_run=True)
        self.assertEqual(len(seen_errors), 7)

        # Sharded evaluate on a completely cold cache succeeds across both shards!
        s0 = evaluate(saved, analysis, cache=eval_cache, shard=(0, 2))
        s1 = evaluate(saved, analysis, cache=eval_cache, shard=(1, 2))
        self.assertEqual(len(s0.branches) + len(s1.branches), 4)
        self.assertTrue(all(b.ok for b in s0.branches + s1.branches))
        self.assertEqual({b["delta"] for b in s0.branches + s1.branches}, {0, abs(len("cg-0.5") - len("direct-0.5"))})

        # `--count` on the same evaluation pipeline agrees on branch count and notes the dry-run stop.
        fresh_cache = Cache(self.tmp / "eval_count_cache")
        buf = io.StringIO()
        from contextlib import redirect_stdout

        def eval_pipe(run: RunContext):
            b = run.compare("source_branch", {str(x.index): x for x in saved.successful})
            return analysis(run, b)

        with redirect_stdout(buf):
            code = experiment_main(eval_pipe, argv=["--count"], cache=fresh_cache)
        self.assertEqual(code, 0)
        self.assertIn("branches: 4", buf.getvalue())
        self.assertIn("stopped during dry run", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
