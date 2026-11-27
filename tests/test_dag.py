"""Tests for DAG mode: planning stage-level tasks, grouping them into arrays, running one task."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import Artifact, Cache, RunContext, Stage, compare, experiment_main  # noqa: E402
from poisson_runner import executor as executor_module  # noqa: E402
from poisson_runner import Compare, run_experiment  # noqa: E402
from poisson_runner.dag import parse_resources, plan_dag, run_dag_task  # noqa: E402


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class DagTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_dag_test_")
        self.tmp = Path(self._tmp.name)
        self._old_env = {k: os.environ.pop(k, None) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID")}
        self.cache = Cache(self.tmp / "cache")
        self.mesh, self.mesh_log = self._stage("mesh")
        self.assemble, self.assemble_log = self._stage("assemble")
        self.solve, self.solve_log = self._stage("solve")

    def tearDown(self) -> None:
        for k, v in self._old_env.items():
            if v is not None:
                os.environ[k] = v
        self._tmp.cleanup()

    def _stage(self, name: str) -> tuple[Stage, Path]:
        log = self.tmp / f"{name}.log"
        binary = _write_executable(
            self.tmp / f"{name}_bin",
            f"""
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log)!r}, "a") as f:
    f.write(a.tag + "\\n")
src = pathlib.Path(a.inp).read_text() + ">" if a.inp else ""
pathlib.Path(a.out).write_text(src + a.tag)
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
    def _lines(log: Path) -> list[str]:
        return log.read_text().splitlines() if log.exists() else []

    def _pipeline(self, *, fine_resources=None, solve_cache=None):
        def pipe(run: RunContext) -> dict[str, Artifact]:
            m = compare("mesh", ["coarse", "fine"])
            res = fine_resources if m == "fine" else None
            msh = run.run_stage(self.mesh, settings={"tag": m}, resources=res)
            order = compare("order", [1, 2])
            asm = run.run_stage(self.assemble, inputs={"inp": msh}, settings={"tag": f"{m}-p{order}"}, resources=res)
            solver = compare("solver", ["cg", "lu"])
            extra = {"cache": solve_cache} if solve_cache is not None else {}
            sol = run.run_stage(
                self.solve, inputs={"inp": asm}, settings={"tag": f"{m}-p{order}-{solver}"}, resources=res, **extra
            )
            return {"solution": sol}

        return pipe

    def _run_all(self, pipe, plan) -> None:
        payload = plan.to_dict()
        for g in payload["groups"]:
            for e in range(len(g["units"])):
                run_dag_task(pipe, payload, g["id"], e, cache=self.cache)

    # ------------------------------------------------------------------ planning

    def test_plan_tasks_groups_and_full_run_then_collect(self) -> None:
        pipe = self._pipeline()
        plan = plan_dag(pipe, cache=self.cache)
        self.assertEqual(len(plan.tasks), 2 + 4 + 8)
        self.assertEqual(plan.shared_calls, (8 - 2) + (8 - 4))  # each mesh/assembly is shared by later branches
        self.assertEqual([(g.stage, len(g.units), g.parents) for g in plan.groups],
                         [("mesh", 2, []), ("assemble", 4, [0]), ("solve", 8, [1])])
        self.assertEqual(plan.leaves, [2])
        self.assertEqual(plan.groups[0].resources, {"cpus": 1, "time": "01:00:00", "mem_per_cpu": "3900M"})
        for t in plan.tasks:
            for p in t.parents:
                self.assertLess(p, t.id)

        self._run_all(pipe, plan)
        self.assertEqual(len(self._lines(self.mesh_log)), 2)
        self.assertEqual(len(self._lines(self.assemble_log)), 4)
        self.assertEqual(sorted(self._lines(self.solve_log)), sorted(
            f"{m}-p{o}-{s}" for m in ("coarse", "fine") for o in (1, 2) for s in ("cg", "lu")))

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=["--collect", "--out", str(self.tmp / "res")], cache=self.cache)
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(len(self._lines(self.solve_log)), 8)  # collect computed nothing
        self.assertEqual(plan_dag(pipe, cache=self.cache).tasks, [])  # replanning: nothing left

    def test_resources_split_by_and_array_chunks(self) -> None:
        fine = {"cpus": 16, "time": "04:00:00"}
        plan = plan_dag(self._pipeline(fine_resources=fine), cache=self.cache,
                        stage_resources={"solve": {"cpus": "4"}})
        shape = [(g.stage, g.resources["cpus"], len(g.units), g.parents) for g in plan.groups]
        self.assertEqual(shape, [
            ("mesh", 1, 1, []), ("assemble", 1, 2, [0]), ("solve", 4, 4, [1]),
            ("mesh", 16, 1, []), ("assemble", 16, 2, [3]), ("solve", 16, 4, [4]),
        ])
        self.assertEqual(plan.groups[5].resources, {"cpus": 16, "time": "04:00:00", "mem_per_cpu": "3900M"})

        split = plan_dag(self._pipeline(), cache=self.cache, split_by="mesh")
        self.assertEqual([(g.stage, g.split, g.parents) for g in split.groups], [
            ("mesh", ("coarse",), []), ("assemble", ("coarse",), [0]), ("solve", ("coarse",), [1]),
            ("mesh", ("fine",), []), ("assemble", ("fine",), [3]), ("solve", ("fine",), [4]),
        ])
        # Two names: a task's label per name is the one all its branches agree on.
        two = plan_dag(self._pipeline(), cache=self.cache, split_by=["mesh", "order"])
        self.assertEqual(two.split_by, ["mesh", "order"])
        self.assertEqual([(g.stage, g.split) for g in two.groups][:3],
                         [("mesh", ("coarse", None)), ("assemble", ("coarse", "1")), ("solve", ("coarse", "1"))])
        self.assertEqual(len(two.groups), 2 + 4 + 4)
        with self.assertRaises(ValueError):
            plan_dag(self._pipeline(), cache=self.cache, split_by="meshes")

        chunked = plan_dag(self._pipeline(), cache=self.cache, max_array_size=3)
        self.assertTrue(all(len(g.units) <= 3 for g in chunked.groups))
        self.assertEqual(sum(len(g.units) for g in chunked.groups), 14)
        for g in chunked.groups:
            for u in g.units:
                unit = chunked.units[u]
                self.assertEqual(sorted({chunked.units[p].group for p in unit.parents}), g.parents)

        with self.assertRaises(ValueError):
            plan_dag(self._pipeline(fine_resources={"gpus": 1}), cache=self.cache)
        with self.assertRaises(ValueError):
            parse_resources("cpus=0")
        self.assertEqual(parse_resources("cpus=8,time=1-00:00:00,mem-per-cpu=8G"),
                         {"cpus": 8, "time": "1-00:00:00", "mem_per_cpu": "8G"})

    def test_plan_rejects_non_writing_modes_and_conflicting_resources(self) -> None:
        with self.assertRaisesRegex(ValueError, "doesn't write"):
            plan_dag(self._pipeline(solve_cache="off"), cache=self.cache)

        def conflict(run: RunContext) -> None:
            k = compare("k", [1, 2])
            run.run_stage(self.mesh, settings={"tag": "shared"}, resources={"cpus": k})

        with self.assertRaisesRegex(ValueError, "same resources"):
            plan_dag(conflict, cache=self.cache)

    def test_refresh_then_use_of_the_same_request_is_one_task(self) -> None:
        pipe = self._pipeline()
        self._run_all(pipe, plan_dag(pipe, cache=self.cache))  # everything cached

        def mixed(run: RunContext) -> None:
            k = compare("k", ["a", "b"])
            msh = run.run_stage(self.mesh, settings={"tag": "coarse"}, cache="refresh" if k == "a" else "use")
            run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": f"after-{k}"})

        plan = plan_dag(mixed, cache=self.cache)
        # The refreshed mesh is one task, and both solves wait for it instead of reading the old entry.
        self.assertEqual([(t.stage, t.parents) for t in plan.tasks], [("mesh", []), ("solve", [0]), ("solve", [0])])

    # ------------------------------------------------------------------ running tasks

    def test_task_needs_only_its_parents_and_guards_mismatches(self) -> None:
        pipe = self._pipeline()
        payload = plan_dag(pipe, cache=self.cache).to_dict()
        solve_group = next(g for g in payload["groups"] if g["stage"] == "solve")
        with self.assertRaisesRegex(RuntimeError, "not in the cache"):
            run_dag_task(pipe, payload, solve_group["id"], 0, cache=self.cache)
        self.assertEqual(self._lines(self.solve_log), [])

        # Independent earlier steps need not be done: the charges task of branch 1 runs on its own.
        def two_roots(run: RunContext) -> None:
            run.run_stage(self.mesh, settings={"tag": "root-mesh"})
            k = compare("k", ["x", "y"])
            run.run_stage(self.assemble, settings={"tag": f"charges-{k}"})

        plan2 = plan_dag(two_roots, cache=self.cache).to_dict()
        charges = next(g for g in plan2["groups"] if g["stage"] == "assemble")
        result = run_dag_task(two_roots, plan2, charges["id"], 1, cache=self.cache)
        self.assertEqual((result.stage, result.branch, result.status), ("assemble", 1, "MISS"))
        self.assertEqual(self._lines(self.mesh_log), [])
        again = run_dag_task(two_roots, plan2, charges["id"], 1, cache=self.cache)
        self.assertEqual(again.status, "HIT")

        # A changed experiment (other settings at that call) is refused, not silently run.
        def edited(run: RunContext) -> None:
            run.run_stage(self.mesh, settings={"tag": "root-mesh"})
            k = compare("k", ["x", "y"])
            run.run_stage(self.assemble, settings={"tag": f"edited-{k}"})

        with self.assertRaisesRegex(RuntimeError, "Plan mismatch"):
            run_dag_task(edited, plan2, charges["id"], 0, cache=self.cache)

        def shorter(run: RunContext) -> None:
            compare("k", ["x", "y"])

        with self.assertRaisesRegex(RuntimeError, "Plan mismatch"):
            run_dag_task(shorter, plan2, charges["id"], 0, cache=self.cache)
        with self.assertRaises(ValueError):
            run_dag_task(two_roots, plan2, 99, 0, cache=self.cache)

    def test_except_exception_in_pipeline_cannot_swallow_the_stop(self) -> None:
        after = []

        def guarded(run: RunContext) -> None:
            try:
                run.run_stage(self.mesh, settings={"tag": "only"})
            except Exception:  # noqa: BLE001 - the point of the test
                after.append("swallowed")
            after.append("continued")

        payload = plan_dag(guarded, cache=self.cache).to_dict()
        planned = list(after)
        self.assertEqual(set(planned), {"continued"})  # planning runs the whole pipeline
        run_dag_task(guarded, payload, 0, 0, cache=self.cache)
        self.assertEqual(after, planned)  # the task's replay stopped at its stage, not swallowed
        self.assertEqual(self._lines(self.mesh_log), ["only"])

    # ------------------------------------------------------------------ review regressions

    def test_task_never_reuses_a_placeholder_for_identical_outputs(self) -> None:
        const, _ = self._stage("const")
        const_bin = self.tmp / "const_bin"  # same output bytes whatever the tag
        const_bin.write_text(const_bin.read_text().replace("pathlib.Path(a.out).write_text(src + a.tag)",
                                                           "pathlib.Path(a.out).write_text('same')"))

        def pipe(run: RunContext) -> None:
            p1 = run.run_stage(const, settings={"tag": "a"})
            p2 = run.run_stage(const, settings={"tag": "b"})
            run.run_stage(self.solve, inputs={"inp": p1}, settings={"tag": "x"}, resources={"cpus": 1})
            run.run_stage(self.solve, inputs={"inp": p2}, settings={"tag": "x"}, resources={"cpus": 2})

        payload = plan_dag(pipe, cache=self.cache, fuse=False).to_dict()  # each call its own task
        self.assertEqual([t["stage"] for t in payload["tasks"]], ["const", "const", "solve", "solve"])
        by_task = {payload["units"][u]["tasks"][0]: (g["id"], i)
                   for g in payload["groups"] for i, u in enumerate(g["units"])}
        for t in (0, 1, 3, 2):  # the second solve first: the first one's call then misses during its replay
            run_dag_task(pipe, payload, *by_task[t], cache=self.cache)
        self.assertEqual(self._lines(self.solve_log), ["x"])  # executed once, then a real cache hit
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=["--collect", "--out", str(self.tmp / "res")], cache=self.cache)
        self.assertEqual(code, 0, err.getvalue())

    def test_collect_finds_outputs_of_calls_that_ask_for_refresh(self) -> None:
        pipe = self._pipeline(solve_cache="refresh")
        self._run_all(pipe, plan_dag(pipe, cache=self.cache))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=["--collect", "--out", str(self.tmp / "res")], cache=self.cache)
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(len(self._lines(self.solve_log)), 8)

    def test_plan_refuses_refreshing_a_request_read_earlier(self) -> None:
        def seeded(run: RunContext) -> None:
            run.run_stage(self.mesh, settings={"tag": "m"})

        self._run_all(seeded, plan_dag(seeded, cache=self.cache))

        def later_refresh(run: RunContext) -> None:
            k = compare("k", ["use", "refresh"])
            msh = run.run_stage(self.mesh, settings={"tag": "m"}, cache=k)
            run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": k})

        with self.assertRaisesRegex(ValueError, "already read from the cache"):
            plan_dag(later_refresh, cache=self.cache)

    def test_task_refuses_edited_inputs_and_keeps_planned_options(self) -> None:
        raw = self.tmp / "input.txt"
        raw.write_text("v1")

        def pipe(run: RunContext) -> None:
            run.run_stage(self.mesh, inputs={"inp": raw}, settings={"tag": "from-file"})

        payload = plan_dag(pipe, cache=self.cache).to_dict()
        raw.write_text("v2")
        with self.assertRaisesRegex(RuntimeError, "input\\(s\\) inp differ"):
            run_dag_task(pipe, payload, 0, 0, cache=self.cache)
        self.assertEqual(self._lines(self.mesh_log), [])

        def simple(run: RunContext) -> None:
            run.run_stage(self.mesh, settings={"tag": "measured"})

        run_dag_task(simple, plan_dag(simple, cache=self.cache).to_dict(), 0, 0, cache=self.cache)
        planned = plan_dag(simple, cache=self.cache, cache_mode="refresh", repeats=3).to_dict()
        self.assertEqual(planned["tasks"][0]["repeats"], 3)
        result = run_dag_task(simple, planned, 0, 0, cache=self.cache)  # no options: the plan decides
        self.assertEqual(result.status, "MISS")
        self.assertEqual(self._lines(self.mesh_log), ["measured"] * 4)

    def test_count_is_unchanged_by_refresh_then_use(self) -> None:
        def seeded(run: RunContext) -> None:
            msh = run.run_stage(self.mesh, settings={"tag": "m"})
            run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": "s"})

        self._run_all(seeded, plan_dag(seeded, cache=self.cache))

        def pipe(run: RunContext) -> None:
            k = compare("k", ["refresh", "use"])
            msh = run.run_stage(self.mesh, settings={"tag": "m"}, cache=k)
            if k == "use":
                run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": "s"})

        from poisson_runner import run_experiment

        self.assertEqual(run_experiment(pipe, cache=self.cache, dry_run=True).cache_misses, 1)

    # ------------------------------------------------------------------ chain fusion

    def _two_input_stage(self, name: str) -> tuple[Stage, Path]:
        """A stage reading `inp` and `aux`; a tag starting with `fail` makes it exit non-zero."""
        log = self.tmp / f"{name}.log"
        binary = _write_executable(
            self.tmp / f"{name}_bin",
            f"""
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--aux", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
with open({str(log)!r}, "a") as f:
    f.write(a.tag + "\\n")
if a.tag.startswith("fail"):
    sys.exit(3)
parts = [pathlib.Path(x).read_text() for x in (a.inp, a.aux) if x]
pathlib.Path(a.out).write_text("|".join(parts) + ">" + a.tag)
pathlib.Path(a.info_out).write_text("{{}}")
""",
        )
        stage = Stage.from_cli(
            name,
            binary=binary,
            input_flags={"inp": "--in", "aux": "--aux"},
            setting_flags={"tag": "--tag"},
            output_flags={"out": "--out"},
            output_specs={"out": ".txt"},
            info_flag="--info-out",
        )
        return stage, log

    def test_fusion_runs_unshared_chains_as_one_task(self) -> None:
        def pipe(run: RunContext) -> dict[str, Artifact]:
            k = compare("k", ["a", "b"])
            msh = run.run_stage(self.mesh, settings={"tag": k})
            asm = run.run_stage(self.assemble, inputs={"inp": msh}, settings={"tag": k})
            return {"solution": run.run_stage(self.solve, inputs={"inp": asm}, settings={"tag": k})}

        unfused = plan_dag(pipe, cache=self.cache, fuse=False)
        self.assertEqual([(g.stage, len(g.units), g.parents) for g in unfused.groups],
                         [("mesh", 2, []), ("assemble", 2, [0]), ("solve", 2, [1])])

        plan_file = self.tmp / "plan.json"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=["--dag-plan", str(plan_file), "--resources", "time=00:30:00"],
                                   cache=self.cache)
        self.assertEqual(code, 0, err.getvalue())
        # One array: each branch's whole chain in one task, with the three time limits added up.
        self.assertIn("dag-group 0|2|-|1|01:30:00|3900M|mesh+assemble+solve|1", out.getvalue())
        self.assertIn("4 fused into the job before them", out.getvalue())
        payload = json.loads(plan_file.read_text())
        self.assertEqual([u["stages"] for u in payload["units"]], [["mesh", "assemble", "solve"]] * 2)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(experiment_main(pipe, argv=["--dag-plan", str(plan_file), "--dag-task", "0/1"],
                                             cache=self.cache), 0)
        lines = out.getvalue().splitlines()  # one line per stage of the chain
        self.assertEqual([ln.split("stage ")[1].split(":")[0] for ln in lines], ["'mesh'", "'assemble'", "'solve'"])
        self.assertTrue(all(": executed" in ln and "branch 1" in ln for ln in lines), lines)
        self.assertEqual(self._lines(self.solve_log), ["b"])
        result = run_dag_task(pipe, payload, 0, 0, cache=self.cache)
        self.assertEqual((result.stages, result.status), (("mesh", "assemble", "solve"), "MISS"))
        self.assertEqual(run_dag_task(pipe, payload, 0, 0, cache=self.cache).status, "HIT")

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=["--collect", "--out", str(self.tmp / "res")], cache=self.cache)
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(sorted(self._lines(self.solve_log)), ["a", "b"])  # collect computed nothing

    def test_fusion_stops_at_shared_stages_joins_and_resource_changes(self) -> None:
        def pipe(run: RunContext) -> None:
            msh = run.run_stage(self.mesh, settings={"tag": "shared"})
            k = compare("k", ["a", "b"])
            asm = run.run_stage(self.assemble, inputs={"inp": msh}, settings={"tag": k})
            sol = run.run_stage(self.solve, inputs={"inp": asm}, settings={"tag": k})
            run.run_stage(self.solve, inputs={"inp": sol}, settings={"tag": f"big-{k}"}, resources={"cpus": 8})

        plan = plan_dag(pipe, cache=self.cache)
        self.assertEqual([(g.stage, len(g.units), g.parents, g.resources["cpus"]) for g in plan.groups],
                         [("mesh", 1, [], 1), ("assemble+solve", 2, [0], 1), ("solve", 2, [1], 8)])
        self._run_all(pipe, plan)
        self.assertEqual(sorted(self._lines(self.solve_log)), ["a", "b", "big-a", "big-b"])

        join, _ = self._two_input_stage("join")

        def joined(run: RunContext) -> None:
            left = run.run_stage(self.mesh, settings={"tag": "left"})
            right = run.run_stage(self.assemble, settings={"tag": "right"})
            run.run_stage(join, inputs={"inp": left, "aux": right}, settings={"tag": "j"})

        # Fusing a parent into a join would make it wait for the other parent too.
        self.assertEqual([(g.stage, g.parents) for g in plan_dag(joined, cache=self.cache).groups],
                         [("mesh", []), ("assemble", []), ("join", [0, 1])])

    def test_fusion_ignores_redundant_skip_edges(self) -> None:
        imp, imp_log = self._two_input_stage("imp")

        def case_like(heights: list[str]):
            def pipe(run: RunContext) -> dict[str, Artifact]:
                inputs = run.run_stage(self.mesh, settings={"tag": "case"})
                h = compare("h", heights)
                msh = run.run_stage(self.assemble, inputs={"inp": inputs}, settings={"tag": h})
                return {"mesh": run.run_stage(imp, inputs={"inp": msh, "aux": inputs}, settings={"tag": h})}

            return pipe

        # `imp` reads `inputs` directly too, but `inputs` is already an ancestor via `assemble`.
        shared = plan_dag(case_like(["1", "2"]), cache=self.cache)
        self.assertEqual([(g.stage, len(g.units), g.parents) for g in shared.groups],
                         [("mesh", 1, []), ("assemble+imp", 2, [0])])
        single = case_like(["1"])
        plan = plan_dag(single, cache=self.cache)
        self.assertEqual([(g.stage, len(g.units), g.parents) for g in plan.groups], [("mesh+assemble+imp", 1, [])])
        self._run_all(single, plan)
        self.assertEqual(self._lines(imp_log), ["1"])
        self.assertEqual(plan_dag(single, cache=self.cache).tasks, [])

    def test_fused_chain_refuses_a_stage_the_pipeline_swallowed(self) -> None:
        flaky, flaky_log = self._two_input_stage("flaky")
        fallback = self.tmp / "fallback.txt"
        fallback.write_text("fallback")

        def pipe(run: RunContext) -> None:
            try:
                x: Artifact | Path = run.run_stage(flaky, settings={"tag": "fail-now"})  # type: ignore[assignment]
            except Exception:  # noqa: BLE001 - the point of the test
                x = fallback
            run.run_stage(self.solve, inputs={"inp": x}, settings={"tag": "after"})

        payload = plan_dag(pipe, cache=self.cache).to_dict()
        self.assertEqual([g["stages"] for g in payload["groups"]], [["flaky", "solve"]])
        with self.assertRaisesRegex(RuntimeError, "'flaky' \\(call 0\\) of this chain didn't finish"):
            run_dag_task(pipe, payload, 0, 0, cache=self.cache)
        self.assertEqual(self._lines(flaky_log), ["fail-now"])
        self.assertEqual(self._lines(self.solve_log), [])  # never ran on the fallback input

    def test_fused_refresh_runs_even_when_an_earlier_member_made_the_same_request(self) -> None:
        ident, ident_log = self._stage("ident")
        ident_bin = self.tmp / "ident_bin"  # output = input bytes, so a chain of it converges
        ident_bin.write_text(ident_bin.read_text().replace(
            "pathlib.Path(a.out).write_text(src + a.tag)", "pathlib.Path(a.out).write_text(src[:-1])"))
        raw = self.tmp / "raw.txt"
        raw.write_text("same")

        def pipe(run: RunContext) -> None:
            first = run.run_stage(ident, inputs={"inp": raw}, settings={"tag": "x"})
            run.run_stage(ident, inputs={"inp": first}, settings={"tag": "x"}, cache="refresh", repeats=3, warmup=1)

        payload = plan_dag(pipe, cache=self.cache).to_dict()
        self.assertEqual([g["stages"] for g in payload["groups"]], [["ident", "ident"]])
        result = run_dag_task(pipe, payload, 0, 0, cache=self.cache)
        self.assertEqual([t.status for t in result.traces], ["MISS", "MISS"])
        self.assertEqual(len(self._lines(ident_log)), 1 + 3 + 1)  # the refresh ran with its repeats

    def test_fusion_respects_time_caps_minutes_and_unlimited_stages(self) -> None:
        def chain(times: list[str]):
            def pipe(run: RunContext) -> None:
                art = None
                for i, (stage, t) in enumerate(zip((self.mesh, self.assemble, self.solve), times)):
                    inputs = {"inp": art} if art is not None else {}
                    art = run.run_stage(stage, inputs=inputs, settings={"tag": f"c{i}"}, resources={"time": t})

            return pipe

        capped = plan_dag(chain(["10:00:00"] * 3), cache=self.cache)
        self.assertEqual([(g.stage, g.resources["time"], g.parents) for g in capped.groups],
                         [("mesh+assemble", "20:00:00", []), ("solve", "10:00:00", [0])])
        roomy = plan_dag(chain(["10:00:00"] * 3), cache=self.cache, max_fused_time="5-00:00:00")
        self.assertEqual([g.resources["time"] for g in roomy.groups], ["1-06:00:00"])
        # Slurm rounds each job's limit up to a minute: one-second stages keep a minute each.
        self.assertEqual([g.resources["time"] for g in plan_dag(chain(["0:01"] * 3), cache=self.cache).groups],
                         ["00:03:00"])
        unlimited = plan_dag(chain(["01:00:00", "0", "01:00:00"]), cache=self.cache)
        self.assertEqual([(g.stage, g.resources["time"]) for g in unlimited.groups],
                         [("mesh", "01:00:00"), ("assemble", "0"), ("solve", "01:00:00")])
        with self.assertRaises(ValueError):
            plan_dag(chain(["01:00:00"] * 3), cache=self.cache, max_fused_time="0")

    def test_fusion_never_replays_a_chain_out_of_order(self) -> None:
        shared: dict[str, Artifact] = {}

        def pipe(run: RunContext) -> None:  # shares an artifact between branches (not recommended)
            k = compare("k", ["first", "second"])
            if k == "first":
                shared["a"] = run.run_stage(self.mesh, settings={"tag": "a"})
            else:
                run.run_stage(self.solve, inputs={"inp": shared["a"]}, settings={"tag": "b"})
                run.run_stage(self.mesh, settings={"tag": "a"})  # the same request, after its consumer

        # Branch "second" calls both, but the consumer first: fusing them there would replay
        # `solve` before `mesh` (planned steps [1, 0]); they stay separate tasks.
        plan = plan_dag(pipe, cache=self.cache)
        self.assertEqual([(g.stage, g.parents) for g in plan.groups], [("mesh", []), ("solve", [0])])

    def test_time_sums_and_bounded_labels(self) -> None:
        from poisson_runner.dag import format_time, group_label, time_seconds

        self.assertEqual([time_seconds(t) for t in ("90", "1:30", "01:00:00", "1-2", "1-02:03", "1-00:00:01")],
                         [5400, 90, 3600, 93600, 93780, 86401])
        self.assertEqual((format_time(3 * 3600), format_time(86400 + 5)), ("03:00:00", "1-00:00:05"))
        self.assertEqual(group_label(["a/b", "c|d"]), "a_b+c_d")
        self.assertEqual(group_label([f"stage{i}" for i in range(15)]), "stage0+13+stage14")
        self.assertLessEqual(len(group_label(["x" * 100])), 48)
        two = group_label(["import_mesh_with_screening", "assemble_stiffness_matrix"])  # 52 chars joined
        self.assertEqual(two, "import_mesh_with_screen+assemble_stiffness_matr")  # both ends kept
        self.assertEqual(group_label(["x" * 47, "y", "z"]), "x" * 22 + "+1+z")  # no "+0+", no trailing "+"
        for bad in ("00:99:00", "1-25:00:00", "1:75", "-1", "1h"):
            with self.assertRaises(ValueError, msg=bad):
                parse_resources(f"time={bad}")
        self.assertEqual(parse_resources("time=90")["time"], "90")  # plain minutes may exceed 59
        self.assertEqual(parse_resources("time=0")["time"], "0")  # Slurm: no limit

    # ------------------------------------------------------------------ CLI and submit.sh

    def test_cli_plan_and_task(self) -> None:
        pipe = self._pipeline()
        plan_file = self.tmp / "plan.json"

        def main(*argv: str) -> tuple[int, str, str]:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = experiment_main(pipe, argv=list(argv), cache=self.cache)
            return code, out.getvalue(), err.getvalue()

        code, out, _ = main("--dag-plan", str(plan_file), "--resources", "cpus=2", "--stage-resources",
                            "solve:cpus=8,time=02:00:00", "--select", "mesh=fine")
        self.assertEqual(code, 0)
        lines = [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("dag-group ")]
        self.assertEqual(lines, ["0|1|-|2|01:00:00|3900M|mesh|0", "1|2|0|2|01:00:00|3900M|assemble|0",
                                 "2|4|1|8|02:00:00|3900M|solve|1"])
        self.assertEqual(json.loads(plan_file.read_text())["selection"], {"mesh": ["fine"]})
        code, out, _ = main("--dag-plan", str(plan_file), "--dag-task", "0/0")
        self.assertEqual(code, 0)
        self.assertIn("stage 'mesh': executed", out)
        self.assertEqual(self._lines(self.mesh_log), ["fine"])
        self.assertEqual(main("--dag-task", "0/0")[0], 2)
        self.assertEqual(main("--no-fuse")[0], 2)  # a planning option without --dag-plan
        self.assertEqual(main("--dag-plan", str(plan_file), "--count")[0], 2)
        self.assertEqual(main("--dag-plan", str(plan_file), "--dag-task", "0/0", "--select", "mesh=fine")[0], 2)

    # ------------------------------------------------------------------ script defaults, split=True, threads

    def _main(self, pipe, *argv: str, **defaults) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = experiment_main(pipe, argv=list(argv), cache=self.cache, **defaults)
        return code, out.getvalue(), err.getvalue()

    @staticmethod
    def _groups(out: str) -> list[str]:
        return [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("dag-group ")]

    def test_script_defaults_and_command_line_overrides(self) -> None:
        pipe = self._pipeline()
        plan_file = str(self.tmp / "plan.json")
        defaults = dict(resources={"cpus": 2, "time": "00:20:00"}, stage_resources={"solve": {"cpus": 4}},
                        split_by="mesh", fuse=False)
        code, out, err = self._main(pipe, "--dag-plan", plan_file, **defaults)
        self.assertEqual(code, 0, err)
        groups = self._groups(out)
        self.assertEqual(len(groups), 6)  # split by mesh, unfused
        self.assertIn("0|1|-|2|00:20:00|3900M|mesh|0", groups)
        self.assertIn("2|4|1|4|00:20:00|3900M|solve|1", groups)
        self.assertEqual(json.loads(Path(plan_file).read_text())["split_by"], ["mesh"])

        # Flags win, key by key: --resources keeps the script's cpus; --no-split and --fuse undo the script.
        code, out, err = self._main(pipe, "--dag-plan", plan_file, "--resources", "time=00:10:00", "--no-split",
                                    "--fuse", "--stage-resources", "solve:mem-per-cpu=8G", **defaults)
        self.assertEqual(code, 0, err)
        self.assertEqual(self._groups(out), ["0|2|-|2|00:10:00|3900M|mesh|0", "1|4|0|2|00:10:00|3900M|assemble|0",
                                             "2|8|1|4|00:10:00|8G|solve|1"])
        code, out, _ = self._main(pipe, "--dag-plan", plan_file, "--split-by", "order", **defaults)
        self.assertEqual(json.loads(Path(plan_file).read_text())["split_by"], ["order"])  # replaces the script's

        # The script's shard_order is used by --shard unless --shard-order is given.
        code, out, _ = self._main(pipe, "--show-settings", shard_order="block", mode="dag", **defaults)
        self.assertEqual(code, 0)
        self.assertEqual([ln for ln in out.splitlines() if ln.startswith("setting ")], [
            "setting mode=dag", "setting cpus=2", "setting time=00:20:00", "setting mem_per_cpu=3900M",
            "setting shard_order=block"])
        code, out, _ = self._main(pipe, "--show-settings", "--resources", "cpus=8", "--shard-order", "stride")
        self.assertIn("setting cpus=8", out)
        self.assertIn("setting mode=shard", out)
        self.assertIn("setting shard_order=stride", out)
        self.assertEqual(self._main(pipe, "--show-settings", "--split-by", "mesh")[0], 2)
        self.assertEqual(self._main(pipe, "--resources", "cpus=2")[0], 2)  # needs --dag-plan or --show-settings

        # --show-settings never creates the cache.
        fresh = self.tmp / "never_created"
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(os.environ, {"POISSON_CACHE_DIR": str(fresh)}):
            self.assertEqual(experiment_main(pipe, argv=["--show-settings"]), 0)
        self.assertFalse(fresh.exists())

        for bad in (dict(mode="slurm"), dict(shard_order="random"), dict(resources={"gpus": 1}),
                    dict(stage_resources={"solve": {"cpus": 0}}), dict(max_fused_time="0"), dict(max_array_size=0),
                    dict(split_by=[""]), dict(fuse="yes")):
            with self.assertRaises(ValueError, msg=str(bad)):
                experiment_main(pipe, argv=["--show-settings"], **bad)
        with self.assertRaises(TypeError):
            experiment_main(pipe, argv=["--count"], shard=(0, 1))

    def test_stage_resources_precedence(self) -> None:
        # Python: per-call > stage_resources > resources. CLI --stage-resources beats per-call.
        fine = {"cpus": 16, "time": "04:00:00"}
        pipe = self._pipeline(fine_resources=fine)
        plan = plan_dag(pipe, cache=self.cache, resources={"cpus": 2}, stage_resources={"solve": {"cpus": 4}})
        by = {(g.stage, g.resources["cpus"]) for g in plan.groups}
        self.assertEqual(by, {("mesh", 2), ("assemble", 2), ("solve", 4), ("mesh", 16), ("assemble", 16), ("solve", 16)})
        plan = plan_dag(pipe, cache=self.cache, stage_resources={"solve": {"cpus": 4}},
                        stage_resource_overrides={"solve": {"time": "06:00:00"}})
        solves = [g.resources for g in plan.groups if g.stage == "solve"]
        self.assertEqual(sorted((r["cpus"], r["time"]) for r in solves), [(4, "06:00:00"), (16, "06:00:00")])
        plan = plan_dag(pipe, cache=self.cache, stage_resource_overrides={"solve": {"cpus": 8}})
        self.assertEqual({g.resources["cpus"] for g in plan.groups if g.stage == "solve"}, {8})

    def test_compare_split_true_and_its_overrides(self) -> None:
        def pipe(run: RunContext) -> None:
            m = compare("mesh", ["coarse", "fine"], split=True)
            msh = run.run_stage(self.mesh, settings={"tag": m})
            s = run.compare(Compare(["cg", "lu"], name="solver"))
            run.run_stage(self.solve, inputs={"inp": msh}, settings={"tag": m + s})

        plan = plan_dag(pipe, cache=self.cache)
        self.assertEqual(plan.split_by, ["mesh"])
        self.assertEqual([(g.stage, g.split) for g in plan.groups],
                         [("mesh", ("coarse",)), ("solve", ("coarse",)), ("mesh", ("fine",)), ("solve", ("fine",))])
        self.assertEqual(plan_dag(pipe, cache=self.cache, compare_splits=False).split_by, [])
        self.assertEqual(plan_dag(pipe, cache=self.cache, split_by="solver").split_by, ["solver", "mesh"])
        code, out, err = self._main(pipe, "--dag-plan", str(self.tmp / "p.json"), "--no-split")
        self.assertEqual((code, len(self._groups(out))), (0, 2), err)  # one mesh array, one solve array
        # split=True is ignored outside DAG mode, and keeps replay checks unchanged.
        self.assertEqual(len(run_experiment(pipe, cache=self.cache, dry_run=True).branches), 4)

        def both(run: RunContext) -> None:
            run.compare(Compare(["a", "b"], name="x"), split=True)

        with self.assertRaisesRegex(TypeError, "split=True to Compare"):
            run_experiment(both, cache=self.cache, dry_run=True, continue_on_error=False)
        with self.assertRaises(TypeError):
            Compare(["a"], name="x", split="yes")

    def test_threads_set_env_key_and_cpus_floor(self) -> None:
        def pipe_for(threads=None, resources=None, env=None):
            def pipe(run: RunContext) -> None:
                t = compare("threads", [1, 4])
                run.run_stage(self.solve, settings={"tag": "same"}, threads=t if threads is None else threads, resources=resources, env=env)
            return pipe

        plan = plan_dag(pipe_for(), cache=self.cache)
        self.assertEqual(len(plan.tasks), 2)  # thread count is part of the request key
        self.assertEqual([t.resources["cpus"] for t in plan.tasks], [1, 4])  # raised to the thread count
        plan = plan_dag(pipe_for(), cache=self.cache, stage_resources={"solve": {"cpus": 16}})
        self.assertEqual([t.resources["cpus"] for t in plan.tasks], [16, 16])  # a fixed allocation stays
        with self.assertRaisesRegex(ValueError, "--stage-resources"):
            plan_dag(pipe_for(), cache=self.cache, stage_resource_overrides={"solve": {"cpus": 2}})
        with self.assertRaisesRegex(ValueError, "at least as many cpus"):
            run_experiment(pipe_for(resources={"cpus": 2}), cache=self.cache, dry_run=True, continue_on_error=False)
        for env in ({"OMP_NUM_THREADS": "2"}, {"OMP_NUM_THREADS": "4,2"}):
            with self.assertRaisesRegex(ValueError, "contradicts"):
                run_experiment(pipe_for(threads=4, env=env), cache=self.cache, dry_run=True, continue_on_error=False)

        def raw(run: RunContext) -> None:  # a raw env thread count isn't a floor: the desync is refused
            run.run_stage(self.solve, settings={"tag": "raw"}, env={"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})

        with self.assertRaisesRegex(ValueError, "default resources"):
            plan_dag(raw, cache=self.cache)
        self.assertEqual(plan_dag(raw, cache=self.cache, resources={"cpus": 8}).tasks[0].resources["cpus"], 8)
        for bad in (0, True, "4"):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_experiment(pipe_for(threads=bad), cache=self.cache, dry_run=True, continue_on_error=False)

        # Running: the env reaches the stage; too many threads for the CPUs is a stored warning.
        def once(run: RunContext) -> Artifact:
            return run.run_stage(self.solve, settings={"tag": "warn"}, threads=3)

        with mock.patch.object(executor_module, "_usable_cpu_count", return_value=2):
            report = run_experiment(once, cache=self.cache, continue_on_error=False)
        trace = report.branches[0].stages[0]
        self.assertTrue(any("requested 3 threads on 2 usable CPUs" in w for w in trace.warnings), trace.warnings)
        # An inherited thread count above the CPUs warns too (from the executor's host metadata).
        def inherited(run: RunContext) -> Artifact:
            return run.run_stage(self.solve, settings={"tag": "inherited"})

        with mock.patch.object(executor_module, "_usable_cpu_count", return_value=2), \
                mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}):
            report = run_experiment(inherited, cache=self.cache, continue_on_error=False)
        self.assertTrue(any("4 threads on 2" in w for w in report.branches[0].stages[0].warnings))
        cached = run_experiment(once, cache=self.cache, continue_on_error=False).branches[0].stages[0]
        self.assertTrue(cached.cached)
        self.assertTrue(any("3 threads on 2 usable CPUs" in w for w in cached.warnings), cached.warnings)
        self.assertEqual(cached.execution["environment"]["threading"]["OMP_NUM_THREADS"], "3")

    def test_shared_request_resources_do_not_depend_on_branch_order(self) -> None:
        # The same request in two branches; only one call gives resources, in either order.
        for first in (0, 1):
            def pipe(run: RunContext) -> None:
                k = compare("k", [0, 1])
                res = {"cpus": 4} if k != first else None
                run.run_stage(self.solve, settings={"tag": "shared"}, env={"OMP_NUM_THREADS": "4"}, resources=res)

            plan = plan_dag(pipe, cache=self.cache)
            self.assertEqual([t.resources["cpus"] for t in plan.tasks], [4], first)
        # A threads= floor survives a later equivalent call that only gives a time.
        def floor(run: RunContext) -> None:
            k = compare("k", [0, 1])
            if k == 0:
                run.run_stage(self.solve, settings={"tag": "floor"}, threads=4)
            else:
                run.run_stage(self.solve, settings={"tag": "floor"}, resources={"time": "00:10:00"},
                              env={"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

        plan = plan_dag(floor, cache=self.cache)
        self.assertEqual([(t.resources["cpus"], t.resources["time"]) for t in plan.tasks], [(4, "00:10:00")])

        # Calls of one request disagreeing on cpus are refused, unless --stage-resources sets cpus.
        def conflict(run: RunContext) -> None:
            k = compare("k", [8, 16])
            run.run_stage(self.solve, settings={"tag": "conflict"}, resources={"cpus": k})

        with self.assertRaisesRegex(ValueError, "same resources"):
            plan_dag(conflict, cache=self.cache)
        plan = plan_dag(conflict, cache=self.cache, stage_resource_overrides={"solve": {"cpus": 12}})
        self.assertEqual([t.resources["cpus"] for t in plan.tasks], [12])

    def test_split_names_missing_from_the_selection_get_no_label(self) -> None:
        def pipe(run: RunContext) -> None:
            m = compare("mesh", ["coarse", "fine"])
            tag = m
            if m == "fine":
                tag += compare("solver", ["cg", "lu"])
            run.run_stage(self.solve, settings={"tag": tag})

        plan = plan_dag(pipe, cache=self.cache, split_by=["mesh", "solver"], select={"mesh": "coarse"})
        self.assertEqual([g.split for g in plan.groups], [("coarse", None)])
        with self.assertRaisesRegex(ValueError, "no compare"):
            plan_dag(pipe, cache=self.cache, split_by="solvers", select={"mesh": "coarse"})

    def test_plan_v3_round_trip_with_splits_and_threads(self) -> None:
        env_bin = _write_executable(self.tmp / "envsolve_bin", """
import argparse, os, pathlib
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--in", dest="inp", default=None)
p.add_argument("--out", required=True)
p.add_argument("--info-out", required=True)
a = p.parse_args()
seen = ",".join(os.environ.get(k, "-") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
pathlib.Path(a.out).write_text(a.tag + ":" + seen)
pathlib.Path(a.info_out).write_text("{}")
""")
        solve = Stage.from_cli("solve", binary=env_bin, input_flags={"inp": "--in"}, setting_flags={"tag": "--tag"},
                               output_flags={"out": "--out"}, output_specs={"out": ".txt"}, info_flag="--info-out")

        def pipe_for(threads: int):
            def pipe(run: RunContext) -> dict[str, Artifact]:
                m = compare("mesh", ["coarse", "fine"], split=True)
                o = compare("order", [1, 2])
                msh = run.run_stage(self.mesh, settings={"tag": m})
                return {"sol": run.run_stage(solve, inputs={"inp": msh}, settings={"tag": f"{m}{o}"},
                                             threads=threads)}
            return pipe

        pipe = pipe_for(2)
        plan_file = self.tmp / "v3.json"
        code, out, err = self._main(pipe, "--dag-plan", str(plan_file), "--split-by", "mesh", "--split-by", "order")
        self.assertEqual(code, 0, err)
        payload = json.loads(plan_file.read_text())
        self.assertEqual((payload["format_version"], payload["split_by"]), (3, ["mesh", "order"]))
        self.assertIn(["fine", "2"], [t["split"] for t in payload["tasks"]])
        self.assertTrue(all(t["resources"]["cpus"] == 2 for t in payload["tasks"] if t["stage"] == "solve"))

        # Planned with threads=2, replayed with threads=4: the guard refuses the task.
        solve_group = next(g["id"] for g in payload["groups"] if g["stages"] == ["solve"])
        mesh_group = payload["groups"][0]["id"]
        self.assertEqual(self._main(pipe, "--dag-plan", str(plan_file), "--dag-task", f"{mesh_group}/0")[0], 0)
        with self.assertRaisesRegex(RuntimeError, "Plan mismatch"):
            run_dag_task(pipe_for(4), payload, solve_group, 0, cache=self.cache)

        for g in payload["groups"]:
            for e in range(len(g["units"])):
                code, _, err = self._main(pipe, "--dag-plan", str(plan_file), "--dag-task", f"{g['id']}/{e}")
                self.assertEqual(code, 0, err)
        code, _, err = self._main(pipe, "--collect", "--out", str(self.tmp / "res"))
        self.assertEqual(code, 0, err)
        report = run_experiment(pipe, cache=self.cache, force_cache_mode="read_only", require_cached=True)
        solves = [s for b in report.branches for s in b.stages if s.stage == "solve"]
        self.assertEqual(len(solves), 4)
        for s in solves:  # the stage process itself saw the planned thread count
            self.assertTrue(s.cached)
            self.assertTrue(Path(s.outputs["out"].path).read_text().endswith(":2,2,2"))

        old = dict(payload, format_version=2)
        with self.assertRaisesRegex(ValueError, "format version 3"):
            run_dag_task(pipe, old, 0, 0, cache=self.cache)

    def test_submit_sh_dag_dependencies_and_collect_on_leaves(self) -> None:
        fake_bin = self.tmp / "fake_bin"
        fake_bin.mkdir()
        sbatch_log = self.tmp / "sbatch.log"
        _write_executable(
            fake_bin / "sbatch",
            f"""
import pathlib, sys
with open({str(sbatch_log)!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
print(f"{{2000 + len(pathlib.Path({str(sbatch_log)!r}).read_text().splitlines())}};delftblue")
""",
        )
        exp = self.tmp / "exp.py"
        exp.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(REPO_ROOT / "tests")!r})
                from pathlib import Path
                from poisson_runner import RunContext, Stage, compare, experiment_main
                BIN = Path({str(self.tmp)!r})
                def stage(name):
                    return Stage.from_cli(name, binary=BIN / f"{{name}}_bin", input_flags={{"inp": "--in"}},
                        setting_flags={{"tag": "--tag"}}, output_flags={{"out": "--out"}},
                        output_specs={{"out": ".txt"}}, info_flag="--info-out")
                def pipe(run: RunContext):
                    m = compare("mesh", ["coarse", "fine"])
                    msh = run.run_stage(stage("mesh"), settings={{"tag": m}})
                    s = compare("solver", ["cg", "lu"])
                    run.run_stage(stage("solve"), inputs={{"inp": msh}}, settings={{"tag": m + s}})
                if __name__ == "__main__":
                    raise SystemExit(experiment_main(pipe))
                """
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "POISSON_CACHE_DIR": str(self.tmp / "sub_cache"),
            "POISSON_REPO": str(REPO_ROOT),
            "PYTHONPATH": str(REPO_ROOT / "python"),
        }
        submit_sh = str(REPO_ROOT / "scripts/delftblue/submit.sh")
        r = subprocess.run(["bash", submit_sh, str(exp), "--dag", "--split-by", "mesh", "--cpus", "2",
                            "--stage-resources", "solve:cpus=8", "--name", "dagtest"],
                           env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        lines = sbatch_log.read_text().splitlines()
        self.assertEqual(len(lines), 5)  # mesh+solve per mesh, then collect
        self.assertNotIn("--dependency", lines[0])
        self.assertIn("--dependency=afterok:2001", lines[1])
        self.assertIn("--array=0-1", lines[1])
        self.assertIn("--cpus-per-task=8", lines[1])
        self.assertIn("--cpus-per-task=2", lines[0])
        self.assertIn("--dependency=afterok:2003", lines[3])
        self.assertIn("--dependency=afterok:2002:2004", lines[4])  # collect waits on the leaf arrays only
        self.assertIn("--collect", lines[4])
        self.assertIn("--dag ", lines[0])

        # Per-group submissions without collect, then one collect after both.
        sbatch_log.unlink()
        r = subprocess.run(["bash", submit_sh, str(exp), "--select", "mesh=fine", "--shard-order", "block",
                            "--no-collect", "--print-id"], env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "2001")
        lines = sbatch_log.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("--array=0-1", lines[0])  # 2 selected branches
        self.assertIn("--shard-order block --select mesh=fine", lines[0])
        r = subprocess.run(["bash", submit_sh, str(exp), "--collect-only", "--after", "11,12"],
                           env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        last = sbatch_log.read_text().splitlines()[-1]
        self.assertIn("--dependency=afterok:11:12", last)
        self.assertIn("--collect", last)
        self.assertNotIn("--select", last)

        # A DAG submission without collect prints its leaf arrays in the form --after takes.
        r = subprocess.run(["bash", submit_sh, str(exp), "--dag", "--split-by", "mesh", "--no-collect", "--print-id"],
                           env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "2004,2006")
        # --print-only shows the collect command with normal spacing (no IFS leaking into it).
        r = subprocess.run(["bash", submit_sh, str(exp), "--collect-only", "--after", "2004,2006", "--print-only"],
                           env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sbatch --parsable --ntasks=1 ", r.stderr)
        self.assertIn(" --dependency=afterok:2004:2006 ", r.stderr)

        for bad in (["--dag", "--warmup"], ["--dag", "--pieces", "2"], ["--split-by", "mesh"],
                    ["--after", "11"], ["--collect-only", "--select", "mesh=fine"], ["--collect-only", "--after", "x"],
                    ["--collect-only", "--after", ","], ["--collect-only", "--after", "1 2"],
                    ["--dag", "--shard-order", "block"], ["--no-fuse"]):
            r = subprocess.run(["bash", submit_sh, str(exp), *bad], env=env, capture_output=True, text=True, check=False)
            self.assertNotEqual(r.returncode, 0, bad)


    def test_submit_sh_follows_the_scripts_mode_and_resources(self) -> None:
        fake_bin = self.tmp / "fake_bin"
        fake_bin.mkdir()
        sbatch_log = self.tmp / "sbatch.log"
        _write_executable(
            fake_bin / "sbatch",
            f"""
import pathlib, sys
with open({str(sbatch_log)!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
print(f"{{3000 + len(pathlib.Path({str(sbatch_log)!r}).read_text().splitlines())}};delftblue")
""",
        )
        exp = self.tmp / "dag_exp.py"
        exp.write_text(
            textwrap.dedent(
                f"""
                from pathlib import Path
                from poisson_runner import RunContext, Stage, compare, experiment_main
                BIN = Path({str(self.tmp)!r})
                def stage(name):
                    return Stage.from_cli(name, binary=BIN / f"{{name}}_bin", input_flags={{"inp": "--in"}},
                        setting_flags={{"tag": "--tag"}}, output_flags={{"out": "--out"}},
                        output_specs={{"out": ".txt"}}, info_flag="--info-out")
                def pipe(run: RunContext):
                    m = compare("mesh", ["coarse", "fine"], split=True)
                    msh = run.run_stage(stage("mesh"), settings={{"tag": m}})
                    s = compare("solver", ["cg", "lu"])
                    run.run_stage(stage("solve"), inputs={{"inp": msh}}, settings={{"tag": m + s}})
                if __name__ == "__main__":
                    raise SystemExit(experiment_main(pipe, mode="dag", resources={{"cpus": 3, "time": "00:05:00"}},
                                                     stage_resources={{"solve": {{"mem_per_cpu": "2G"}}}}))
                """
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "POISSON_CACHE_DIR": str(self.tmp / "sub_cache"),
            "POISSON_REPO": str(REPO_ROOT),
            "PYTHONPATH": str(REPO_ROOT / "python"),
        }
        submit_sh = str(REPO_ROOT / "scripts/delftblue/submit.sh")

        def submit(*args: str) -> subprocess.CompletedProcess[str]:
            if sbatch_log.exists():
                sbatch_log.unlink()
            return subprocess.run(["bash", submit_sh, str(exp), *args], env=env, capture_output=True, text=True,
                                  check=False)

        _write_executable(fake_bin / "srun", "")  # a Slurm host either way; SLURM_JOB_ID decides the warning
        env["SLURM_JOB_ID"] = "1"

        # No --dag: the script's mode, resources and split=True; --stage-resources needs no --dag here.
        r = submit("--stage-resources", "solve:cpus=8", "--name", "dagdef")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        lines = sbatch_log.read_text().splitlines()
        self.assertEqual(len(lines), 5)  # mesh + solve per mesh (split), then collect
        self.assertIn("--dag ", lines[0])
        self.assertIn("--cpus-per-task=3", lines[0])
        self.assertIn("--time=00:05:00", lines[0])
        self.assertIn("--cpus-per-task=8", lines[1])
        self.assertIn("--mem-per-cpu=2G", lines[1])
        self.assertIn("--mem-per-cpu=3900M", lines[4])  # the collect job keeps its own default
        self.assertNotIn("warning", r.stderr)

        # --shards runs it as pieces with the script's resources; flags given here win.
        r = submit("--shards", "--time", "00:07:00")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        lines = sbatch_log.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("--array=0-3", lines[0])
        self.assertIn("--cpus-per-task=3", lines[0])
        self.assertIn("--time=00:07:00", lines[0])
        self.assertIn("--shard-order stride", lines[0])
        self.assertIn("--shards --pieces 4 --shard-order stride", r.stdout)  # the rerun hint keeps --shards
        r = submit("--shards", "--mem-per-cpu", "1000M", "--print-only")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stderr.count("--mem-per-cpu=1000M"), 2)  # pieces and collect

        r = submit("--collect-only", "--after", ",5,,6")  # mode-independent; empty ids (all cached) skipped
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--dependency=afterok:5:6", sbatch_log.read_text())
        r = submit("--shards", "--pieces", "4", "--only", "1,3", "--print-only")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--array=1,3", r.stderr)

        # Planning a DAG outside a Slurm job where srun exists warns about login nodes.
        del env["SLURM_JOB_ID"]
        r = submit("--print-only")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("outside a Slurm job", r.stderr)

        for bad in (["--dag", "--shards"], ["--warmup"], ["--pieces", "2"], ["--shard-order", "block"],
                    ["--shards", "--split-by", "mesh"], ["--split-by", "mesh", "--no-split"],
                    ["--collect-only", "--dag"], ["--collect-only", "--no-fuse"], ["--no-fuse", "--fuse"],
                    ["--shards", "--pieces", "4", "--only", "4"], ["--shards", "--pieces", "4", "--only", "2-7"],
                    ["--shards", "--pieces", "9", "--only", "1"], ["--shards", "--pieces", "4", "--only", "x"],
                    ["--collect-only", "--after", ",,"], ["--shards", "--pieces", "4", "--only", "3-1"],
                    ["--shards", "--pieces", "4", "--only", "18446744073709551616"]):
            r = submit(*bad)
            self.assertNotEqual(r.returncode, 0, bad)


if __name__ == "__main__":
    unittest.main()
