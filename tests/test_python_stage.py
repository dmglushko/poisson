"""Tests for `@python_stage`: cached pipeline stages backed by plain Python functions."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import Cache, compare, load_value, python_stage, run_experiment  # noqa: E402

_COUNTER = 0


def _plain(inputs):
    return 1


class PythonStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="poisson_python_stage_test_")
        self.tmp = Path(self._tmp.name)
        self.cache = Cache(self.tmp / "cache")
        self.calls = self.tmp / "calls.log"
        self.data = self.tmp / "data.txt"
        self.data.write_text("a\nb\nc\n", encoding="utf-8")
        self.pkg_root = self.tmp / "src"
        self.pkg_root.mkdir()
        sys.path.insert(0, str(self.pkg_root))

    def tearDown(self) -> None:
        sys.path.remove(str(self.pkg_root))
        for name in [m for m in sys.modules if m.startswith("pstest_")]:
            del sys.modules[name]
        self._tmp.cleanup()

    # ------------------------------------------------------------------ helpers

    def make_package(self, source: str, *, extra: dict[str, str] | None = None):
        """Write a package `pstest_<n>` with `stages.py` = `source`, import and return that module."""
        global _COUNTER
        _COUNTER += 1
        name = f"pstest_{_COUNTER}"
        pkg = self.pkg_root / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        header = f"from pathlib import Path\nCALLS = Path({str(self.calls)!r})\n"
        (pkg / "stages.py").write_text(header + textwrap.dedent(source), encoding="utf-8")
        for fname, text in (extra or {}).items():
            (pkg / fname).write_text(text, encoding="utf-8")
        importlib.invalidate_caches()
        return importlib.import_module(f"{name}.stages"), pkg

    def n_calls(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def run_one(self, fn, **kwargs):
        report = run_experiment(lambda run: fn(run), cache=self.cache, continue_on_error=True, **kwargs)
        self.assertEqual(len(report.branches), 1)
        return report, report.branches[0]

    # ------------------------------------------------------------------ value stages

    def test_value_stage_is_cached_and_reports_info(self) -> None:
        mod, _ = self.make_package(
            """
            import warnings
            from poisson_runner import python_stage

            @python_stage
            def count_lines(inputs, *, scale, info):
                with CALLS.open("a") as f:
                    f.write("x\\n")
                n = len(inputs["text"].read_text().splitlines())
                info["n"] = n
                warnings.warn("careful")
                return {"lines": n * scale}
            """
        )

        def pipeline(run):
            art = mod.count_lines(run, inputs={"text": run.raw_file(self.data)}, scale=2)
            return {"value": load_value(art)}

        _, b1 = self.run_one(pipeline)
        self.assertTrue(b1.ok, b1.error)
        self.assertEqual(b1.outputs["value"], {"lines": 6})
        trace = b1.stages[0]
        self.assertFalse(trace.cached)
        self.assertEqual(trace.info["n"], 3)
        self.assertIsInstance(trace.info["time"], float)
        self.assertEqual(trace.warnings, ["UserWarning: careful"])

        _, b2 = self.run_one(pipeline)
        self.assertTrue(b2.stages[0].cached)
        self.assertEqual(b2.outputs["value"], {"lines": 6})
        self.assertEqual(self.n_calls(), 1)

    def test_settings_inputs_and_code_change_the_key(self) -> None:
        mod, pkg = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def f(inputs, *, k=1):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return [k, type(k).__name__]
            """,
            extra={"helpers.py": "VALUE = 1\n"},
        )

        def with_k(k):
            return lambda run: {"v": load_value(mod.f(run, inputs={"text": run.raw_file(self.data)}, k=k))}

        _, b = self.run_one(with_k(1))
        self.assertEqual(b.outputs["v"], [1, "int"])
        self.run_one(with_k(1))
        self.assertEqual(self.n_calls(), 1)
        _, b = self.run_one(with_k(1.0))  # the function can tell 1.0 from 1, so it is a new entry
        self.assertEqual(b.outputs["v"], [1.0, "float"])
        self.assertEqual(self.n_calls(), 2)

        self.data.write_text("changed\n", encoding="utf-8")
        self.run_one(with_k(1))
        self.assertEqual(self.n_calls(), 3)

        (pkg / "helpers.py").write_text("VALUE = 2\n", encoding="utf-8")  # any file of the package
        self.run_one(with_k(1))
        self.assertEqual(self.n_calls(), 4)

    def test_hash_package_false_tracks_only_the_module(self) -> None:
        mod, pkg = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage(hash_package=False)
            def f(inputs):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return 1
            """,
            extra={"helpers.py": "VALUE = 1\n"},
        )
        self.run_one(lambda run: mod.f(run))
        (pkg / "helpers.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.run_one(lambda run: mod.f(run))
        self.assertEqual(self.n_calls(), 1)

    def test_bad_return_values_fail_the_branch(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def nan(inputs):
                return float("nan")

            @python_stage
            def a_set(inputs):
                return {1, 2}

            @python_stage
            def int_keys(inputs):
                return {1: "a"}
            """
        )
        for fn in (mod.nan, mod.a_set, mod.int_keys):
            _, branch = self.run_one(lambda run, fn=fn: fn(run))
            self.assertFalse(branch.ok, fn.__name__)

    def test_dry_run_placeholder_value_cannot_be_read(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def f(inputs):
                return 1
            """
        )
        _, branch = self.run_one(lambda run: load_value(mod.f(run)), dry_run=True)
        self.assertFalse(branch.ok)
        self.assertIn("dry-run placeholder", branch.error)

    # ------------------------------------------------------------------ file stages

    def test_file_stage_writes_declared_outputs(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage(outputs={"upper": ".txt", "count": ".txt"})
            def shout(inputs, outputs, *, times):
                text = inputs["text"].read_text()
                outputs["upper"].write_text(text.upper() * times)
                outputs["count"].write_text(str(len(text)))

            @python_stage(outputs={"out": ".txt"})
            def returns_value(inputs, outputs):
                outputs["out"].write_text("x")
                return 1

            @python_stage(outputs={"out": ".txt"})
            def writes_nothing(inputs, outputs):
                pass

            @python_stage(outputs={"out": "txt"})
            def single(inputs, outputs):
                outputs["out"].write_text("one")
            """
        )
        _, branch = self.run_one(lambda run: mod.shout(run, inputs={"text": run.raw_file(self.data)}, times=2))
        self.assertTrue(branch.ok, branch.error)
        outs = branch.result
        self.assertEqual(set(outs), {"upper", "count"})
        self.assertEqual(outs["upper"].path.read_text(), "A\nB\nC\n" * 2)
        self.assertEqual(outs["upper"].path.suffix, ".txt")

        _, branch = self.run_one(lambda run: mod.returns_value(run))
        self.assertFalse(branch.ok)
        self.assertIn("must return None", branch.error)
        _, branch = self.run_one(lambda run: mod.writes_nothing(run))
        self.assertFalse(branch.ok)
        self.assertIn("did not write", branch.error)
        _, branch = self.run_one(lambda run: mod.single(run))  # one output named "out": a single Artifact
        self.assertEqual(branch.result.path.read_text(), "one")
        self.assertEqual(branch.result.path.suffix, ".txt")

    def test_list_inputs_arrive_as_lists_of_paths(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def total(inputs):
                return sum(len(p.read_text()) for p in inputs["files"])
            """
        )
        other = self.tmp / "other.txt"
        other.write_text("12", encoding="utf-8")
        _, branch = self.run_one(
            lambda run: load_value(mod.total(run, inputs={"files": [run.raw_file(self.data), run.raw_file(other)]}))
        )
        self.assertEqual(branch.result, 8)

    # ------------------------------------------------------------------ refusals

    def test_definition_time_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "top level"):
            python_stage(lambda inputs: 1)

        def outer():
            def inner(inputs):
                return 1

            return inner

        with self.assertRaisesRegex(ValueError, "top level"):
            python_stage(outer())
        with self.assertRaisesRegex(ValueError, "outputs must name"):
            self.make_package(
                """
                from poisson_runner import python_stage

                @python_stage(outputs={})
                def f(inputs, outputs):
                    pass
                """
            )
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.make_package(
                """
                from poisson_runner import python_stage

                @python_stage
                def f(inputs, *, cache):
                    return 1
                """
            )
        with self.assertRaisesRegex(TypeError, "must take"):
            self.make_package(
                """
                from poisson_runner import python_stage

                @python_stage(outputs={"out": ".txt"})
                def f(inputs):
                    pass
                """
            )
        with self.assertRaisesRegex(ValueError, "extension"):
            self.make_package(
                """
                from poisson_runner import python_stage

                @python_stage(outputs={"out": ".a/b"})
                def f(inputs, outputs):
                    pass
                """
            )
        with self.assertRaisesRegex(TypeError, "not a single path"):
            python_stage(_plain, code_files=str(self.data))
        with self.assertRaisesRegex(ValueError, "wrapper"):
            self.make_package(
                """
                from poisson_runner import python_stage
                from .deco import logged

                @python_stage
                @logged
                def f(inputs):
                    return 1
                """,
                extra={
                    "deco.py": (
                        "import functools\n\n"
                        "def logged(fn):\n"
                        "    @functools.wraps(fn)\n"
                        "    def wrapper(*args, **kwargs):\n"
                        "        return fn(*args, **kwargs)\n"
                        "    return wrapper\n"
                    )
                },
            )

    def test_call_time_refusals(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def f(inputs, *, k, info):
                return k

            @python_stage(packages=["definitely-not-installed-poisson-xyz"])
            def needs_pkg(inputs):
                return 1
            """
        )
        cases = {
            "pass files through inputs": lambda run: mod.f(run, k=self.data),
            "plain JSON": lambda run: mod.f(run, k={1, 2}),
            "settings don't match": lambda run: mod.f(run, k=1, other=2),
            "filled by the stage": lambda run: mod.f(run, k=1, info={}),
            "not installed": lambda run: mod.needs_pkg(run),
        }
        for message, fn in cases.items():
            _, branch = self.run_one(fn)
            self.assertFalse(branch.ok, message)
            self.assertIn(message, branch.error)

    # ------------------------------------------------------------------ what the key tracks

    def test_two_wrappers_of_one_function_dont_share_a_stage(self) -> None:
        mod, _ = self.make_package(
            """
            def f(inputs):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return 1
            """
        )
        a = python_stage(mod.f, version="a")
        b = python_stage(mod.f, version="b")
        report, branch = self.run_one(lambda run: [a(run), b(run), python_stage(mod.f, version="a")(run)])
        self.assertTrue(branch.ok, branch.error)
        keys = [t.request_key for t in branch.stages]
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(keys[0], keys[2])  # an identical wrapper reuses the first one's entry
        self.assertEqual(self.n_calls(), 2)

    def test_code_files_and_version_change_the_key(self) -> None:
        extra_file = self.tmp / "extra.dat"
        extra_file.write_text("1", encoding="utf-8")
        extra_dir = self.tmp / "extra_dir"
        extra_dir.mkdir()
        (extra_dir / "a.py").write_text("A = 1\n", encoding="utf-8")
        mod, _ = self.make_package(
            """
            def f(inputs):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return 1
            """
        )
        stage = python_stage(mod.f, code_files=[extra_file, extra_dir])
        self.run_one(lambda run: stage(run))
        self.run_one(lambda run: stage(run))
        self.assertEqual(self.n_calls(), 1)
        extra_file.write_text("2", encoding="utf-8")
        self.run_one(lambda run: stage(run))
        self.assertEqual(self.n_calls(), 2)
        (extra_dir / "b.py").write_text("B = 1\n", encoding="utf-8")  # directories are rescanned per experiment
        self.run_one(lambda run: stage(run))
        self.assertEqual(self.n_calls(), 3)
        self.run_one(lambda run: python_stage(mod.f, code_files=[extra_file, extra_dir], version="2")(run))
        self.assertEqual(self.n_calls(), 4)

    def test_stale_bytecode_of_tracked_code_is_never_used(self) -> None:
        mod, pkg = self.make_package(
            """
            from poisson_runner import python_stage
            from . import helpers

            @python_stage
            def f(inputs):
                return helpers.VALUE
            """,
            extra={"helpers.py": "VALUE = 1\n"},
        )  # importing it here wrote helpers' .pyc
        helper = pkg / "helpers.py"
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertEqual(b.result, 1)
        st = helper.stat()
        helper.write_text("VALUE = 2\n", encoding="utf-8")  # same size, and the old timestamp:
        os.utime(helper, ns=(st.st_atime_ns, st.st_mtime_ns))  # the old .pyc still looks valid
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertFalse(b.stages[0].cached)
        self.assertEqual(b.result, 2)

    def test_package_files_are_rescanned_per_experiment(self) -> None:
        mod, pkg = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage
            def f(inputs):
                try:
                    from . import later
                except ImportError:
                    return "absent"
                return later.VALUE
            """
        )
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertEqual(b.result, "absent")
        (pkg / "later.py").write_text("VALUE = 'present'\n", encoding="utf-8")
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertEqual(b.result, "present")

    def test_namespace_package_hashes_all_its_files(self) -> None:
        pkg = self.pkg_root / "pstest_ns"
        pkg.mkdir()  # no __init__.py
        (pkg / "helper.py").write_text("X = 1\n", encoding="utf-8")
        (pkg / "stages.py").write_text(
            "from poisson_runner import python_stage\n"
            "from pstest_ns import helper\n\n"
            "@python_stage\n"
            "def f(inputs):\n"
            "    return helper.X\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        mod = importlib.import_module("pstest_ns.stages")
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertEqual(b.result, 1)
        (pkg / "helper.py").write_text("X = 22\n", encoding="utf-8")
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertEqual(b.result, 22)

    def test_moving_the_package_keeps_its_entries(self) -> None:
        source = """
            from poisson_runner import python_stage

            @python_stage
            def f(inputs):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return 1
            """
        mod, pkg = self.make_package(source)
        name = pkg.name
        _, b1 = self.run_one(lambda run: mod.f(run))
        other_root = self.tmp / "elsewhere" / "deeper"
        shutil.copytree(pkg, other_root / name)
        for m in [m for m in sys.modules if m.startswith(name)]:
            del sys.modules[m]
        sys.path.remove(str(self.pkg_root))
        sys.path.insert(0, str(other_root))
        try:
            importlib.invalidate_caches()
            moved = importlib.import_module(f"{name}.stages")
            self.assertEqual(Path(moved.__file__).parent.parent, other_root)
            _, b2 = self.run_one(lambda run: moved.f(run))
        finally:
            sys.path.remove(str(other_root))
            sys.path.insert(0, str(self.pkg_root))
        self.assertTrue(b2.stages[0].cached)
        self.assertEqual(b1.stages[0].request_key, b2.stages[0].request_key)

    def test_packages_are_probed_with_the_stage_sys_path(self) -> None:
        mod, _ = self.make_package(
            """
            from poisson_runner import python_stage

            @python_stage(packages=["pstest-localdist"])
            def f(inputs):
                import importlib.metadata
                return importlib.metadata.version("pstest-localdist")
            """
        )
        # A distribution that only the stage's import root provides (not the startup sys.path).
        dist = self.pkg_root / "pstest_localdist-2.5.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: pstest-localdist\nVersion: 2.5\n")
        _, b = self.run_one(lambda run: load_value(mod.f(run)))
        self.assertTrue(b.ok, b.error)
        self.assertEqual(b.result, "2.5")

    def test_env_values_are_converted_to_str(self) -> None:
        mod, _ = self.make_package(
            """
            import os
            from poisson_runner import python_stage

            @python_stage
            def f(inputs):
                return os.environ["PSTEST_VALUE"]
            """
        )
        _, b = self.run_one(lambda run: load_value(mod.f(run, env={"PSTEST_VALUE": 5})))
        self.assertTrue(b.ok, b.error)
        self.assertEqual(b.result, "5")

    def test_strict_interpreter_separates_environments(self) -> None:
        import venv

        venv_dir = self.tmp / "venv"
        venv.create(venv_dir, symlinks=True, with_pip=False)
        other_py = venv_dir / "bin" / "python"
        mod, _ = self.make_package(
            """
            def f(inputs):
                with CALLS.open("a") as fh:
                    fh.write("x\\n")
                return 1
            """
        )
        loose = python_stage(mod.f)
        strict = python_stage(mod.f, strict_interpreter=True)
        _, b = self.run_one(lambda run: [loose(run), loose(run, python_executable=other_py)])
        self.assertTrue(b.ok, b.error)
        self.assertEqual(self.n_calls(), 1)  # same Python version, no declared packages: one entry
        _, b = self.run_one(lambda run: [strict(run), strict(run, python_executable=other_py)])
        self.assertTrue(b.ok, b.error)
        self.assertEqual(self.n_calls(), 3)  # different sys.prefix: separate entries

    # ------------------------------------------------------------------ warnings

    def test_warnings_are_deduplicated_and_merged_with_runner_warnings(self) -> None:
        mod, _ = self.make_package(
            """
            import time
            import warnings
            from poisson_runner import python_stage

            @python_stage
            def f(inputs, *, info):
                info["warnings"] = "from info"
                for _ in range(3):
                    warnings.warn("loop")
                warnings.warn("old", DeprecationWarning)  # hidden by Python's default filters
                return time.perf_counter_ns()  # nondeterministic
            """
        )
        _, b = self.run_one(lambda run: mod.f(run, repeats=2))
        warnings_ = b.stages[0].warnings
        self.assertEqual(sum("loop" in w for w in warnings_), 1)
        self.assertIn("from info", warnings_)
        self.assertFalse(any("old" in w for w in warnings_))
        self.assertTrue(any("different output bytes" in w for w in warnings_), warnings_)

    # ------------------------------------------------------------------ experiment script

    def test_function_defined_in_the_experiment_script(self) -> None:
        script = self.tmp / "exp" / "sweep.py"
        script.parent.mkdir()
        (script.parent / "helper.py").write_text("FACTOR = 10\n", encoding="utf-8")
        script.write_text(
            textwrap.dedent(
                f"""
                from pathlib import Path
                from helper import FACTOR
                from poisson_runner import Cache, compare, experiment_main, load_value, python_stage

                CALLS = Path({str(self.calls)!r})

                @python_stage
                def scaled(inputs, *, k):
                    with CALLS.open("a") as fh:
                        fh.write("x\\n")
                    return k * FACTOR

                def pipeline(run):
                    k = compare("k", [1, 2])
                    art = scaled(run, k=k)
                    return {{"v": None if run.dry_run else load_value(art)}}

                # Unguarded on purpose: the stage process must not start the experiment again.
                raise SystemExit(experiment_main(pipeline, cache=Cache({str(self.tmp / "cache")!r})))
                """
            ),
            encoding="utf-8",
        )
        env = os.environ | {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "python"), os.environ.get("PYTHONPATH", "")])}
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300, check=False
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 2)  # two branches, run once each

        script.write_text(script.read_text().replace("k * FACTOR", "k * FACTOR + 1"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 4)  # editing the script reruns its stages

    def run_script(self, *argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        env = os.environ | {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "python"), os.environ.get("PYTHONPATH", "")])}
        return subprocess.run(
            [sys.executable, *argv], capture_output=True, text=True, env=env, timeout=300, check=False, cwd=cwd
        )

    def test_script_that_fails_while_loading_fails_the_stage(self) -> None:
        script = self.tmp / "failing.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import os, sys
                from poisson_runner import Cache, experiment_main, python_stage

                @python_stage
                def f(inputs):
                    return 1

                if os.environ.get("POISSON_RUNNER_PYTHON_STAGE"):
                    sys.exit(3)  # only inside the stage process

                def pipeline(run):
                    f(run)

                raise SystemExit(experiment_main(pipeline, cache=Cache({str(self.tmp / "cache")!r})))
                """
            ),
            encoding="utf-8",
        )
        proc = self.run_script(str(script))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("exit code 3", proc.stdout + proc.stderr)

    def test_script_run_with_python_m_is_imported_as_its_module(self) -> None:
        pkg = self.tmp / "proj" / "demo"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "helper.py").write_text("FACTOR = 10\n", encoding="utf-8")
        (pkg / "sweep.py").write_text(
            textwrap.dedent(
                f"""
                from pathlib import Path
                from poisson_runner import Cache, experiment_main, load_value, python_stage
                from .helper import FACTOR

                CALLS = Path({str(self.calls)!r})

                @python_stage
                def scaled(inputs):
                    with CALLS.open("a") as fh:
                        fh.write("x\\n")
                    return FACTOR

                def pipeline(run):
                    return {{"v": load_value(scaled(run))}}

                if __name__ == "__main__":
                    raise SystemExit(experiment_main(pipeline, cache=Cache({str(self.tmp / "cache")!r})))
                """
            ),
            encoding="utf-8",
        )
        for _ in range(2):
            proc = self.run_script("-m", "demo.sweep", cwd=pkg.parent)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 1)
        (pkg / "helper.py").write_text("FACTOR = 20\n", encoding="utf-8")  # the package is hashed
        proc = self.run_script("-m", "demo.sweep", cwd=pkg.parent)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 2)

    def test_dag_mode_plans_runs_and_collects_python_stages(self) -> None:
        script = self.tmp / "dag_sweep.py"
        script.write_text(
            textwrap.dedent(
                f"""
                from pathlib import Path
                from poisson_runner import Cache, compare, experiment_main, load_value, python_stage

                CALLS = Path({str(self.calls)!r})

                @python_stage
                def first(inputs, *, k):
                    with CALLS.open("a") as fh:
                        fh.write("x\\n")
                    return k

                @python_stage(outputs={{"out": ".txt"}})
                def second(inputs, outputs):
                    with CALLS.open("a") as fh:
                        fh.write("x\\n")
                    outputs["out"].write_text(inputs["value"].read_text() * 2)

                def pipeline(run):
                    k = compare("k", [1, 2])
                    return {{"out": second(run, inputs={{"value": first(run, k=k)}})}}

                if __name__ == "__main__":
                    raise SystemExit(experiment_main(pipeline, cache=Cache({str(self.tmp / "cache")!r})))
                """
            ),
            encoding="utf-8",
        )
        plan = self.tmp / "plan.json"
        proc = self.run_script(str(script), "--dag-plan", str(plan), "--no-fuse")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        groups = [line.split("|")[:2] for line in proc.stdout.splitlines() if line.startswith("dag-group ")]
        self.assertEqual(len(groups), 2, proc.stdout)
        for head, count in groups:
            for task in range(int(count)):
                proc = self.run_script(str(script), "--dag-plan", str(plan), "--dag-task", f"{head.split()[1]}/{task}")
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 4)
        proc = self.run_script(str(script), "--collect", "--out", str(self.tmp / "out"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.n_calls(), 4)


if __name__ == "__main__":
    unittest.main()
