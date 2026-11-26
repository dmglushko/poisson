"""poisson_runner Stage factories and helpers for poisson_test_solver.

Kept inside `poisson_test_solver` (not `poisson_runner.stages`) so the core runner stays
independent of this external-library test backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from poisson_runner.stages import Stage, StageCall

from .validation import require_finite_float, require_int

if TYPE_CHECKING:
    from poisson_runner.cache import Artifact
    from poisson_runner.driver import RunContext

_PKG_DIR = Path(__file__).resolve().parent

# Every third-party distribution any stage imports; each stage hashes its own subset.
_ALL_DEPS: tuple[str, ...] = ("numpy", "scipy", "h5py", "scikit-fem", "pyamg")

# Run by the child interpreter itself (same `-P` flag and environment as the stage), so the
# reported prefix and versions are those the stage will actually import.
_ENV_PROBE = """\
import importlib.metadata as md, json, sys
def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "missing"
print(json.dumps({
    "prefix": sys.prefix,
    "version": sys.version,
    "version_info": list(sys.version_info),
    "packages": {name: version(name) for name in sys.argv[1:]},
}, sort_keys=True))
"""

# Imports this exact package copy (the one whose files are hashed into the code hash) whatever
# PYTHONPATH says. The expected package directory arrives as argv[1] (inserted by
# `_make_python_cli_stage`) rather than being baked into the code, so the cache key does not
# depend on where the checkout lives; the hashed code files and versions already pin it.
_BOOTSTRAP = """\
import sys
from pathlib import Path
expected = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(expected.parent))
import poisson_test_solver
actual = Path(poisson_test_solver.__file__).resolve().parent
if actual != expected:
    raise RuntimeError(f"imported poisson_test_solver from {actual}, expected {expected}")
from poisson_test_solver.cli import main
sys.exit(main(sys.argv[1:]))
"""

def _package_code_files() -> list[Path]:
    return sorted(_PKG_DIR.glob("*.py"))


def _query_child_environment(py: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ask interpreter `py` for its prefix, version, and dependency versions.

    Not memoized: the probe runs under the stage's own environment (`os.environ` overlaid with
    the per-stage `env`, exactly as the runner launches it), so a changed `PYTHONPATH`,
    `PYTHONHOME`, `PYTHONNOUSERSITE`, or `PYTHONUSERBASE` is always reflected. Stages are built
    once per experiment (`run.get_stage`), so this costs one subprocess per stage.
    """
    key = str(py)
    try:
        proc = subprocess.run(
            [key, "-P", "-c", _ENV_PROBE, *_ALL_DEPS],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=os.environ | dict(env or {}),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot run Python interpreter {key}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"Python interpreter {key} failed its environment probe: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _deps_version_tag(
    stage_tag: str, py: Path, packages: tuple[str, ...], env: Mapping[str, str] | None = None
) -> str:
    probe = _query_child_environment(py, env)
    vi = probe["version_info"]
    parts = [
        stage_tag,
        f"py={vi[0]}.{vi[1]}.{vi[2]}",
        f"sys.version={probe['version']}",
        f"prefix={probe['prefix']}",
    ]
    for pkg in packages:
        parts.append(f"{pkg}={probe['packages'].get(pkg, 'missing')}")
    return "|".join(parts)


# Inherited variables that change which packages the child interpreter imports.
_PYTHON_ENV_KEYS: tuple[str, ...] = ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONUSERBASE")


def _stage_key(kind: str, py: Path, env: Mapping[str, str] | None) -> tuple[Any, ...]:
    """`run.get_stage` key: one Stage per interpreter, per-stage `env` override, and effective
    Python import environment (`os.environ` overlaid with `env`), so changing an inherited
    `PYTHONPATH` & co. within one experiment builds (and probes) a fresh Stage."""
    effective = os.environ | dict(env or {})
    python_env = tuple((k, effective.get(k)) for k in _PYTHON_ENV_KEYS)
    return (f"poisson_test_solver.{kind}", str(py), tuple(sorted(dict(env or {}).items())), python_env)


def _resolve_python(python_executable: Path | str | None) -> Path:
    # Keep symlinks intact (abspath, NOT resolve) so virtualenv `bin/python` symlinks
    # preserve `pyvenv.cfg` discovery when launched as a subprocess.
    return Path(os.path.abspath(python_executable if python_executable is not None else sys.executable))


def _make_python_cli_stage(
    name: str,
    *,
    subcommand: str,
    input_flags: dict[str, str],
    config_flag: str | None,
    version_tag: str,
    dep_packages: tuple[str, ...],
    python_executable: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Stage:
    py = _resolve_python(python_executable)
    # `-P` keeps the working directory off sys.path; no `-I`/`-E`, so PYTHONPATH and user
    # site-packages (DelftBlue module setups, `pip install --user`) still provide the deps.
    static_args = ["-P", "-c", _BOOTSTRAP]
    stage = Stage.from_cli(
        name=name,
        binary=py,
        extra_args=[*static_args, subcommand],
        input_flags=input_flags,
        config_flag=config_flag,
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        code_files=_package_code_files(),
        version=_deps_version_tag(version_tag, py, dep_packages, env),
        track_shared_libs=False,
    )
    # Restore the unresolved venv symlink path (Stage.from_cli calls Path.resolve()).
    stage.executable = py
    base_command = stage.build_command
    n_prefix = 1 + len(static_args)

    def build_command(call: StageCall) -> list[str]:
        cmd = base_command(call)
        return [*cmd[:n_prefix], str(_PKG_DIR), *cmd[n_prefix:]]

    stage.build_command = build_command
    return stage


def make_assemble_stage(
    *, python_executable: Path | str | None = None, env: Mapping[str, str] | None = None
) -> Stage:
    """Create the cached Stage for the `assemble` subcommand.

    Pass the same per-stage `env` the stage will run with so the dependency probe sees it.
    """
    return _make_python_cli_stage(
        "test_assemble",
        subcommand="assemble",
        input_flags={"mesh": "--mesh"},
        config_flag="--config",
        version_tag="test_assemble_v2",
        dep_packages=("numpy", "scipy", "h5py", "scikit-fem"),
        python_executable=python_executable,
        env=env,
    )


def make_solve_stage(
    *, python_executable: Path | str | None = None, env: Mapping[str, str] | None = None
) -> Stage:
    """Create the cached Stage for the `solve` subcommand.

    Pass the same per-stage `env` the stage will run with so the dependency probe sees it.
    """
    return _make_python_cli_stage(
        "test_solve",
        subcommand="solve",
        input_flags={"problem": "--in"},
        config_flag="--config",
        version_tag="test_solve_v2",
        dep_packages=("numpy", "scipy", "h5py", "pyamg"),
        python_executable=python_executable,
        env=env,
    )


def make_field_stage(
    *, python_executable: Path | str | None = None, env: Mapping[str, str] | None = None
) -> Stage:
    """Create the cached Stage for the `field` subcommand.

    Pass the same per-stage `env` the stage will run with so the dependency probe sees it.
    """
    return _make_python_cli_stage(
        "test_field",
        subcommand="field",
        input_flags={"mesh": "--mesh", "problem": "--problem", "solution": "--solution"},
        config_flag=None,
        version_tag="test_field_v2",
        dep_packages=("numpy", "scipy", "h5py"),
        python_executable=python_executable,
        env=env,
    )


def assemble(
    run: RunContext,
    mesh: Artifact | str | os.PathLike[str],
    *,
    constant_source: float | None = None,
    intorder: int = 2,
    python_executable: Path | str | None = None,
    **options: Any,
) -> Artifact:
    """Assemble `mesh` into an `AssembledProblem` `.h5` artifact."""
    py = _resolve_python(python_executable)
    env = options.get("env")
    stage = run.get_stage(
        _stage_key("assemble", py, env),
        lambda _build_dir: make_assemble_stage(python_executable=py, env=env),
    )
    settings: dict[str, Any] = {"intorder": require_int(intorder, "intorder")}
    if constant_source is not None:
        settings["constant_source"] = require_finite_float(constant_source, "constant_source")
    return run.run_stage(stage, inputs={"mesh": mesh}, settings=settings, **options)


def solve(
    run: RunContext,
    problem: Artifact | str | os.PathLike[str],
    *,
    method: str = "auto",
    rtol: float = 1e-12,
    maxiter: int = 2000,
    direct_threshold: int | None = None,
    seed: int = 0,
    python_executable: Path | str | None = None,
    **options: Any,
) -> Artifact:
    """Solve `problem` into a `SolverResult` `.h5` artifact (`seed` makes `amg` reproducible)."""
    py = _resolve_python(python_executable)
    env = options.get("env")
    stage = run.get_stage(
        _stage_key("solve", py, env),
        lambda _build_dir: make_solve_stage(python_executable=py, env=env),
    )
    settings: dict[str, Any] = {
        "method": method,
        "rtol": require_finite_float(rtol, "rtol"),
        "maxiter": require_int(maxiter, "maxiter"),
        "seed": require_int(seed, "seed"),
    }
    if direct_threshold is not None:
        settings["direct_threshold"] = require_int(direct_threshold, "direct_threshold")
    return run.run_stage(stage, inputs={"problem": problem}, settings=settings, **options)


def build_field(
    run: RunContext,
    mesh: Artifact | str | os.PathLike[str],
    problem: Artifact | str | os.PathLike[str],
    solution: Artifact | str | os.PathLike[str],
    *,
    python_executable: Path | str | None = None,
    **options: Any,
) -> Artifact:
    """Build a `NodalField` `.h5` artifact suitable for `run.adapt_mesh(...)`."""
    py = _resolve_python(python_executable)
    env = options.get("env")
    stage = run.get_stage(
        _stage_key("field", py, env),
        lambda _build_dir: make_field_stage(python_executable=py, env=env),
    )
    return run.run_stage(
        stage,
        inputs={"mesh": mesh, "problem": problem, "solution": solution},
        **options,
    )


def solve_mesh(
    run: RunContext,
    mesh: Artifact | str | os.PathLike[str],
    *,
    constant_source: float | None = None,
    intorder: int = 2,
    method: str = "auto",
    rtol: float = 1e-12,
    maxiter: int = 2000,
    direct_threshold: int | None = None,
    seed: int = 0,
    python_executable: Path | str | None = None,
    **options: Any,
) -> dict[str, Artifact]:
    """Convenience helper running assemble -> solve -> build_field in sequence."""
    problem = assemble(
        run,
        mesh,
        constant_source=constant_source,
        intorder=intorder,
        python_executable=python_executable,
        **options,
    )
    solver_result = solve(
        run,
        problem,
        method=method,
        rtol=rtol,
        maxiter=maxiter,
        direct_threshold=direct_threshold,
        seed=seed,
        python_executable=python_executable,
        **options,
    )
    field = build_field(
        run,
        mesh,
        problem,
        solver_result,
        python_executable=python_executable,
        **options,
    )
    return {"problem": problem, "solver_result": solver_result, "field": field}
