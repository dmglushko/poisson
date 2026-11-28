"""Cached pipeline stages backed by plain Python functions (`@python_stage`)."""

from __future__ import annotations

import functools
import inspect
import json
import os
import site
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .cache import Artifact, CacheMode, _validate_ext, normalize_cache_mode
from .hashing import blake3_json, canonical_json
from .stages import Stage, StageCall

if TYPE_CHECKING:
    from .driver import RunContext

# Set in the child process before the stage function's module is imported, so an experiment
# script that defines the function can be loaded without starting the experiment again
# (`experiment_main` returns at once when it is set).
CHILD_ENV = "POISSON_RUNNER_PYTHON_STAGE"

# Keyword arguments of a stage call that go to `run_stage`, so a stage function can't use them
# as setting names.
_RUN_OPTIONS = ("cache", "use_cache", "repeats", "warmup", "env", "resources", "threads")
_RESERVED = frozenset({*_RUN_OPTIONS, "inputs", "outputs", "python_executable"})

# Directory holding this copy of the `poisson_runner` package.
_RUNNER_PATH = Path(__file__).resolve().parent.parent

# Inherited variables that change which packages the child interpreter imports.
_PYTHON_ENV_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONUSERBASE")

# Run by the child interpreter itself, with the same `-P` flag, environment and sys.path
# additions as the stage, so the versions are those the stage will actually import.
# argv[1] is `{"prepend": [...], "append": [...]}`, then the distribution names.
_ENV_PROBE = """\
import importlib.metadata as md, json, sys
paths = json.loads(sys.argv[1])
sys.path[:0] = paths["prepend"]
sys.path.extend(paths["append"])
def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
print(json.dumps({
    "prefix": sys.prefix,
    "version": sys.version,
    "version_info": list(sys.version_info[:3]),
    "packages": {name: version(name) for name in sys.argv[2:]},
}, sort_keys=True))
"""

# Standard library only: the stage may run under another interpreter that has no poisson_runner
# installed (the runner's own copy, itself standard library only, is appended to its sys.path
# when it comes from a checkout). Written to `bootstrap.py` in the staging directory and run as
# `python -P bootstrap.py manifest.json`.
_BOOTSTRAP = """\
import importlib, json, math, os, sys, time, warnings
from importlib.machinery import PathFinder, SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
os.environ[m["child_env"]] = "1"
sys.path[:0] = m["prepend"]
sys.path.extend(m["append"])

# The cache hashed these files' source, so compile them from source: a stale `.pyc` whose
# timestamp and size still match must never run under the new hash.
tracked = {str(Path(p).resolve()) for p in m["tracked"]}

class FromSource(SourceFileLoader):
    def path_stats(self, path):
        raise OSError("tracked by the cache: compiled from source")  # skips reading and writing .pyc

class FromSourceFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not spec.origin or not isinstance(spec.loader, SourceFileLoader):
            return None
        if str(Path(spec.origin).resolve()) not in tracked:
            return None
        spec.loader = FromSource(spec.loader.name, spec.loader.path)
        return spec

    @classmethod
    def invalidate_caches(cls):
        pass

sys.meta_path.insert(sys.meta_path.index(PathFinder) if PathFinder in sys.meta_path else len(sys.meta_path),
                     FromSourceFinder)

expected = Path(m["file"]).resolve()
if m["kind"] == "script":
    name = "__poisson_stage_script__"
    spec = spec_from_file_location(name, expected, loader=FromSource(name, str(expected)))
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise  # the script itself failed while loading
        # else: `raise SystemExit(experiment_main(...))` at the end of the experiment script
else:
    try:
        mod = importlib.import_module(m["module"])
    except SystemExit as exc:
        raise RuntimeError(
            f"importing {m['module']} raised SystemExit({exc.code!r}); a module run with `python -m` must "
            "guard `experiment_main(...)` with `if __name__ == \\"__main__\\":`"
        ) from exc
    actual = Path(mod.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"imported {m['module']} from {actual}, expected {expected}")
fn = mod
for part in m["qualname"].split("."):
    fn = getattr(fn, part, None)
    if fn is None:
        raise RuntimeError(f"{expected} does not define {m['qualname']}")
if type(fn).__name__ == "PythonStage":
    fn = fn.__wrapped__  # the name holds the @python_stage wrapper; call the function itself

def check_json(value, where):
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite float {value!r} is not JSON (return None instead)")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            check_json(v, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: dict key {k!r} is not a str")
            check_json(v, f"{where}[{k!r}]")
        return
    raise TypeError(f"{where}: {type(value).__name__} is not JSON (use float(), int(), list(), ...)")

def as_path(value):
    return [Path(v) for v in value] if isinstance(value, list) else Path(value)

inputs = {k: as_path(v) for k, v in m["inputs"].items()}
outputs = {k: Path(v) for k, v in m["outputs"].items()}
for p in outputs.values():
    p.unlink(missing_ok=True)
info = {}
kwargs = dict(m["settings"])
if m["takes_info"]:
    kwargs["info"] = info
args = [inputs] if m["mode"] == "value" else [inputs, dict(outputs)]
# Python's own warning filters apply: library DeprecationWarnings stay hidden, and a warning
# raised repeatedly at one place is recorded once.
with warnings.catch_warnings(record=True) as caught:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

if m["mode"] == "value":
    check_json(result, "return value")
    text = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    outputs["out"].write_text(text + "\\n", encoding="utf-8")
else:
    if result is not None:
        raise TypeError(f"{m['qualname']} declares output files, so it must return None, got {type(result).__name__}")
    missing = [k for k, p in outputs.items() if not p.is_file()]
    if missing:
        raise RuntimeError(f"{m['qualname']} did not write output(s): {', '.join(missing)}")

if not isinstance(info, dict):
    raise TypeError("info must stay a dict")
info.setdefault("time", elapsed)
own = info.get("warnings")
if own is None:
    own = []
elif isinstance(own, str):
    own = [own]
elif isinstance(own, (list, tuple)):
    own = [str(w) for w in own]
else:
    raise TypeError(f"info['warnings'] must be a list of str, got {type(own).__name__}")
messages = list(dict.fromkeys([*own, *(f"{w.category.__name__}: {w.message}" for w in caught)]))
if messages:
    info["warnings"] = messages
else:
    info.pop("warnings", None)
check_json(info, "info")
Path(m["info"]).write_text(json.dumps(info, sort_keys=True, allow_nan=False) + "\\n", encoding="utf-8")
"""


def _resolve_python(python_executable: Path | str | None) -> Path:
    # Keep symlinks intact (abspath, not resolve) so a virtualenv `bin/python` still finds its
    # `pyvenv.cfg` when launched as a subprocess.
    return Path(os.path.abspath(python_executable if python_executable is not None else sys.executable))


def _normalize_env(env: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Same rules as `RunContext._normalize_env`: str keys, values converted with `str()`."""
    if env is None:
        return None
    out: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(k, str):
            raise TypeError(f"Stage env keys must be str, got {type(k).__name__}")
        out[k] = str(v)
    return out


def _probe_environment(
    py: Path, packages: Sequence[str], env: Mapping[str, str] | None, paths: Mapping[str, list[str]]
) -> dict[str, Any]:
    """Ask interpreter `py`, under the stage's own environment and sys.path, for its versions."""
    try:
        proc = subprocess.run(
            [str(py), "-P", "-c", _ENV_PROBE, json.dumps(paths), *packages],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=os.environ | dict(env or {}),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot run Python interpreter {py}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"Python interpreter {py} failed its environment probe: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


@functools.lru_cache(maxsize=1)
def _installed_roots() -> tuple[Path, ...]:
    """Where this interpreter's standard library and installed packages live."""
    candidates = [sysconfig.get_path(key) for key in ("stdlib", "platstdlib", "purelib", "platlib")]
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:  # some embedded interpreters
        pass
    try:
        candidates.append(site.getusersitepackages())
    except AttributeError:
        pass
    return tuple(sorted({Path(p).resolve() for p in candidates if p}))


def _is_installed(path: Path) -> bool:
    return any(path == r or r in path.parents for r in _installed_roots())


def _package_files(directory: Path) -> list[Path]:
    """Every `*.py` under `directory`, skipping `__pycache__` and dot directories, sorted."""
    files = []
    for p in directory.rglob("*.py"):
        rel = p.relative_to(directory)
        if any(part == "__pycache__" or part.startswith(".") for part in rel.parts[:-1]):
            continue
        files.append(p)
    return sorted(files, key=lambda p: p.relative_to(directory).as_posix())


class _Target:
    """Where the stage function lives, how the child process imports it, and which source files
    decide its cache entries."""

    def __init__(self, fn: Callable[..., Any], *, hash_package: bool) -> None:
        module_name = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", None)
        if not inspect.isfunction(fn) or not module_name or not qualname:
            raise TypeError(f"python_stage needs a plain Python function, got {fn!r}")
        if "<lambda>" in qualname or "<locals>" in qualname or fn.__closure__:
            raise ValueError(
                f"python_stage function {qualname} must be defined at the top level of a module or script "
                "(not a lambda, a function inside another function, or a decorator's wrapper), so the stage "
                "process can import it"
            )
        src = inspect.getsourcefile(fn)
        if src is None:
            raise ValueError(f"python_stage function {module_name}:{qualname} has no source file")
        self.file = Path(src).resolve()
        self.qualname = qualname
        self.hash_package = hash_package
        module = sys.modules.get(module_name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is not None and Path(mod_file).resolve() != self.file:
            raise ValueError(
                f"python_stage function {module_name}:{qualname} is defined in {self.file}, not in its module's "
                f"file {mod_file} (a wrapper made with functools.wraps?); decorate the original function"
            )
        spec = getattr(module, "__spec__", None)
        if module_name == "__main__" and spec is not None and spec.name and spec.name != "__main__":
            module_name = spec.name  # `python -m pkg.sweep`: import it as the module it is
        self.module = module_name
        self.installed = _is_installed(self.file)
        self._package: Any = None
        if module_name == "__main__":
            # A function in the experiment script: the child loads the script by path.
            self.kind = "script"
            self.root: Path | None = self.file.parent
            self.label = f"{self.file.name}:{qualname}"
            return
        self.kind = "module"
        self.label = f"{module_name}:{qualname}"
        depth = module_name.count(".") + (1 if self.file.name == "__init__.py" else 0)
        top = sys.modules.get(module_name.split(".")[0])
        if top is not None and getattr(top, "__path__", None) is not None:
            self._package = top
        # An installed module is imported from the child's own environment (a different copy
        # then fails the bootstrap's file check), so its directory isn't put on sys.path.
        self.root = None if self.installed else self.file.parents[depth]

    def code_files(self) -> list[tuple[Path, str]]:
        """`(file, name)` of the source files to hash, scanned now; `name` is relative to the
        import root, so renaming a file changes the hash but moving the checkout does not."""
        if self.kind == "script":
            return [(self.file, "<script>")]  # a renamed or copied script with the same code shares entries
        pkg = self._package
        if self.hash_package and pkg is not None and not self.installed:
            pkg_file = getattr(pkg, "__file__", None)
            dirs = [Path(pkg_file).resolve().parent] if pkg_file else [Path(d).resolve() for d in list(pkg.__path__)]
            files: list[tuple[Path, str]] = []
            for d in dirs:
                if _is_installed(d) or not d.is_dir():
                    continue
                files.extend((p, p.relative_to(d.parent).as_posix()) for p in _package_files(d))
            if any(p == self.file for p, _ in files):
                return files
        name = self.file.relative_to(self.root).as_posix() if self.root is not None else self.file.name
        return [(self.file, name)]


class PythonStage:
    """A cached pipeline stage running a plain Python function in its own process.

    Created with `@python_stage` (see there); call it as `stage(run, inputs={...}, **settings)`.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        outputs: Mapping[str, str] | None = None,
        packages: Sequence[str] = (),
        code_files: Sequence[Path | str] = (),
        version: str | None = None,
        hash_package: bool = True,
        strict_interpreter: bool = False,
        python_executable: Path | str | None = None,
        default_cache: CacheMode | str | bool = CacheMode.USE,
    ) -> None:
        self.fn = fn
        self.target = _Target(fn, hash_package=hash_package)
        self.name = name if name is not None else fn.__name__
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("python_stage name must be a non-empty string")
        self.mode = "value" if outputs is None else "files"
        if outputs is None:
            self.output_specs = {"out": ".json"}
        else:
            if not outputs:
                raise ValueError(f"python_stage {self.name}: outputs must name at least one file")
            self.output_specs = {}
            for k, ext in outputs.items():
                if not isinstance(k, str) or not k or not isinstance(ext, str):
                    raise ValueError(f"python_stage {self.name}: outputs must map names to extensions like '.h5'")
                ext = ext if ext.startswith(".") else f".{ext}"
                try:
                    self.output_specs[k] = _validate_ext(ext)
                except ValueError as exc:
                    raise ValueError(f"python_stage {self.name}: output {k!r}: {exc}") from exc
        if isinstance(packages, str):
            raise TypeError("packages must be a list of distribution names, not a str")
        self.packages = tuple(sorted(set(packages)))
        if isinstance(code_files, (str, bytes, os.PathLike)):
            raise TypeError("code_files must be a list of paths, not a single path")
        self.code_file_paths: list[Path] = []
        for f in code_files:
            p = Path(f).resolve()
            if not p.exists():
                raise FileNotFoundError(f"python_stage {self.name}: code_files entry {f} does not exist")
            self.code_file_paths.append(p)
        self.version = version
        self.hash_package = hash_package
        self.strict_interpreter = strict_interpreter
        self.python_executable = python_executable
        self.default_cache = normalize_cache_mode(default_cache)

        self._signature = inspect.signature(fn)
        params = list(self._signature.parameters.values())
        n_positional = 1 if self.mode == "value" else 2
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional) < n_positional:
            want = "(inputs, **settings)" if self.mode == "value" else "(inputs, outputs, **settings)"
            raise TypeError(f"python_stage function {self.target.label} must take {want}")
        self._positional = [p.name for p in positional[:n_positional]]
        self.takes_info = "info" in self._signature.parameters and (
            self._signature.parameters["info"].kind == inspect.Parameter.KEYWORD_ONLY
        )
        clashes = sorted(
            p.name for p in params if p.name in _RESERVED and p.name not in self._positional
        )
        if clashes:
            raise ValueError(
                f"python_stage function {self.target.label} uses reserved parameter name(s) "
                f"{', '.join(clashes)}; rename them"
            )
        functools.update_wrapper(self, fn)

    # ------------------------------------------------------------------ call

    def __call__(
        self,
        run: RunContext,
        *,
        inputs: Mapping[str, Any] = (),  # type: ignore[assignment]
        python_executable: Path | str | None = None,
        **kwargs: Any,
    ) -> Artifact | dict[str, Artifact]:
        """Run (or reuse) the stage. Returns the `.json` value Artifact, or the output Artifacts
        (a dict, or the single Artifact when the only output is named `out`)."""
        run_options = {k: kwargs.pop(k) for k in _RUN_OPTIONS if k in kwargs}
        settings = dict(kwargs)
        self._check_settings(settings)
        py = _resolve_python(python_executable if python_executable is not None else self.python_executable)
        env = _normalize_env(run_options.get("env"))
        if "env" in run_options:
            run_options["env"] = env
        effective = os.environ | dict(env or {})
        # Everything that shapes the Stage, so two wrappers of one function with different
        # options never share one, while an identical wrapper rebuilt per branch reuses it.
        key = (
            "python_stage",
            str(self.target.file),
            self.target.module,
            self.target.qualname,
            self.name,
            self.mode,
            tuple(self.output_specs.items()),
            self.packages,
            tuple(str(p) for p in self.code_file_paths),
            self.version,
            self.hash_package,
            self.strict_interpreter,
            self.default_cache,
            str(py),
            tuple(sorted((env or {}).items())),
            tuple((k, effective.get(k)) for k in _PYTHON_ENV_KEYS),
        )
        stage = run.get_stage(key, lambda _build_dir: self.make_stage(py, env))
        return run.run_stage(stage, inputs=dict(inputs), settings=settings, **run_options)

    def _check_settings(self, settings: dict[str, Any]) -> None:
        where = f"python_stage {self.name}"
        if self.takes_info and "info" in settings:
            raise ValueError(f"{where}: 'info' is filled by the stage itself, not passed as a setting")
        _reject_paths(settings, where)
        try:
            canonical_json(settings)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{where}: settings must be plain JSON values ({exc})") from exc
        extra = {"info": {}} if self.takes_info else {}
        try:
            self._signature.bind(*[None] * len(self._positional), **extra, **settings)
        except TypeError as exc:
            raise TypeError(f"{where}: settings don't match {self.target.label}{self._signature}: {exc}") from exc

    # ------------------------------------------------------------------ stage

    def _sys_path_additions(self) -> dict[str, list[str]]:
        target = self.target
        prepend = [str(target.root)] if target.root is not None else []
        append = [] if _is_installed(_RUNNER_PATH / "poisson_runner") else [str(_RUNNER_PATH)]
        return {"prepend": prepend, "append": append}

    def _code_files(self) -> list[tuple[Path, str]]:
        files = self.target.code_files()
        for p in self.code_file_paths:
            if p.is_dir():
                files.extend((f, f"<code_files>/{p.name}/{f.relative_to(p).as_posix()}") for f in _package_files(p))
            elif p.is_file():
                files.append((p, f"<code_files>/{p.name}"))
            else:
                raise FileNotFoundError(f"python_stage {self.name}: code_files entry {p} no longer exists")
        return files

    def make_stage(self, py: Path, env: Mapping[str, str] | None = None) -> Stage:
        """Build the `Stage` for interpreter `py`, probing it under the stage's `env`. The source
        files are scanned here, so each experiment sees files added or removed since."""
        paths = self._sys_path_additions()
        probe = _probe_environment(py, self.packages, env, paths)
        missing = [p for p, v in probe["packages"].items() if v is None]
        if missing:
            raise RuntimeError(
                f"python_stage {self.name}: package(s) {', '.join(missing)} are not installed for {py}"
            )
        target = self.target
        code_files = self._code_files()
        vi = probe["version_info"]
        version_parts: dict[str, Any] = {
            "v": 1,
            "target": target.label if target.kind == "module" else f"<script>:{target.qualname}",
            "mode": self.mode,
            "takes_info": self.takes_info,
            "code_names": [name for _, name in code_files],
            "python": f"{vi[0]}.{vi[1]}.{vi[2]}",
            "packages": probe["packages"],
        }
        if self.strict_interpreter:
            version_parts["sys.version"] = probe["version"]
            version_parts["prefix"] = probe["prefix"]
        if self.version is not None:
            version_parts["user"] = self.version
        version = f"python_stage:{blake3_json(version_parts)[:16]}"

        manifest_base = {
            "child_env": CHILD_ENV,
            **paths,
            "tracked": [str(p) for p, _ in code_files],
            "kind": target.kind,
            "module": target.module,
            "qualname": target.qualname,
            "file": str(target.file),
            "mode": self.mode,
            "takes_info": self.takes_info,
        }

        def build_command(call: StageCall) -> list[str]:
            manifest = dict(manifest_base)
            manifest["inputs"] = {
                k: [str(a.path) for a in v] if isinstance(v, list) else str(v.path) for k, v in call.inputs.items()
            }
            manifest["outputs"] = {k: str(p) for k, p in call.outputs.items()}
            manifest["info"] = str(call.info_path)
            manifest["settings"] = json.loads(canonical_json(call.settings))
            staging = call.info_path.parent
            manifest_path = staging / "python_stage.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            bootstrap_path = staging / "python_stage_bootstrap.py"
            bootstrap_path.write_text(_BOOTSTRAP, encoding="utf-8")
            return [str(py), "-P", str(bootstrap_path), str(manifest_path)]

        return Stage(
            name=self.name,
            build_command=build_command,
            executable=py,
            output_specs=dict(self.output_specs),
            code_files=[p for p, _ in code_files],
            version=version,
            default_cache=self.default_cache,
            track_shared_libs=False,
            hash_executable=self.strict_interpreter,
            normalize_settings=False,  # the function can tell 1 from 1.0
        )


def _reject_paths(value: Any, where: str) -> None:
    if isinstance(value, (os.PathLike, Artifact)):
        raise TypeError(
            f"{where}: file {value} given as a setting; pass files through inputs={{...}} so the cache "
            "hashes their contents, not their path"
        )
    if isinstance(value, Mapping):
        for v in value.values():
            _reject_paths(v, where)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _reject_paths(v, where)


def python_stage(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    outputs: Mapping[str, str] | None = None,
    packages: Sequence[str] = (),
    code_files: Sequence[Path | str] = (),
    version: str | None = None,
    hash_package: bool = True,
    strict_interpreter: bool = False,
    python_executable: Path | str | None = None,
    default_cache: CacheMode | str | bool = CacheMode.USE,
) -> Any:
    """Turn a top-level Python function into a cached pipeline stage.

    Two kinds, chosen by `outputs`:

    - value stage (no `outputs`): `fn(inputs, **settings)` returns a JSON value (dicts with str
      keys, lists, str, int, finite float, bool, None), cached as a `.json` output; read it with
      `load_value(artifact)`.
    - file stage (`outputs={"mesh": ".h5", ...}`): `fn(inputs, outputs, **settings)` writes each
      file to the path `outputs[name]` and returns None.

    `inputs` maps names to `Path` (or a list of them, for a list input). A keyword-only `info`
    parameter receives a dict for log data (`.info.json`); `time` and caught Python warnings are
    added to it. Call the stage as `stage(run, inputs={...}, **settings)`; `cache=`, `repeats=`,
    `warmup=`, `env=`, `resources=`, `threads=` and `python_executable=` go to the runner.

    The function runs in its own Python process (`python_executable`, default this one). Its
    cache entry depends on the input contents, the settings (`1` and `1.0` differ), the source
    files (the function's whole package, `hash_package=False` for just its module file; the
    whole script for a function in the experiment script), `code_files` (files, or every `*.py`
    under a directory), `version`, the Python version, and the versions of the distributions in
    `packages`. Anything else it reads is not tracked.
    """

    def wrap(f: Callable[..., Any]) -> PythonStage:
        return PythonStage(
            f,
            name=name,
            outputs=outputs,
            packages=packages,
            code_files=code_files,
            version=version,
            hash_package=hash_package,
            strict_interpreter=strict_interpreter,
            python_executable=python_executable,
            default_cache=default_cache,
        )

    return wrap(fn) if fn is not None else wrap


def load_value(artifact: Artifact) -> Any:
    """Read the JSON value a value stage returned."""
    if artifact.content_id.startswith("dryrun:"):
        raise ValueError(
            f"{artifact.stage} output is a dry-run placeholder; check `run.dry_run` before reading its value"
        )
    return json.loads(artifact.path.read_text(encoding="utf-8"))
