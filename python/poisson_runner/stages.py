"""Stage definitions and thin wrappers around per-stage C++ CLI binaries."""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .cache import Artifact, Cache, CacheMode, file_signature, normalize_cache_mode
from .hashing import blake3_json

if TYPE_CHECKING:
    from .driver import RunContext

_LDD_LINE_RE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*$")
_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library:\s+\[([^\]]+)\]")


@functools.lru_cache(maxsize=256)
def _shared_library_deps_for(path: str, _signature: tuple[str, int, int, int, int]) -> tuple[tuple[str, str], ...]:
    ldd = shutil.which("ldd")
    if ldd is None:
        return ()
    try:
        proc = subprocess.run([ldd, path], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ()
    if proc.returncode != 0:
        return ()  # Static binary or interpreted script: nothing to track.
    resolved: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = _LDD_LINE_RE.match(line)
        if m:
            resolved[m.group(1)] = m.group(2)

    needed: list[str] | None = None
    readelf = shutil.which("readelf")
    if readelf is not None:
        try:
            dyn = subprocess.run([readelf, "-d", path], capture_output=True, text=True, timeout=30, check=False)
            if dyn.returncode == 0:
                needed = _NEEDED_RE.findall(dyn.stdout)
        except (OSError, subprocess.SubprocessError):
            needed = None
    names = needed if needed is not None else sorted(resolved)
    return tuple((name, resolved[name]) for name in sorted(set(names)) if name in resolved)


def shared_library_deps(path: Path | str) -> list[tuple[str, Path]]:
    """Return `(soname, resolved_path)` for the shared libraries an ELF executable links directly.

    Uses `readelf -d` (direct `NEEDED` entries) resolved through `ldd`; falls back to every library
    `ldd` lists if `readelf` is unavailable. Returns `[]` for scripts, static binaries, or non-Linux
    hosts without `ldd`. Memoized per file signature, so rebuilding the binary re-scans it.
    """
    p = Path(path).resolve()
    return [(name, Path(lib)) for name, lib in _shared_library_deps_for(str(p), file_signature(p))]


def resolve_binary(name: str, build_dir: Path | str | None = None) -> Path:
    """Locate a compiled stage binary in `build_dir`, `POISSON_BUILD_DIR`, or `<repo>/build`."""
    p = Path(name)
    if p.is_file():
        return p.resolve()

    search_dirs: list[Path] = []
    if build_dir is not None:
        search_dirs.append(Path(build_dir))
    env_dir = os.environ.get("POISSON_BUILD_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))

    repo_root = Path(__file__).resolve().parents[2]
    search_dirs.extend([repo_root / "build", repo_root / "build/Release", repo_root / "build/Debug"])

    for directory in search_dirs:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"Stage binary '{name}' not found. Build the project with CMake or set POISSON_BUILD_DIR. "
        f"Searched: {searched}"
    )


@dataclass(frozen=True)
class StageCall:
    """Resolved paths and parameters passed to a Stage's `build_command` callback."""

    executable: Path | None
    inputs: dict[str, Artifact | list[Artifact]]
    outputs: dict[str, Path]
    info_path: Path
    settings: dict[str, Any]
    config_path: Path | None = None


@dataclass
class Stage:
    """Describes a cacheable pipeline stage backed by a CLI binary or script."""

    name: str
    build_command: Callable[[StageCall], list[str]]
    executable: Path | None = None
    output_specs: dict[str, str] = field(default_factory=lambda: {"out": ".h5"})
    code_files: list[Path] = field(default_factory=list)
    version: str | None = None
    ignore_binary_hash: bool = False
    default_cache: CacheMode | str | bool = CacheMode.USE
    uses_config_file: bool = False
    track_shared_libs: bool = True

    def __post_init__(self) -> None:
        self.default_cache = normalize_cache_mode(self.default_cache)

    @classmethod
    def from_cli(
        cls,
        name: str,
        *,
        binary: Path | str,
        input_flags: Mapping[str, str] = (),  # type: ignore[assignment]
        setting_flags: Mapping[str, str] = (),  # type: ignore[assignment]
        config_flag: str | None = None,
        output_flags: Mapping[str, str] | None = None,
        output_specs: Mapping[str, str] | None = None,
        info_flag: str | None = "--info-out",
        extra_args: list[str] | tuple[str, ...] = (),
        build_dir: Path | str | None = None,
        code_files: list[Path] | None = None,
        version: str | None = None,
        ignore_binary_hash: bool = False,
        default_cache: CacheMode | str | bool = CacheMode.USE,
        track_shared_libs: bool = True,
    ) -> Stage:
        """Declaratively register a CLI stage without writing a custom `build_command` lambda.

        - `input_flags`: maps input slot name -> CLI flag (list inputs repeat the flag).
        - `setting_flags`: maps flat scalar setting name -> CLI flag (optional).
        - `config_flag`: if set (e.g. `"--config"`), writes the entire nested `settings` dict
          to a temporary JSON file and passes `[config_flag, <path>]` — ideal for stages with
          many nested component options like AMR or `PoissonDriver`.
        - `track_shared_libs`: also hash the shared libraries the binary links directly (HDF5,
          libstdc++, ...) into the code hash, so a library upgrade invalidates this stage.
        """
        exe = resolve_binary(str(binary), build_dir=build_dir)
        in_map = dict(input_flags)
        set_map = dict(setting_flags)
        out_specs = dict(output_specs) if output_specs is not None else {"out": ".h5"}
        out_flags = dict(output_flags) if output_flags is not None else {k: f"--{k}" for k in out_specs}
        static_args = list(extra_args)

        def build_cmd(call: StageCall) -> list[str]:
            cmd = [str(call.executable), *static_args]
            for in_name, flag in in_map.items():
                val = call.inputs.get(in_name)
                if val is None:
                    continue
                if isinstance(val, list):
                    for art in val:
                        cmd.extend([flag, str(art.path)])
                else:
                    cmd.extend([flag, str(val.path)])

            if config_flag is not None:
                if call.config_path is None:
                    raise RuntimeError(f"Stage '{name}' expects a config_path for {config_flag}")
                cmd.extend([config_flag, str(call.config_path)])

            for set_name, flag in set_map.items():
                if set_name not in call.settings or call.settings[set_name] is None:
                    continue
                v = call.settings[set_name]
                if isinstance(v, bool):
                    cmd.extend([flag, "true" if v else "false"])
                else:
                    cmd.extend([flag, str(v)])

            for out_name, flag in out_flags.items():
                cmd.extend([flag, str(call.outputs[out_name])])

            if info_flag is not None:
                cmd.extend([info_flag, str(call.info_path)])
            return cmd

        spec_hash = blake3_json(
            {
                "in": in_map,
                "set": set_map,
                "cfg": config_flag,
                "out": out_flags,
                "info": info_flag,
                "extra": static_args,
            }
        )[:16]
        spec_version = f"{version}:{spec_hash}" if version else spec_hash

        return cls(
            name=name,
            executable=exe,
            output_specs=out_specs,
            code_files=list(code_files) if code_files is not None else [],
            version=spec_version,
            ignore_binary_hash=ignore_binary_hash,
            default_cache=default_cache,
            uses_config_file=config_flag is not None,
            track_shared_libs=track_shared_libs,
            build_command=build_cmd,
        )

    def tracked_code_files(self) -> list[Path]:
        """Return deduplicated resolved paths of all binaries/scripts hashed for this stage."""
        if self.ignore_binary_hash:
            return []
        files: list[Path] = []
        if self.executable is not None:
            files.append(Path(self.executable))
        files.extend(Path(f) for f in self.code_files)
        seen: set[Path] = set()
        unique: list[Path] = []
        for f in files:
            rp = f.resolve()
            if rp not in seen:
                seen.add(rp)
                unique.append(rp)
        return unique

    def tracked_shared_libs(self) -> list[tuple[str, Path]]:
        """Shared libraries linked directly by the stage executable (empty if not tracked)."""
        if self.ignore_binary_hash or not self.track_shared_libs or self.executable is None:
            return []
        return shared_library_deps(self.executable)

    def compute_code_hash(self, cache: Cache) -> str:
        """Hash the stage's binary, its directly linked shared libraries, code_files, and version."""
        parts: dict[str, Any] = {}
        if self.version is not None:
            parts["version"] = self.version
        if not self.ignore_binary_hash:
            parts["files"] = [cache.hash_file(f) for f in self.tracked_code_files()]
            libs = self.tracked_shared_libs()
            if libs:
                # Keyed by soname (not install path) so identical libraries in different prefixes match.
                parts["shared_libs"] = [[name, cache.hash_file(lib)] for name, lib in libs]
        return blake3_json(parts)


def make_import_mesh_stage(build_dir: Path | str | None = None) -> Stage:
    """Create the Stage for `poisson_import --mesh ...` -> `MeshProperties` `.h5`."""
    return Stage.from_cli(
        name="import_mesh",
        binary="poisson_import",
        input_flags={"mesh": "--mesh", "materials": "--material", "charges": "--charges"},
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="import_mesh_v1",
    )


def make_import_free_charges_stage(build_dir: Path | str | None = None) -> Stage:
    """Create the Stage for `poisson_import --primaries ...` -> `FreeCharges` `.h5`."""
    return Stage.from_cli(
        name="import_free_charges",
        binary="poisson_import",
        input_flags={"primaries": "--primaries"},
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="import_free_charges_v1",
    )


def make_adapt_stage(build_dir: Path | str | None = None) -> Stage:
    """Create the Stage for one `poisson_adapt` cycle.

    Inputs `mesh`, `solution`, optional `state` (the previous cycle's state); outputs `mesh`
    (refined, or the input mesh when not adapted), `parents` (identity when not adapted),
    `state` (for the next cycle), and `eta` (per-tet indicators of the input mesh).
    """
    return Stage.from_cli(
        name="adapt_mesh",
        binary="poisson_adapt",
        input_flags={"mesh": "--mesh", "solution": "--solution", "state": "--state"},
        config_flag="--config",
        output_flags={"mesh": "--out", "parents": "--parents-out", "state": "--state-out", "eta": "--eta-out"},
        output_specs={"mesh": ".h5", "parents": ".h5", "state": ".h5", "eta": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="adapt_mesh_v2",
    )


# ---------------------------------------------------------------------------------------------
# Project stage helpers. Each takes the active `RunContext` and forwards any extra keyword
# options (`cache`, `use_cache`, `repeats`, `warmup`, `env`, `resources`) to `run.run_stage`.
# Adding a new pipeline stage means adding a `make_*_stage` factory and a helper here; the
# generic driver does not need to change.
# ---------------------------------------------------------------------------------------------


def import_mesh(
    run: RunContext,
    mesh_file: Artifact | str | os.PathLike[str],
    *,
    materials: Sequence[Artifact | str | os.PathLike[str]] = (),
    charges: Artifact | str | os.PathLike[str] | None = None,
    **options: Any,
) -> Artifact:
    """Run the cached `poisson_import --mesh` stage and return the `MeshProperties` Artifact."""
    stage = run.get_stage("import_mesh", make_import_mesh_stage)

    mat_pairs = []
    for m in materials:
        art = m if isinstance(m, Artifact) else run.raw_file(m, name="material")
        parser_mode = "mat" if art.path.suffix == ".mat" else "yaml"
        mat_pairs.append((art.content_id, parser_mode, art.path.name, art))
    # Order-independent request key: the importer treats materials as a set.
    mat_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    inputs: dict[str, Any] = {"mesh": mesh_file}
    settings: dict[str, Any] = {}
    if mat_pairs:
        inputs["materials"] = [item[3] for item in mat_pairs]
        settings["material_formats"] = [item[1] for item in mat_pairs]
    if charges is not None:
        inputs["charges"] = charges

    result = run.run_stage(stage, inputs=inputs, settings=settings, **options)
    assert isinstance(result, Artifact)
    return result


def import_free_charges(
    run: RunContext,
    primaries_file: Artifact | str | os.PathLike[str],
    **options: Any,
) -> Artifact:
    """Run the cached `poisson_import --primaries` stage and return the `FreeCharges` Artifact."""
    stage = run.get_stage("import_free_charges", make_import_free_charges_stage)
    result = run.run_stage(stage, inputs={"primaries": primaries_file}, **options)
    assert isinstance(result, Artifact)
    return result


def adapt_mesh(
    run: RunContext,
    mesh: Artifact | str | os.PathLike[str],
    solution: Artifact | str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
    state: Artifact | str | os.PathLike[str] | None = None,
    **options: Any,
) -> dict[str, Artifact]:
    """Run one cached `poisson_adapt` cycle.

    Returns `{"mesh", "parents", "state", "eta"}` (see `make_adapt_stage`). Chain cycles by
    passing the previous result's `mesh` and `state`; omit `state` for an imported mesh. The
    status (`adapted` / `converged` / `limit_reached` / `unchanged`) is `result["mesh"].info["status"]`.
    """
    stage = run.get_stage("adapt_mesh", make_adapt_stage)
    inputs: dict[str, Any] = {"mesh": mesh, "solution": solution}
    if state is not None:
        inputs["state"] = state
    result = run.run_stage(stage, inputs=inputs, settings=config, **options)
    assert isinstance(result, dict)
    return result
