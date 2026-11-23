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
        setting_flags={"screening": "--screening"},
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="import_mesh_v2",
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
    screening: bool = True,
    **options: Any,
) -> Artifact:
    """Run the cached `poisson_import --mesh` stage and return the `MeshProperties` Artifact.

    `screening=False` imports the mesh unscreened (`MeshProperties::screened = false`), so the
    screened term is dropped even where a material has a screening length.
    """
    stage = run.get_stage("import_mesh", make_import_mesh_stage)

    mat_pairs = []
    for m in materials:
        art = m if isinstance(m, Artifact) else run.raw_file(m, name="material")
        parser_mode = "mat" if art.path.suffix == ".mat" else "yaml"
        mat_pairs.append((art.content_id, parser_mode, art.path.name, art))
    # Order-independent request key: the importer treats materials as a set.
    mat_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    inputs: dict[str, Any] = {"mesh": mesh_file}
    settings: dict[str, Any] = {"screening": "on" if screening else "off"}
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


# ---------------------------------------------------------------------------------------------
# Verification: case inputs -> gmsh -> import, the exact solution on a mesh, point samples,
# and error norms against a case (src/verification/architecture.md). A case is anything with
# `spec()` (the `{"name", "params"}` JSON the C++ stages read) and `screened`, normally a
# `poisson_runner.Case`.
# ---------------------------------------------------------------------------------------------


def make_case_inputs_stage(build_dir: Path | str | None = None) -> Stage:
    """`poisson_case` input mode: outputs `geometry` (.geo), `materials` (.yaml), `charges` (.cha)."""
    return Stage.from_cli(
        name="case_inputs",
        binary="poisson_case",
        config_flag="--config",
        output_flags={"geometry": "--geo-out", "materials": "--materials-out", "charges": "--charges-out"},
        output_specs={"geometry": ".geo", "materials": ".yaml", "charges": ".cha"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="case_inputs_v1",
    )


def make_exact_field_stage(build_dir: Path | str | None = None) -> Stage:
    """`poisson_case` exact mode: the exact potential at every node of `mesh`, as a NodalField."""
    return Stage.from_cli(
        name="exact_field",
        binary="poisson_case",
        input_flags={"mesh": "--mesh"},
        config_flag="--config",
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="exact_field_v1",
    )


def make_sample_stage(build_dir: Path | str | None = None) -> Stage:
    """`poisson_sample`: a nodal field (and optionally a case's exact solution) at chosen points."""
    return Stage.from_cli(
        name="sample_field",
        binary="poisson_sample",
        input_flags={"mesh": "--mesh", "field": "--field"},
        config_flag="--config",
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="sample_field_v1",
    )


def make_verify_stage(build_dir: Path | str | None = None) -> Stage:
    """`poisson_diff`: error norms of a nodal field against a case (in the info), per-tet/node errors (.h5)."""
    return Stage.from_cli(
        name="verify",
        binary="poisson_diff",
        input_flags={"mesh": "--mesh", "field": "--field"},
        config_flag="--config",
        output_flags={"out": "--out"},
        output_specs={"out": ".h5"},
        info_flag="--info-out",
        build_dir=build_dir,
        version="verify_v1",
    )


GMSH_ALGORITHMS = ("del3d", "front3d", "hxt", "mmg3d", "initial3d")

# .geo commands that read other files. The cache hashes only the .geo itself, so a script
# that loads anything else could be served a stale mesh.
_GEO_EXTERNAL_RE = re.compile(r"\b(Include|Merge|ShapeFromFile|System|SystemCall|OnelabRun|Background\s+Mesh)\b")
_GEO_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _check_self_contained_geo(path: Path) -> None:
    text = _GEO_COMMENT_RE.sub(" ", path.read_text(encoding="utf-8", errors="replace"))
    match = _GEO_EXTERNAL_RE.search(text)
    if match:
        raise ValueError(
            f"{path}: the .geo uses '{match.group(1)}', which reads another file; gmsh_mesh caches by the .geo "
            "text alone, so it only accepts self-contained scripts"
        )


def make_gmsh_stage(build_dir: Path | str | None = None) -> Stage:
    """Mesh a `.geo` script with the system `gmsh` into a Gmsh 2.2 ASCII `.msh` (the importer's format).

    Always first order, single-threaded (`-nt 1`, so the same input gives the same mesh) and
    `-format msh22`. Settings: `h_max`, `h_min`, `h_charge`, `r_fine`, `r_coarse` (set as .geo
    constants with `-setnumber`), `algorithm` (`-algo`), `optimize` (`-optimize`),
    `optimize_netgen` (`-optimize_netgen`), `random_factor` (`-rand`).
    """
    del build_dir   # gmsh is a system binary
    gmsh = shutil.which("gmsh")
    if gmsh is None:
        raise FileNotFoundError("gmsh not found on PATH; it is needed to mesh verification cases")

    def build(call: StageCall) -> list[str]:
        s = call.settings
        cmd = [str(call.executable)]
        for name in ("h_max", "h_min", "h_charge", "r_fine", "r_coarse"):
            if s.get(name) is not None:
                cmd += ["-setnumber", name, repr(float(s[name]))]
        cmd.append(str(call.inputs["geometry"].path))   # type: ignore[union-attr]
        cmd += ["-3", "-format", "msh22", "-order", "1", "-nt", "1"]
        if s.get("algorithm") is not None:
            cmd += ["-algo", str(s["algorithm"])]
        if s.get("optimize"):
            cmd.append("-optimize")
        if s.get("optimize_netgen"):
            cmd.append("-optimize_netgen")
        if s.get("random_factor") is not None:
            cmd += ["-rand", repr(float(s["random_factor"]))]
        cmd += ["-o", str(call.outputs["out"])]
        return cmd

    return Stage(
        name="gmsh",
        build_command=build,
        executable=Path(gmsh).resolve(),
        output_specs={"out": ".msh"},
        version="gmsh_v1",
    )


def case_inputs(run: RunContext, case: Any, **options: Any) -> dict[str, Artifact]:
    """Run `poisson_case` for `case`: returns `{"geometry", "materials", "charges"}`."""
    stage = run.get_stage("case_inputs", make_case_inputs_stage)
    result = run.run_stage(stage, settings={"case": case.spec()}, **options)
    assert isinstance(result, dict)
    return result


def gmsh_mesh(
    run: RunContext,
    geometry: Artifact | str | os.PathLike[str],
    *,
    h_max: float = 1.0,
    h_min: float | None = None,
    h_charge: float | None = None,
    r_fine: float | None = None,
    r_coarse: float | None = None,
    algorithm: str | None = None,
    optimize: bool = False,
    optimize_netgen: bool = False,
    random_factor: float | None = None,
    **options: Any,
) -> Artifact:
    """Mesh a `.geo` with gmsh (see `make_gmsh_stage`); returns the `.msh` Artifact.

    The `.geo` must be self-contained (no `Include`, `Merge`, ...), since only its own text is
    hashed into the cache key.

    `h_charge` > 0 makes the mesh finer around the case's charges: `h_charge` within `r_fine`
    of a charge, growing to `h_max` at `r_coarse` (which must then exceed `r_fine`).
    """
    if not h_max > 0:
        raise ValueError("h_max must be positive")
    if h_charge is not None and h_charge > 0:
        if r_fine is None or r_coarse is None or not 0 <= r_fine < r_coarse:
            raise ValueError("h_charge needs 0 <= r_fine < r_coarse")
        if h_charge > h_max:
            raise ValueError("h_charge must not exceed h_max")
    if algorithm is not None and algorithm not in GMSH_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {GMSH_ALGORITHMS}, not {algorithm!r}")
    geo = geometry.path if isinstance(geometry, Artifact) else Path(geometry)
    if not (isinstance(geometry, Artifact) and geometry.content_id.startswith("dryrun:")):
        _check_self_contained_geo(geo)
    stage = run.get_stage("gmsh", make_gmsh_stage)
    settings: dict[str, Any] = {
        "h_max": h_max,
        "h_min": h_min,
        "h_charge": h_charge,
        "r_fine": r_fine,
        "r_coarse": r_coarse,
        "algorithm": algorithm,
        "optimize": optimize,
        "optimize_netgen": optimize_netgen,
        "random_factor": random_factor,
    }
    settings = {k: v for k, v in settings.items() if v is not None and v is not False}
    result = run.run_stage(stage, inputs={"geometry": geometry}, settings=settings, **options)
    assert isinstance(result, Artifact)
    return result


def case_mesh(run: RunContext, case: Any, **mesh_options: Any) -> Artifact:
    """The imported `MeshProperties` of `case`: `poisson_case` -> gmsh -> `import_mesh`.

    `mesh_options` go to `gmsh_mesh` (`h_max`, `h_charge`, `algorithm`, ...). Screening is
    imported as the case says (`case.screened`).
    """
    inputs = case_inputs(run, case)
    msh = gmsh_mesh(run, inputs["geometry"], **mesh_options)
    charges = inputs["charges"]
    # An empty .cha (no charges) is left out, so the importer does not warn about it. Only a
    # dry-run placeholder has no known size; it is passed on.
    has_charges = charges.content_id.startswith("dryrun:") or charges.size_bytes > 0
    return import_mesh(
        run,
        msh,
        materials=[inputs["materials"]],
        charges=charges if has_charges else None,
        screening=case.screened,
    )


def exact_field(run: RunContext, case: Any, mesh: Artifact | str | os.PathLike[str], **options: Any) -> Artifact:
    """The exact potential of `case` at every node of `mesh`, as a NodalField `.h5`."""
    stage = run.get_stage("exact_field", make_exact_field_stage)
    result = run.run_stage(stage, inputs={"mesh": mesh}, settings={"case": case.spec()}, **options)
    assert isinstance(result, Artifact)
    return result


def sample_field(
    run: RunContext,
    mesh: Artifact | str | os.PathLike[str],
    field: Artifact | str | os.PathLike[str],
    *,
    line: tuple[Sequence[float], Sequence[float], int] | None = None,
    points: Sequence[Sequence[float]] | None = None,
    case: Any = None,
    **options: Any,
) -> Artifact:
    """Values of a nodal `field` at `line=(from, to, n)` or explicit `points`, as PointSamples `.h5`.

    With `case`, the exact solution is evaluated at the same points too (`u_exact`, `grad_exact`).
    """
    if (line is None) == (points is None):
        raise ValueError("give exactly one of line=(from, to, n) and points=[...]")
    settings: dict[str, Any] = {}
    if line is not None:
        start, end, n = line
        settings["line"] = {"from": [float(v) for v in start], "to": [float(v) for v in end], "n": int(n)}
    else:
        settings["points"] = [[float(v) for v in p] for p in points]   # type: ignore[union-attr]
    if case is not None:
        settings["case"] = case.spec()
    stage = run.get_stage("sample_field", make_sample_stage)
    result = run.run_stage(stage, inputs={"mesh": mesh, "field": field}, settings=settings, **options)
    assert isinstance(result, Artifact)
    return result


def verify(
    run: RunContext,
    case: Any,
    mesh: Artifact | str | os.PathLike[str],
    field: Artifact | str | os.PathLike[str],
    *,
    exclusion_radius: float | None = None,
    quadrature_degree: int = 4,
    check_mesh: bool = True,
    **options: Any,
) -> Artifact:
    """Error of a nodal `field` against `case` on `mesh` (`poisson_diff`).

    The norms are in the result's info (`l2_error`, `h1_error`, `max_error`, `l2_exact`, ...;
    CSV columns `verify.<name>`); the `.h5` holds the per-tet and per-node errors. By default
    no error is measured within the case's exclusion radius around each charge.
    """
    settings: dict[str, Any] = {"case": case.spec(), "quadrature_degree": int(quadrature_degree), "check_mesh": bool(check_mesh)}
    if exclusion_radius is not None:
        if exclusion_radius < 0:
            raise ValueError("exclusion_radius must not be negative")
        settings["exclusion_radius"] = float(exclusion_radius)
    stage = run.get_stage("verify", make_verify_stage)
    result = run.run_stage(stage, inputs={"mesh": mesh, "field": field}, settings=settings, **options)
    assert isinstance(result, Artifact)
    return result
