"""Stage definitions and thin wrappers around per-stage C++ CLI binaries."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import Artifact, Cache, CacheMode, normalize_cache_mode
from .hashing import blake3_json


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
    ) -> Stage:
        """Declaratively register a CLI stage without writing a custom `build_command` lambda.

        - `input_flags`: maps input slot name -> CLI flag (list inputs repeat the flag).
        - `setting_flags`: maps flat scalar setting name -> CLI flag (optional).
        - `config_flag`: if set (e.g. `"--config"`), writes the entire nested `settings` dict
          to a temporary JSON file and passes `[config_flag, <path>]` — ideal for stages with
          many nested component options like AMR or `PoissonDriver`.
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

    def compute_code_hash(self, cache: Cache) -> str:
        """Hash the stage's compiled binary (and any extra code_files / manual version)."""
        parts: dict[str, Any] = {}
        if self.version is not None:
            parts["version"] = self.version
        if not self.ignore_binary_hash:
            parts["files"] = [cache.hash_file(f) for f in self.tracked_code_files()]
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


def make_adapt_stage(
    build_dir: Path | str | None = None,
    *,
    with_parents: bool = False,
    with_history: bool = False,
) -> Stage:
    """Create the Stage for `poisson_adapt --mesh ... --solution ... --config ...`."""
    out_specs: dict[str, str] = {"out": ".h5"}
    out_flags: dict[str, str] = {"out": "--out"}
    if with_parents:
        out_specs["parents"] = ".h5"
        out_flags["parents"] = "--parents-out"
    if with_history:
        out_specs["history"] = ".h5"
        out_flags["history"] = "--history-out"

    return Stage.from_cli(
        name="adapt_mesh",
        binary="poisson_adapt",
        input_flags={"mesh": "--mesh", "solution": "--solution", "history": "--history"},
        config_flag="--config",
        output_flags=out_flags,
        output_specs=out_specs,
        info_flag="--info-out",
        build_dir=build_dir,
        version=f"adapt_mesh_v1_p{int(with_parents)}_h{int(with_history)}",
    )
