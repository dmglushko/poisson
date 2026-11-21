"""Fast BLAKE3 hashing backed by the vendored libpoisson_blake3 shared library."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
from typing import Any

_LIB: ctypes.CDLL | None = None


def _candidate_library_paths() -> list[Path]:
    names = ("libpoisson_blake3.so", "libpoisson_blake3.dylib", "poisson_blake3.dll")
    candidates: list[Path] = []

    explicit = os.environ.get("POISSON_BLAKE3_LIB")
    if explicit:
        candidates.append(Path(explicit))

    build_dir = os.environ.get("POISSON_BUILD_DIR")
    if build_dir:
        for name in names:
            candidates.append(Path(build_dir) / name)

    repo_root = Path(__file__).resolve().parents[2]
    for subdir in ("build", "build/Release", "build/Debug"):
        for name in names:
            candidates.append(repo_root / subdir / name)

    return candidates


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    for path in _candidate_library_paths():
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            break
    else:
        found = ctypes.util.find_library("poisson_blake3")
        if found:
            lib = ctypes.CDLL(found)
        else:
            searched = ", ".join(str(p) for p in _candidate_library_paths()[:6])
            raise RuntimeError(
                "Could not locate libpoisson_blake3 shared library. "
                "Build the CMake project first or set POISSON_BUILD_DIR / POISSON_BLAKE3_LIB. "
                f"Searched: {searched}"
            )

    lib.poisson_blake3_bytes.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    lib.poisson_blake3_bytes.restype = None

    lib.poisson_blake3_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.poisson_blake3_file.restype = ctypes.c_int

    lib.poisson_blake3_version.argtypes = []
    lib.poisson_blake3_version.restype = ctypes.c_char_p

    _LIB = lib
    return lib


def blake3_version() -> str:
    """Return the vendored BLAKE3 library version string."""
    lib = _load_library()
    return lib.poisson_blake3_version().decode("ascii")


def blake3_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the 64-character lowercase hex BLAKE3 digest of `data`."""
    lib = _load_library()
    raw = bytes(data)
    out = ctypes.create_string_buffer(32)
    lib.poisson_blake3_bytes(raw, len(raw), out)
    return out.raw.hex()


def blake3_file(path: str | os.PathLike[str]) -> str:
    """Stream-hash a file with BLAKE3 inside C (SIMD-accelerated) and return 64-char hex."""
    lib = _load_library()
    p = Path(path)
    out = ctypes.create_string_buffer(32)
    err = lib.poisson_blake3_file(os.fsencode(p), out)
    if err != 0:
        msg = os.strerror(err) if err > 0 else "I/O error"
        raise OSError(err, f"BLAKE3 failed to hash {p}: {msg}", str(p))
    return out.raw.hex()


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot deterministically hash non-finite float: {value!r}")
        return 0.0 if value == 0.0 else value   # normalize -0.0 to 0.0
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
            if not isinstance(k, str):
                raise TypeError(
                    f"Canonical JSON requires string dict keys to avoid key collisions, got {type(k).__name__}: {k!r}"
                )
            result[k] = _canonicalize(v)
        return result
    raise TypeError(f"Value of type {type(value).__name__} is not canonical JSON-serializable: {value!r}")


def canonical_json(value: Any) -> str:
    """Serialize `value` to canonical deterministic JSON."""
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def blake3_json(value: Any) -> str:
    """Return the BLAKE3 hex digest of the canonical JSON representation of `value`."""
    return blake3_bytes(canonical_json(value).encode("utf-8"))
