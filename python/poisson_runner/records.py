"""Tagged JSON encoding of branch outputs and choice values for saved experiment reports.

Plain JSON values pass through unchanged. Everything else is wrapped in a single-key tagged
object so it can be rebuilt on load:

- `Artifact`          -> `{"$artifact": {...descriptor...}}`
- `Path`              -> `{"$path": "..."}` (same tag as `hashing._canonicalize`)
- `tuple`             -> `{"$tuple": [...]}` (lists stay lists)
- non-finite float    -> `{"$float": "nan" | "inf" | "-inf"}`
- registered record   -> `{"$record": "<type name>", "data": {...}}`
- dict with a `$` key -> `{"$map": {...}}` (so user keys never clash with tags)
- anything else       -> `{"$unserializable": "<type name>", "repr": "..."}` (lenient mode only)

Records are rebuilt only through the explicit registry (`register_record`), never by importing
a module named in the file. A record whose type is not registered at load time comes back as
an `UnresolvedRecord` holding the raw data.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from .cache import Artifact, _SAFE_EXT_RE, _validate_hex64

__all__ = [
    "RecordError",
    "UnresolvedRecord",
    "Unserializable",
    "decode",
    "encode",
    "register_record",
    "registered_records",
]

T = TypeVar("T")

_REGISTRY_BY_NAME: dict[str, type] = {}
_REGISTRY_BY_TYPE: dict[type, str] = {}


class RecordError(TypeError):
    """A value cannot be encoded for a saved report (strict mode)."""


@dataclass(frozen=True)
class UnresolvedRecord:
    """A saved record whose type was not registered when the report was loaded."""

    type_name: str
    data: Any


@dataclass(frozen=True)
class Unserializable:
    """Stand-in for a value that could not be saved (e.g. a callable used as a compare() option)."""

    type_name: str
    repr: str = field(default="")


def register_record(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: save instances as `{"$record": name, "data": ...}` and rebuild them on load.

    The class provides `to_dict()` / `from_dict(data)` (classmethod), or is a dataclass (its
    fields are saved and passed back as keyword arguments). Names must be unique.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register_record() needs a non-empty type name")

    def deco(cls: type[T]) -> type[T]:
        existing = _REGISTRY_BY_NAME.get(name)
        # Re-registering the same class (e.g. after a module reload) is fine; another class is not.
        if existing is not None and (existing.__module__, existing.__qualname__) != (cls.__module__, cls.__qualname__):
            raise ValueError(
                f"Record type name {name!r} is already registered for {existing.__module__}.{existing.__qualname__}"
            )
        has_methods = callable(getattr(cls, "to_dict", None)) and callable(getattr(cls, "from_dict", None))
        if not has_methods:
            if not dataclasses.is_dataclass(cls):
                raise TypeError(f"{cls.__qualname__} must be a dataclass or define to_dict() and from_dict()")
            no_init = [f.name for f in dataclasses.fields(cls) if not f.init]
            if no_init:
                raise TypeError(
                    f"{cls.__qualname__} has init=False fields {no_init}; define to_dict()/from_dict() for it"
                )
        _REGISTRY_BY_NAME[name] = cls
        _REGISTRY_BY_TYPE[cls] = name
        return cls

    return deco


def registered_records() -> dict[str, type]:
    """Snapshot of the registry (type name -> class)."""
    return dict(_REGISTRY_BY_NAME)


def _record_data(obj: Any) -> Any:
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _infer_ext(path: Path, content_id: str) -> str | None:
    name = path.name
    if _validate_hex64(content_id) and name.startswith(content_id) and len(name) > len(content_id):
        ext = name[len(content_id) :]
        if _SAFE_EXT_RE.match(ext) and ".." not in ext:
            return ext
    suffix = path.suffix
    if suffix and _SAFE_EXT_RE.match(suffix) and ".." not in suffix:
        return suffix
    return None


def _artifact_descriptor(
    art: Artifact,
    objects_dir: Path | None,
    ephemeral_dir: Path | None = None,
) -> dict[str, Any]:
    desc: dict[str, Any] = {
        "name": art.name,
        "stage": art.stage,
        "content_id": art.content_id,
        "request_key": art.request_key,
        "cached": art.cached,
        "size_bytes": art.size_bytes,
        "path": str(art.path),
        "info": art.info if isinstance(art.info, dict) else {},
        "execution": art.execution if isinstance(art.execution, dict) else {},
    }
    if art.content_id.startswith("dryrun:"):
        desc["location"] = "external"
        return desc

    ext = art.ext
    if objects_dir is not None and art.path.parent == objects_dir:
        location = "cache"
    elif ephemeral_dir is not None and _is_under(art.path, ephemeral_dir):
        location = "ephemeral"
    elif art.stage == "input":
        location = "external"
    elif ext is not None or (_validate_hex64(art.content_id) and art.path.name.startswith(art.content_id)):
        location = "cache"
    else:
        location = "external"

    desc["location"] = location
    if location in ("cache", "ephemeral") and _validate_hex64(art.content_id):
        if ext is None:
            ext = _infer_ext(art.path, art.content_id)
        if ext is not None:
            desc["ext"] = ext
            desc["object"] = f"{art.content_id}{ext}"
    return desc


def encode(
    value: Any,
    *,
    strict: bool,
    objects_dir: Path | None = None,
    ephemeral_dir: Path | None = None,
    where: str = "value",
) -> Any:
    """Encode `value` to tagged JSON. `strict=True` raises `RecordError` on anything unsupported."""

    def enc(v: Any, path: str) -> Any:
        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, float):
            if math.isfinite(v):
                return v
            return {"$float": "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")}
        if isinstance(v, Artifact):
            return {"$artifact": _artifact_descriptor(v, objects_dir, ephemeral_dir)}
        if isinstance(v, Path):
            return {"$path": str(v)}
        if isinstance(v, UnresolvedRecord):
            return {"$record": v.type_name, "data": enc(v.data, f"{path}<{v.type_name}>")}
        if isinstance(v, Unserializable):
            return {"$unserializable": v.type_name, "repr": v.repr}
        record_name = _REGISTRY_BY_TYPE.get(type(v))
        if record_name is not None:
            return {"$record": record_name, "data": enc(_record_data(v), f"{path}<{record_name}>")}
        if isinstance(v, tuple):
            return {"$tuple": [enc(item, f"{path}[{i}]") for i, item in enumerate(v)]}
        if isinstance(v, list):
            return [enc(item, f"{path}[{i}]") for i, item in enumerate(v)]
        if isinstance(v, dict):
            out: dict[str, Any] = {}
            for k, item in v.items():
                if not isinstance(k, str):
                    if strict:
                        raise RecordError(f"{path}: dict keys must be str, got {type(k).__name__} {k!r}")
                    k = str(k)
                    if k in out or k in v:
                        k = f"{k}~{type(item).__name__}"
                out[k] = enc(item, f"{path}.{k}")
            if any(k.startswith("$") for k in out):
                return {"$map": out}
            return out
        if strict:
            raise RecordError(
                f"{path}: cannot save value of type {type(v).__name__}; return Artifacts, JSON scalars, "
                "lists/dicts, Paths, or classes registered with @register_record(...)"
            )
        text = repr(v)
        return {"$unserializable": type(v).__qualname__, "repr": text if len(text) <= 200 else text[:197] + "..."}

    return enc(value, where)


def decode(
    value: Any,
    *,
    objects_dir: Path | None = None,
    bundle_objects_dir: Path | None = None,
    bundled_names: set[str] | frozenset[str] | None = None,
) -> Any:
    """Inverse of `encode`. Resolves bundled objects first, then `objects_dir`, then saved `path`."""

    def dec(v: Any) -> Any:
        if isinstance(v, list):
            return [dec(item) for item in v]
        if not isinstance(v, dict):
            return v
        if len(v) >= 1:
            if "$artifact" in v and len(v) == 1:
                d = v["$artifact"]
                path = Path(d["path"])
                cid = d["content_id"]
                ext = d.get("ext")
                obj = d.get("object")
                inferred_obj = obj
                if inferred_obj is None and d.get("stage") != "input" and _validate_hex64(cid):
                    inferred_ext = ext or _infer_ext(path, cid)
                    if inferred_ext is not None:
                        inferred_obj = f"{cid}{inferred_ext}"
                        if ext is None:
                            ext = inferred_ext
                resolved = False
                if inferred_obj and bundle_objects_dir is not None:
                    if bundled_names is not None:
                        if inferred_obj in bundled_names:
                            path = bundle_objects_dir / inferred_obj
                            resolved = True
                    elif (bundle_objects_dir / inferred_obj).is_file():
                        path = bundle_objects_dir / inferred_obj
                        resolved = True
                if not resolved and obj and objects_dir is not None:
                    path = objects_dir / obj
                return Artifact(
                    path=path,
                    content_id=cid,
                    name=d.get("name", "out"),
                    stage=d.get("stage", "input"),
                    request_key=d.get("request_key"),
                    cached=bool(d.get("cached", False)),
                    size_bytes=int(d.get("size_bytes", 0) or 0),
                    info=d.get("info") or {},
                    execution=d.get("execution") or {},
                    ext=ext,
                )
            if "$path" in v and len(v) == 1:
                return Path(v["$path"])
            if "$tuple" in v and len(v) == 1:
                return tuple(dec(item) for item in v["$tuple"])
            if "$float" in v and len(v) == 1:
                return float(v["$float"])
            if "$map" in v and len(v) == 1:
                return {k: dec(item) for k, item in v["$map"].items()}
            if "$unserializable" in v and set(v) <= {"$unserializable", "repr"}:
                return Unserializable(str(v["$unserializable"]), str(v.get("repr", "")))
            if "$record" in v and set(v) == {"$record", "data"}:
                name = str(v["$record"])
                data = dec(v["data"])
                cls = _REGISTRY_BY_NAME.get(name)
                if cls is None:
                    return UnresolvedRecord(name, data)
                from_dict = getattr(cls, "from_dict", None)
                if callable(from_dict):
                    return from_dict(data)
                return cls(**data)
        return {k: dec(item) for k, item in v.items()}

    return dec(value)
