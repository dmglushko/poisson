"""Bundling saved-run artifacts into `<run_dir>/objects/<content_id><ext>` via hard links or copies.

One mechanism solves both local cache eviction (#1) and cluster `/scratch` purges (#4):
- Same filesystem: `os.link` creates a zero-copy hard link. `cache clean --all` only unlinks the
  cache's name; deleting the run folder (`rm -rf results/old_run`) releases the inode automatically.
- Different filesystem (`EXDEV`/`EPERM`): copy from a single open file descriptor and verify BLAKE3.
- Ephemeral outputs (`cache="off"` or `--collect --allow-recompute`) are always materialized before
  `report.json` is committed so a saved run never points into `ephemeral/`.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import uuid
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import Cache, _LINK_UNSUPPORTED_ERRNOS, _SAFE_EXT_RE, _validate_hex64
from .hashing import blake3_file

BUNDLE_MODES = ("link", "link_or_copy", "none")
BUNDLE_INCLUDES = ("outputs", "stages")

_COPY_CHUNK_BYTES = 1024 * 1024
_READONLY_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def format_bytes(num: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{num} B"
        value /= 1024.0
    return f"{num} B"


@dataclass(frozen=True)
class BundleSummary:
    """Outcome of bundling a saved run's artifacts into `<run_dir>/objects/`."""

    linked: int = 0
    copied: int = 0
    already_present: int = 0
    left_in_cache: int = 0
    left_in_cache_bytes: int = 0
    missing: int = 0
    missing_objects: tuple[str, ...] = ()
    linked_bytes: int = 0
    copied_bytes: int = 0


@dataclass(frozen=True)
class RunArtifactStatus:
    """Where a saved run's referenced artifacts currently live."""

    total: int
    bundled: int
    bundled_bytes: int
    cache_only: int
    cache_only_bytes: int
    missing: int
    missing_objects: tuple[str, ...]
    external_missing: tuple[str, ...]
    corrupted: tuple[str, ...]


def validate_bundle_options(bundle: str, include: str) -> None:
    if bundle not in BUNDLE_MODES:
        raise ValueError(f"bundle must be one of {', '.join(BUNDLE_MODES)}, got {bundle!r}")
    if include not in BUNDLE_INCLUDES:
        raise ValueError(f"include must be one of {', '.join(BUNDLE_INCLUDES)}, got {include!r}")


def _walk_artifacts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _walk_artifacts(item)
    elif isinstance(value, dict):
        if "$artifact" in value and len(value) == 1 and isinstance(value["$artifact"], dict):
            yield value["$artifact"]
            return
        for item in value.values():
            yield from _walk_artifacts(item)


def iter_report_artifacts(payload: Mapping[str, Any], *, include: str = "outputs") -> Iterator[dict[str, Any]]:
    """Yield every `$artifact` descriptor in `payload` within `include` (`outputs` or `stages`)."""
    for branch in payload.get("branches") or ():
        if not isinstance(branch, Mapping):
            continue
        yield from _walk_artifacts(branch.get("outputs"))
        yield from _walk_artifacts(branch.get("choice_values"))
        if include == "stages":
            for stage in branch.get("stages") or ():
                if not isinstance(stage, Mapping):
                    continue
                yield from _walk_artifacts(stage.get("outputs"))
                yield from _walk_artifacts(stage.get("settings"))
                yield from _walk_artifacts(stage.get("info"))
                yield from _walk_artifacts(stage.get("execution"))


def descriptor_object_name(desc: Mapping[str, Any]) -> str | None:
    """Return `<content_id><ext>` for a cache/ephemeral artifact descriptor, or None for external files."""
    obj = desc.get("object")
    if isinstance(obj, str) and obj and "/" not in obj and "\\" not in obj and not obj.startswith("."):
        return obj
    if desc.get("location") == "external" or desc.get("stage") == "input":
        return None
    cid = desc.get("content_id")
    if not _validate_hex64(cid):
        return None
    ext = desc.get("ext")
    if not isinstance(ext, str) or not _SAFE_EXT_RE.match(ext) or ".." in ext:
        suffix = Path(str(desc.get("path", ""))).suffix
        ext = suffix if suffix and _SAFE_EXT_RE.match(suffix) and ".." not in suffix else None
    if ext is None:
        return None
    return f"{cid}{ext}"


def _descriptor_is_ephemeral(desc: Mapping[str, Any], cache_root: Path | None) -> bool:
    loc = desc.get("location")
    if loc == "ephemeral":
        return True
    if loc in ("cache", "external"):
        return False
    if cache_root is None or desc.get("stage") == "input":
        return False
    try:
        Path(str(desc.get("path", ""))).relative_to(cache_root / "ephemeral")
        return True
    except ValueError:
        return False


def collect_referenced_objects(payload: Mapping[str, Any]) -> set[str]:
    """All `<content_id><ext>` names referenced anywhere in `payload` (outputs and stages)."""
    names: set[str] = set()
    for desc in iter_report_artifacts(payload, include="stages"):
        obj = descriptor_object_name(desc)
        if obj is not None:
            names.add(obj)
    return names


def list_bundled_names(bundle_objects_dir: Path) -> set[str]:
    """Read `<run_dir>/objects` once into a set, ignoring dot-files (such as in-flight `.tmp-*`)."""
    try:
        entries = os.listdir(bundle_objects_dir)
    except OSError:
        return set()
    return {name for name in entries if not name.startswith(".")}


def _same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def _copy_from_fd_verified(src: Path, dst: Path, *, content_id: str) -> int:
    """Copy `src` from a single open fd into a temp file beside `dst`, verify BLAKE3, and rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp-{uuid.uuid4().hex}"
    try:
        with src.open("rb") as rfp, tmp.open("wb") as wfp:
            while True:
                chunk = rfp.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                wfp.write(chunk)
        actual_cid = blake3_file(tmp)
        if actual_cid != content_id:
            raise RuntimeError(
                f"Content hash mismatch while bundling {dst.name}: expected {content_id}, got {actual_cid}"
            )
        size = tmp.stat().st_size
        try:
            os.chmod(tmp, 0o444)
        except OSError:
            pass
        os.replace(tmp, dst)
        return size
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _materialize_one(
    src: Path,
    dst: Path,
    *,
    content_id: str,
    expected_size: int | None,
    allow_copy: bool,
) -> tuple[str, int]:
    """Place `src` at `dst` via hard link, falling back to verified copy when `allow_copy` is True."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        if _same_inode(src, dst):
            return "already_present", dst.stat().st_size
        try:
            existing_size = dst.stat().st_size
            if expected_size is None or existing_size == expected_size:
                return "already_present", existing_size
        except OSError:
            pass
        # Size mismatch or unreadable existing file: replace atomically.
        tmp = dst.parent / f".tmp-link-{uuid.uuid4().hex}"
        try:
            os.link(src, tmp)
            os.replace(tmp, dst)
            return "linked", dst.stat().st_size
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            if exc.errno not in _LINK_UNSUPPORTED_ERRNOS:
                raise
            if not allow_copy:
                sz = src.stat().st_size if src.is_file() else (expected_size or 0)
                return "left_in_cache", sz
            return "copied", _copy_from_fd_verified(src, dst, content_id=content_id)

    try:
        os.link(src, dst)
        return "linked", dst.stat().st_size
    except FileExistsError:
        if dst.is_file():
            return "already_present", dst.stat().st_size
        raise
    except OSError as exc:
        if exc.errno not in _LINK_UNSUPPORTED_ERRNOS:
            raise
        if not allow_copy:
            sz = src.stat().st_size if src.is_file() else (expected_size or 0)
            return "left_in_cache", sz
        return "copied", _copy_from_fd_verified(src, dst, content_id=content_id)


@contextlib.contextmanager
def _optional_cache_lease(cache_root: Path | None, cache: Cache | None) -> Iterator[Any]:
    """Acquire a lease when a writable cache exists; degrade gracefully on read-only/foreign roots."""
    if cache is not None:
        try:
            with cache.lease() as lease:
                yield lease
            return
        except OSError as exc:
            if exc.errno not in _READONLY_ERRNOS:
                raise
            warnings.warn(
                f"Cache at {cache.root} is not writable for leasing ({exc}); bundling without a lease",
                stacklevel=3,
            )
            yield None
            return
    if cache_root is None or not (cache_root / "objects").is_dir():
        yield None
        return
    try:
        opened = Cache(cache_root)
        with opened.lease() as lease:
            yield lease
    except OSError as exc:
        if exc.errno not in _READONLY_ERRNOS:
            raise
        warnings.warn(
            f"Cache at {cache_root} is not writable for leasing ({exc}); bundling without a lease",
            stacklevel=3,
        )
        yield None


def materialize_ephemeral_artifacts(
    payload: Mapping[str, Any],
    run_dir: Path,
    *,
    include: str = "outputs",
    cache_root: Path | None = None,
    allow_missing: bool = False,
) -> int:
    """Materialize ephemeral artifacts into `<run_dir>/objects/` before committing `report.json`."""
    bundle_objects_dir = run_dir / "objects"
    seen: dict[str, dict[str, Any]] = {}
    for desc in iter_report_artifacts(payload, include=include):
        if not _descriptor_is_ephemeral(desc, cache_root):
            continue
        obj = descriptor_object_name(desc)
        if obj is not None and obj not in seen:
            seen[obj] = desc

    count = 0
    missing_names: list[str] = []
    for obj, desc in seen.items():
        dst = bundle_objects_dir / obj
        expected_size = desc.get("size_bytes")
        expected_size = int(expected_size) if isinstance(expected_size, int) and not isinstance(expected_size, bool) else None
        if dst.is_file() and (expected_size is None or dst.stat().st_size == expected_size):
            continue
        src = Path(str(desc["path"]))
        if not src.is_file():
            if allow_missing:
                missing_names.append(obj)
                continue
            raise FileNotFoundError(
                f"Cannot materialize ephemeral output {obj!r}: source file {src} no longer exists"
            )
        _materialize_one(
            src,
            dst,
            content_id=str(desc["content_id"]),
            expected_size=expected_size,
            allow_copy=True,
        )
        count += 1
    if missing_names:
        warnings.warn(
            f"{len(missing_names)} ephemeral artifact(s) were already missing and could not be bundled: "
            f"{', '.join(missing_names[:5])}",
            stacklevel=2,
        )
    return count


def bundle_report_objects(
    payload: Mapping[str, Any],
    run_dir: Path,
    *,
    mode: str = "link_or_copy",
    include: str = "outputs",
    cache_root: Path | None = None,
    cache: Cache | None = None,
) -> BundleSummary:
    """Bundle cache (and any remaining ephemeral) artifacts of `payload` into `<run_dir>/objects/`."""
    validate_bundle_options(mode, include)
    if mode == "none":
        return BundleSummary()

    bundle_objects_dir = run_dir / "objects"
    cache_objects_dir = cache.objects_dir if cache is not None else (cache_root / "objects" if cache_root else None)

    unique: dict[str, dict[str, Any]] = {}
    for desc in iter_report_artifacts(payload, include=include):
        if desc.get("location") == "external" or desc.get("stage") == "input":
            continue
        obj = descriptor_object_name(desc)
        if obj is not None and obj not in unique:
            unique[obj] = desc

    if not unique:
        return BundleSummary()

    linked = copied = already = left = left_bytes = linked_bytes = copied_bytes = 0
    missing: list[str] = []
    allow_copy = mode == "link_or_copy"

    with _optional_cache_lease(cache_root, cache) as lease:
        if lease is not None and cache_objects_dir is not None:
            lease.pin(list(unique.keys()))
        for obj, desc in unique.items():
            dst = bundle_objects_dir / obj
            expected_size = desc.get("size_bytes")
            expected_size = (
                int(expected_size) if isinstance(expected_size, int) and not isinstance(expected_size, bool) else None
            )
            # Pick source: prefer cache objects_dir, else the recorded path (e.g. another bundle or ephemeral).
            src: Path | None = None
            if cache_objects_dir is not None and (cache_objects_dir / obj).is_file():
                src = cache_objects_dir / obj
            else:
                recorded = Path(str(desc["path"]))
                if recorded.is_file():
                    src = recorded

            if dst.is_file():
                if src is not None and _same_inode(src, dst):
                    already += 1
                    continue
                try:
                    if expected_size is None or dst.stat().st_size == expected_size:
                        already += 1
                        continue
                except OSError:
                    pass

            if src is None:
                missing.append(obj)
                continue

            is_eph = _descriptor_is_ephemeral(desc, cache_root)
            try:
                status, sz = _materialize_one(
                    src,
                    dst,
                    content_id=str(desc["content_id"]),
                    expected_size=expected_size,
                    allow_copy=allow_copy or is_eph,
                )
            except FileNotFoundError:
                missing.append(obj)
                continue

            if status == "already_present":
                already += 1
            elif status == "linked":
                linked += 1
                linked_bytes += sz
            elif status == "copied":
                copied += 1
                copied_bytes += sz
            elif status == "left_in_cache":
                left += 1
                left_bytes += sz

    return BundleSummary(
        linked=linked,
        copied=copied,
        already_present=already,
        left_in_cache=left,
        left_in_cache_bytes=left_bytes,
        missing=len(missing),
        missing_objects=tuple(missing),
        linked_bytes=linked_bytes,
        copied_bytes=copied_bytes,
    )


def prune_unreferenced_objects(run_dir: Path, referenced: set[str]) -> int:
    """Delete files in `<run_dir>/objects/` not listed in `referenced` (including stray `.tmp-*`)."""
    objects_dir = run_dir / "objects"
    if not objects_dir.is_dir():
        return 0
    removed = 0
    for p in list(objects_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name not in referenced:
            p.unlink(missing_ok=True)
            removed += 1
    try:
        objects_dir.rmdir()  # remove if empty
    except OSError:
        pass
    return removed


def _load_saved_payload(directory: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = Path(directory)
    report_json = path / "report.json" if path.is_dir() else path
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    if payload.get("saved_run") is not True or not isinstance(payload.get("format_version"), int):
        raise ValueError(f"{report_json} is not a saved run (use ExperimentReport.save, not save_json)")
    return report_json.parent, payload


def bundle_saved_run(
    directory: str | os.PathLike[str],
    *,
    mode: str = "link_or_copy",
    include: str = "outputs",
    cache_dir: str | os.PathLike[str] | None = None,
) -> BundleSummary:
    """Idempotently bundle a previously saved run's artifacts into `<dir>/objects/`."""
    validate_bundle_options(mode, include)
    run_dir, payload = _load_saved_payload(directory)
    saved_root = payload.get("cache_root")
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else (Path(saved_root) if saved_root else None)
    with (run_dir / ".report.lock").open("a") as lock_fp:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is modifying the saved run at {run_dir}") from None
        except OSError:
            pass
        summary = bundle_report_objects(payload, run_dir, mode=mode, include=include, cache_root=cache_root)
        prune_unreferenced_objects(run_dir, collect_referenced_objects(payload))
    return summary


def saved_run_status(
    directory: str | os.PathLike[str],
    *,
    include: str = "outputs",
    cache_dir: str | os.PathLike[str] | None = None,
    verify: bool = False,
) -> RunArtifactStatus:
    """Inspect where a saved run's artifacts live: bundled, in cache only, or missing."""
    validate_bundle_options("link", include)
    run_dir, payload = _load_saved_payload(directory)
    saved_root = payload.get("cache_root")
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else (Path(saved_root) if saved_root else None)
    cache_objects_dir = cache_root / "objects" if cache_root is not None else None
    bundle_objects_dir = run_dir / "objects"
    bundled_names = list_bundled_names(bundle_objects_dir)

    unique: dict[str, dict[str, Any]] = {}
    external_missing: list[str] = []
    seen_external: set[str] = set()

    for desc in iter_report_artifacts(payload, include=include):
        if desc.get("location") == "external" or desc.get("stage") == "input":
            raw_path = str(desc.get("path", ""))
            if raw_path and raw_path not in seen_external:
                seen_external.add(raw_path)
                if not Path(raw_path).is_file():
                    external_missing.append(raw_path)
            continue
        obj = descriptor_object_name(desc)
        if obj is not None and obj not in unique:
            unique[obj] = desc

    bundled = bundled_bytes = cache_only = cache_only_bytes = 0
    missing: list[str] = []
    corrupted: list[str] = []

    for obj, desc in unique.items():
        cid = str(desc["content_id"])
        if obj in bundled_names:
            p = bundle_objects_dir / obj
            if p.is_file():
                bundled += 1
                bundled_bytes += p.stat().st_size
                if verify and blake3_file(p) != cid:
                    corrupted.append(obj)
                continue
        if cache_objects_dir is not None and (cache_objects_dir / obj).is_file():
            p = cache_objects_dir / obj
            cache_only += 1
            cache_only_bytes += p.stat().st_size
            if verify and blake3_file(p) != cid:
                corrupted.append(obj)
            continue
        recorded = Path(str(desc["path"]))
        if recorded.is_file():
            cache_only += 1
            cache_only_bytes += recorded.stat().st_size
            if verify and blake3_file(recorded) != cid:
                corrupted.append(obj)
            continue
        missing.append(obj)

    return RunArtifactStatus(
        total=len(unique),
        bundled=bundled,
        bundled_bytes=bundled_bytes,
        cache_only=cache_only,
        cache_only_bytes=cache_only_bytes,
        missing=len(missing),
        missing_objects=tuple(missing),
        external_missing=tuple(external_missing),
        corrupted=tuple(corrupted),
    )
