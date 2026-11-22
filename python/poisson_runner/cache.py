"""Two-layer content-addressed stage cache with flock leases, atomic writes, and LRU eviction."""

from __future__ import annotations

import contextlib
import datetime as dt
import errno
from enum import Enum
import fcntl
import json
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .hashing import blake3_file, blake3_json


class LockMode(str, Enum):
    """Cross-process cache locking strategy.

    - `FLOCK`: POSIX `fcntl.flock` (default on local Linux/WSL; auto-falls back to `LOCKFILE`
      if the filesystem reports ENOSYS/EOPNOTSUPP/ENOLCK).
    - `LOCKFILE`: Atomic `O_CREAT | O_EXCL` lockfile + stale-age recovery (safe across nodes
      on Lustre/GPFS/NFSv4 shared `/scratch`).
    - `NONE`: Lock-free reads/writes using unique temp files + atomic `os.replace`. Requires
      `max_bytes=None` (no mid-run eviction); run `cache clean --max-gb` between sweeps.
    """

    FLOCK = "flock"
    LOCKFILE = "lockfile"
    NONE = "none"


def normalize_lock_mode(value: LockMode | str) -> LockMode:
    if isinstance(value, LockMode):
        return value
    if isinstance(value, str):
        try:
            return LockMode(value.strip().lower())
        except ValueError:
            pass
    valid = ", ".join(m.value for m in LockMode)
    raise ValueError(f"Invalid lock mode {value!r}; expected one of: {valid}")


_FLOCK_UNSUPPORTED_ERRNOS = frozenset({errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK})


class CacheMode(str, Enum):
    """2x2 read/write cache policy for stage execution."""

    USE = "use"             # Read cache if present; write cache on miss (default)
    READ_ONLY = "read_only" # Read cache if present; do not write cache on miss
    REFRESH = "refresh"     # Ignore existing cache; force run and overwrite cache
    OFF = "off"             # Do not read or write cache

    @property
    def reads(self) -> bool:
        return self in (CacheMode.USE, CacheMode.READ_ONLY)

    @property
    def writes(self) -> bool:
        return self in (CacheMode.USE, CacheMode.REFRESH)


def normalize_cache_mode(value: CacheMode | str | bool) -> CacheMode:
    """Convert a CacheMode, mode string, or bool (`True` -> USE, `False` -> OFF) to CacheMode."""
    if isinstance(value, CacheMode):
        return value
    if isinstance(value, bool):
        return CacheMode.USE if value else CacheMode.OFF
    if isinstance(value, str):
        try:
            return CacheMode(value)
        except ValueError:
            pass
    valid = ", ".join(m.value for m in CacheMode)
    raise ValueError(f"Invalid cache mode {value!r}; expected one of: {valid} (or True/False)")

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXT_RE = re.compile(r"^\.[A-Za-z0-9._-]{1,32}$")


def _validate_ext(ext: str) -> str:
    if not isinstance(ext, str) or not _SAFE_EXT_RE.match(ext) or ".." in ext:
        raise ValueError(f"Unsafe or invalid output file extension: {ext!r}")
    return ext


def _validate_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def file_signature(path: str | os.PathLike[str]) -> tuple[str, int, int, int, int]:
    """Return `(resolved_path, st_ino, st_size, st_mtime_ns, st_ctime_ns)` for drift detection."""
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@dataclass(frozen=True)
class Artifact:
    """Reference to an immutable file on disk identified by its BLAKE3 content hash."""

    path: Path
    content_id: str
    name: str = "out"
    stage: str = "input"
    request_key: str | None = None
    cached: bool = False
    size_bytes: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheStats:
    """Summary of disk usage and contents in a Cache directory."""

    root: Path
    object_count: int
    request_count: int
    object_bytes: int
    metadata_bytes: int
    temp_count: int
    max_bytes: int | None
    ephemeral_count: int = 0
    ephemeral_bytes: int = 0


class CacheLease:
    """Cross-process lease protecting active cache objects from eviction."""

    def __init__(self, cache: Cache) -> None:
        self._cache = cache
        self._host = socket.gethostname().replace("/", "_")
        self._path = cache.leases_dir / f"{self._host}_{os.getpid()}_{uuid.uuid4().hex}.lease"
        self._fp: Any = None
        self._active = False
        self._uses_flock = False
        self._object_names: set[str] = set()
        self._stop_heartbeat: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def _start_heartbeat(self) -> None:
        interval = max(0.05, self._cache.lock_stale_s / 3.0)
        stop_ev = threading.Event()
        self._stop_heartbeat = stop_ev

        def _beat() -> None:
            while not stop_ev.wait(interval):
                try:
                    now = time.time()
                    os.utime(self._path, (now, now))
                except OSError:
                    break

        t = threading.Thread(target=_beat, name="poisson-cache-lease-heartbeat", daemon=True)
        self._heartbeat_thread = t
        t.start()

    def __enter__(self) -> CacheLease:
        if self._cache.lock_mode == LockMode.NONE:
            return self
        with self._cache._locked():
            self._cache.leases_dir.mkdir(parents=True, exist_ok=True)
            self._fp = open(self._path, "w+", encoding="utf-8")
            if self._cache.lock_mode == LockMode.FLOCK:
                try:
                    fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
                    self._uses_flock = True
                except OSError as exc:
                    if exc.errno not in _FLOCK_UNSUPPORTED_ERRNOS:
                        self._fp.close()
                        self._fp = None
                        self._path.unlink(missing_ok=True)
                        raise
                    self._uses_flock = False
            self._active = True
            self._sync()
        if not self._uses_flock:
            self._start_heartbeat()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def pin(self, object_names: Iterable[str], *, _locked: bool = False) -> None:
        changed = False
        for name in object_names:
            if name not in self._object_names:
                self._object_names.add(name)
                changed = True
        if changed and self._active and self._fp is not None:
            if _locked:
                self._sync()
            else:
                with self._cache._locked():
                    self._sync()

    @property
    def object_names(self) -> frozenset[str]:
        return frozenset(self._object_names)

    def _sync(self) -> None:
        assert self._fp is not None
        self._fp.seek(0)
        self._fp.truncate()
        json.dump(
            {
                "host": self._host,
                "pid": os.getpid(),
                "mode": "flock" if self._uses_flock else "lockfile",
                "objects": sorted(self._object_names),
            },
            self._fp,
        )
        self._fp.flush()

    def close(self) -> None:
        self._active = False
        if self._stop_heartbeat is not None:
            self._stop_heartbeat.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=1.0)
            self._stop_heartbeat = None
            self._heartbeat_thread = None
        if self._fp is not None:
            try:
                self._path.unlink(missing_ok=True)
            finally:
                try:
                    if self._uses_flock:
                        fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
                finally:
                    self._fp.close()
                    self._fp = None
                    self._uses_flock = False


class Cache:
    """Content-addressed disk cache backed by BLAKE3.

    Layout:
      <root>/objects/<content_id><ext>   -- deterministic data files (deduplicated by content)
      <root>/requests/<request_key>.json -- provenance, output content IDs, stage Info & metrics
      <root>/tmp/<host>_<pid>_<uuid>...  -- in-progress staging files (atomically moved on success)
      <root>/ephemeral/...               -- outputs of `use_cache=False` runs (never pruned as tmp)
      <root>/leases/...                  -- live artifact leases across processes
      <root>/.cache.lock                 -- cross-process lock coordinating lookup/publish/eviction
    """

    def __init__(
        self,
        root: Path | str | None = None,
        max_bytes: int | None = None,
        *,
        verify_hashes: bool = False,
        lock_mode: LockMode | str | None = None,
        lock_timeout_s: float = 60.0,
        lock_stale_s: float = 300.0,
    ) -> None:
        if root is None:
            root = os.environ.get("POISSON_CACHE_DIR", ".poisson_cache")
        self.root = Path(root).resolve()

        if max_bytes is None:
            env_gb = os.environ.get("POISSON_CACHE_MAX_GB")
            if env_gb:
                max_bytes = int(float(env_gb) * (1024**3))
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative or None")
        self.max_bytes = max_bytes
        self.verify_hashes = verify_hashes

        self._lock_mode_marker = self.root / ".lock_mode"
        explicit_lock_mode = lock_mode is not None
        if lock_mode is None:
            env_lock = os.environ.get("POISSON_CACHE_LOCK_MODE")
            if env_lock:
                self.lock_mode = normalize_lock_mode(env_lock)
                explicit_lock_mode = True
            elif self._lock_mode_marker.is_file():
                try:
                    self.lock_mode = normalize_lock_mode(self._lock_mode_marker.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self.lock_mode = LockMode.FLOCK
            else:
                self.lock_mode = LockMode.FLOCK
        else:
            self.lock_mode = normalize_lock_mode(lock_mode)

        if self.lock_mode == LockMode.NONE and self.max_bytes is not None:
            raise ValueError(
                "max_bytes auto-eviction cannot be combined with lock_mode='none'; "
                "use lock_mode='lockfile' (or 'flock') or run `cache clean --max-gb` between sweeps"
            )
        if lock_timeout_s <= 0:
            raise ValueError("lock_timeout_s must be positive")
        if lock_stale_s <= 0:
            raise ValueError("lock_stale_s must be positive")
        self.lock_timeout_s = float(lock_timeout_s)
        self.lock_stale_s = float(lock_stale_s)

        self.objects_dir = self.root / "objects"
        self.requests_dir = self.root / "requests"
        self.tmp_dir = self.root / "tmp"
        self.ephemeral_dir = self.root / "ephemeral"
        self.leases_dir = self.root / "leases"
        self._lock_path = self.root / ".cache.lock"
        self._excl_lock_path = self.root / ".cache.lock.excl"
        for d in (self.objects_dir, self.requests_dir, self.tmp_dir, self.ephemeral_dir, self.leases_dir):
            d.mkdir(parents=True, exist_ok=True)

        if explicit_lock_mode or not self._lock_mode_marker.is_file():
            try:
                tmp_marker = self.make_temp_path(".lock_mode")
                tmp_marker.write_text(self.lock_mode.value + "\n", encoding="utf-8")
                os.replace(tmp_marker, self._lock_mode_marker)
            except OSError:
                pass

        # In-process memoization keyed by (resolved_path, st_ino, st_size, st_mtime_ns, st_ctime_ns).
        self._stat_hash_memo: dict[tuple[str, int, int, int, int], str] = {}

    def _try_break_stale_lockfile(self) -> bool:
        """Atomically reclaim an abandoned `.cache.lock.excl` if its owner is dead (same host) or older than `lock_stale_s`."""
        try:
            st = self._excl_lock_path.stat()
        except FileNotFoundError:
            return True
        except OSError:
            return False

        now = time.time()
        local_host = socket.gethostname().replace("/", "_")
        stale = (now - st.st_mtime) >= self.lock_stale_s
        if not stale:
            try:
                meta = json.loads(self._excl_lock_path.read_text(encoding="utf-8"))
                if (
                    isinstance(meta, dict)
                    and meta.get("host") == local_host
                    and isinstance(meta.get("pid"), int)
                    and not _pid_is_alive(meta["pid"])
                ):
                    stale = True
            except (OSError, json.JSONDecodeError):
                pass
        if stale:
            # Rename to a unique path first so only one waiter reclaims this stale lockfile.
            reclaim_path = self.root / f".cache.lock.stale.{uuid.uuid4().hex}"
            try:
                os.rename(self._excl_lock_path, reclaim_path)
            except FileNotFoundError:
                return True
            except OSError:
                return False
            reclaim_path.unlink(missing_ok=True)
            return True
        return False

    @contextlib.contextmanager
    def _locked_lockfile(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        payload = json.dumps(
            {
                "host": socket.gethostname().replace("/", "_"),
                "pid": os.getpid(),
                "token": token,
            }
        ).encode("utf-8")

        deadline = time.monotonic() + self.lock_timeout_s
        backoff = 0.005
        acquired = False
        while True:
            try:
                fd = os.open(str(self._excl_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if self._try_break_stale_lockfile():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for cache lockfile {self._excl_lock_path} after {self.lock_timeout_s:.1f}s"
                    )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 0.2)
            else:
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                acquired = True
                break

        stop_beat = threading.Event()
        beat_interval = max(0.05, self.lock_stale_s / 3.0)

        def _beat_lock() -> None:
            while not stop_beat.wait(beat_interval):
                try:
                    now = time.time()
                    os.utime(self._excl_lock_path, (now, now))
                except OSError:
                    break

        beat_thread = threading.Thread(target=_beat_lock, name="poisson-cache-lock-heartbeat", daemon=True)
        beat_thread.start()
        try:
            yield
        finally:
            stop_beat.set()
            beat_thread.join(timeout=1.0)
            if acquired:
                try:
                    meta = json.loads(self._excl_lock_path.read_text(encoding="utf-8"))
                    if isinstance(meta, dict) and meta.get("token") == token:
                        self._excl_lock_path.unlink(missing_ok=True)
                except (OSError, json.JSONDecodeError):
                    pass

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if self.lock_mode == LockMode.NONE:
            yield
            return
        if self.lock_mode == LockMode.LOCKFILE:
            with self._locked_lockfile():
                yield
            return

        self.root.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+", encoding="utf-8") as lock_fp:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                if exc.errno in _FLOCK_UNSUPPORTED_ERRNOS:
                    # Fallback transparently on filesystems without flock support (e.g. Lustre without -o flock).
                    with self._locked_lockfile():
                        yield
                    return
                raise
            try:
                yield
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def lease(self) -> CacheLease:
        """Create a cross-process flock lease protecting artifacts used during an active run."""
        return CacheLease(self)

    def hash_file(self, path: str | os.PathLike[str]) -> str:
        """Hash a file with BLAKE3, memoized by `(path, inode, size, mtime_ns, ctime_ns)`."""
        sig = file_signature(path)
        cached = self._stat_hash_memo.get(sig)
        if cached is not None:
            return cached
        digest = blake3_file(sig[0])
        # Re-verify file didn't change while hashing before memoizing.
        if file_signature(sig[0]) == sig:
            self._stat_hash_memo[sig] = digest
        return digest

    def raw_file(self, path: str | os.PathLike[str], *, name: str = "file") -> Artifact:
        """Wrap an external input file as an Artifact identified by its BLAKE3 content hash."""
        p = Path(path).resolve()
        st = p.stat()
        digest = self.hash_file(p)
        return Artifact(
            path=p,
            content_id=digest,
            name=name,
            stage="input",
            request_key=None,
            cached=True,
            size_bytes=st.st_size,
        )

    def make_temp_path(self, suffix: str = "") -> Path:
        """Allocate a collision-free temporary staging path inside `<root>/tmp`."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname().replace("/", "_")
        return self.tmp_dir / f"{host}_{os.getpid()}_{uuid.uuid4().hex}{suffix}"

    def make_ephemeral_path(self, suffix: str = "") -> Path:
        """Allocate an untracked output path inside `<root>/ephemeral` (for `use_cache=False`)."""
        self.ephemeral_dir.mkdir(parents=True, exist_ok=True)
        return self.ephemeral_dir / f"{os.getpid()}_{uuid.uuid4().hex}{suffix}"

    @staticmethod
    def compute_request_key(
        stage_name: str,
        code_hash: str,
        inputs: dict[str, Any],
        settings: dict[str, Any],
        output_specs: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Compute the BLAKE3 request key for a stage invocation."""
        payload: dict[str, Any] = {
            "v": 2,
            "stage": stage_name,
            "code_hash": code_hash,
            "inputs": inputs,
            "settings": settings,
            "output_specs": dict(output_specs) if output_specs is not None else {"out": ".h5"},
        }
        if env:
            payload["env"] = dict(env)
        return blake3_json(payload)

    def _request_path(self, request_key: str) -> Path:
        return self.requests_dir / f"{request_key}.json"

    def _object_path(self, content_id: str, ext: str) -> Path:
        _validate_ext(ext)
        if not _validate_hex64(content_id):
            raise ValueError(f"Invalid content_id: {content_id!r}")
        return self.objects_dir / f"{content_id}{ext}"

    @staticmethod
    def _touch(path: Path) -> None:
        try:
            now = time.time()
            os.utime(path, (now, now))
        except OSError:
            pass

    def _active_leased_objects(self) -> tuple[set[str], bool]:
        """Inspect `<root>/leases` and collect object filenames held by live leases.

        Returns `(leased_names, all_readable)`. If any live lease cannot be parsed,
        `all_readable` is False so callers can conservatively skip eviction.
        """
        leased: set[str] = set()
        all_readable = True
        if not self.leases_dir.is_dir():
            return leased, all_readable
        local_host = socket.gethostname().replace("/", "_")
        now = time.time()
        for lease_path in list(self.leases_dir.glob("*.lease")):
            try:
                fp = open(lease_path, "r+", encoding="utf-8")
            except OSError:
                continue
            try:
                data: Any = None
                try:
                    fp.seek(0)
                    data = json.load(fp)
                except (OSError, json.JSONDecodeError):
                    data = None

                lease_mode = data.get("mode", "flock") if isinstance(data, dict) else "flock"
                if self.lock_mode == LockMode.FLOCK and lease_mode == "flock":
                    try:
                        # If we can acquire LOCK_EX non-blockingly, the owning process is gone -> stale lease.
                        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        # Live flock lease held by an active process.
                        if isinstance(data, dict) and isinstance(data.get("objects"), list):
                            for item in data["objects"]:
                                if isinstance(item, str):
                                    leased.add(item)
                        else:
                            all_readable = False
                        continue
                    except OSError as exc:
                        if exc.errno in _FLOCK_UNSUPPORTED_ERRNOS:
                            lease_mode = "lockfile"
                        else:
                            all_readable = False
                            continue
                    else:
                        # Stale lease from a crashed process -> remove it.
                        lease_path.unlink(missing_ok=True)
                        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
                        continue

                if self.lock_mode != LockMode.FLOCK or lease_mode != "flock":
                    # Lockfile / non-flock lease: live if same-host PID is alive or file is younger than lock_stale_s.
                    try:
                        st = lease_path.stat()
                    except OSError:
                        continue
                    is_live = False
                    if isinstance(data, dict) and data.get("host") == local_host and isinstance(data.get("pid"), int):
                        is_live = _pid_is_alive(data["pid"])
                    elif (now - st.st_mtime) < self.lock_stale_s:
                        is_live = True

                    if is_live:
                        if isinstance(data, dict) and isinstance(data.get("objects"), list):
                            for item in data["objects"]:
                                if isinstance(item, str):
                                    leased.add(item)
                        else:
                            all_readable = False
                    else:
                        lease_path.unlink(missing_ok=True)
            finally:
                fp.close()
        return leased, all_readable

    def lookup(
        self,
        request_key: str,
        *,
        expected_outputs: Mapping[str, str] | None = None,
        lease: CacheLease | None = None,
    ) -> dict[str, Artifact] | None:
        """Look up a cached request. Validates metadata shape, expected outputs, size, and optional digest."""
        with self._locked():
            req_path = self._request_path(request_key)
            if not req_path.is_file():
                return None

            try:
                record = json.loads(req_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                req_path.unlink(missing_ok=True)
                return None

            if not isinstance(record, dict):
                req_path.unlink(missing_ok=True)
                return None

            outputs_meta = record.get("outputs")
            if not isinstance(outputs_meta, dict) or not outputs_meta:
                req_path.unlink(missing_ok=True)
                return None

            if expected_outputs is not None and set(outputs_meta.keys()) != set(expected_outputs.keys()):
                return None

            info = record.get("info")
            execution = record.get("execution")
            if not isinstance(info, dict):
                info = {}
            if not isinstance(execution, dict):
                execution = {}
            stage_name = str(record.get("stage", ""))

            artifacts: dict[str, Artifact] = {}
            leased_names: list[str] = []

            for out_name, meta in outputs_meta.items():
                if not isinstance(out_name, str) or not isinstance(meta, dict):
                    req_path.unlink(missing_ok=True)
                    return None
                content_id = meta.get("content_id")
                ext = meta.get("ext")
                recorded_size = meta.get("size_bytes")
                if not _validate_hex64(content_id) or not isinstance(ext, str):
                    req_path.unlink(missing_ok=True)
                    return None
                try:
                    _validate_ext(ext)
                except ValueError:
                    req_path.unlink(missing_ok=True)
                    return None
                if expected_outputs is not None and expected_outputs.get(out_name) != ext:
                    return None
                if not isinstance(recorded_size, int) or isinstance(recorded_size, bool) or recorded_size < 0:
                    req_path.unlink(missing_ok=True)
                    return None

                obj_path = self._object_path(content_id, ext)
                if not obj_path.is_file():
                    req_path.unlink(missing_ok=True)
                    return None

                actual_size = obj_path.stat().st_size
                if actual_size != recorded_size:
                    # Truncated or damaged cached object -> purge both object and stale request.
                    obj_path.unlink(missing_ok=True)
                    req_path.unlink(missing_ok=True)
                    return None

                if self.verify_hashes:
                    if blake3_file(obj_path) != content_id:
                        obj_path.unlink(missing_ok=True)
                        req_path.unlink(missing_ok=True)
                        return None

                leased_names.append(obj_path.name)
                artifacts[out_name] = Artifact(
                    path=obj_path,
                    content_id=content_id,
                    name=out_name,
                    stage=stage_name,
                    request_key=request_key,
                    cached=True,
                    size_bytes=actual_size,
                    info=info,
                    execution=execution,
                )

            if lease is not None:
                lease.pin(leased_names, _locked=True)

            self._touch(req_path)
            for art in artifacts.values():
                self._touch(art.path)

            return artifacts

    def store(
        self,
        *,
        request_key: str,
        stage_name: str,
        code_hash: str,
        inputs_spec: dict[str, Any],
        settings: dict[str, Any],
        temp_outputs: dict[str, tuple[Path, str]],
        info: dict[str, Any],
        execution: dict[str, Any],
        lease: CacheLease | None = None,
    ) -> dict[str, Artifact]:
        """Atomically publish stage outputs under lock, pin them in `lease`, and enforce LRU cap."""
        # Hash outputs before acquiring the cache lock so slow disk reads don't stall other processes.
        hashed_outputs: dict[str, tuple[Path, str, str]] = {}
        for out_name, (tmp_path, ext) in temp_outputs.items():
            _validate_ext(ext)
            if not tmp_path.is_file():
                raise RuntimeError(f"Stage '{stage_name}' did not produce expected output '{out_name}' at {tmp_path}")
            content_id = blake3_file(tmp_path)
            hashed_outputs[out_name] = (tmp_path, ext, content_id)

        with self._locked():
            outputs_meta: dict[str, dict[str, Any]] = {}
            artifacts: dict[str, Artifact] = {}
            newly_stored_names: set[str] = set()

            for out_name, (tmp_path, ext, content_id) in hashed_outputs.items():
                obj_path = self._object_path(content_id, ext)
                newly_stored_names.add(obj_path.name)
                os.replace(tmp_path, obj_path)
                try:
                    os.chmod(obj_path, 0o444)
                except OSError:
                    pass
                self._touch(obj_path)
                size_bytes = obj_path.stat().st_size
                outputs_meta[out_name] = {
                    "content_id": content_id,
                    "ext": ext,
                    "size_bytes": size_bytes,
                }
                artifacts[out_name] = Artifact(
                    path=obj_path,
                    content_id=content_id,
                    name=out_name,
                    stage=stage_name,
                    request_key=request_key,
                    cached=False,
                    size_bytes=size_bytes,
                    info=info,
                    execution=execution,
                )

            if lease is not None:
                lease.pin(newly_stored_names, _locked=True)

            record = {
                "v": 2,
                "request_key": request_key,
                "stage": stage_name,
                "code_hash": code_hash,
                "inputs": inputs_spec,
                "settings": settings,
                "outputs": outputs_meta,
                "info": info,
                "execution": {
                    **execution,
                    "stored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            }
            req_tmp = self.make_temp_path(".json")
            req_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(req_tmp, self._request_path(request_key))

            if self.max_bytes is not None:
                self._evict_lru_unlocked(max_bytes=self.max_bytes, extra_protected_names=newly_stored_names)

            return artifacts

    def _evict_lru_unlocked(
        self,
        *,
        max_bytes: int,
        extra_protected_names: Iterable[str] = (),
    ) -> int:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        active_leased, all_readable = self._active_leased_objects()
        if not all_readable:
            return 0
        protected = set(extra_protected_names) | active_leased
        entries: list[tuple[float, int, str, Path]] = []
        total_bytes = 0
        for p in self.objects_dir.iterdir():
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            total_bytes += st.st_size
            entries.append((st.st_mtime, st.st_size, p.name, p))

        if total_bytes <= max_bytes:
            return 0

        entries.sort(key=lambda item: item[0])
        evicted_names: set[str] = set()
        for _, size, obj_name, p in entries:
            if total_bytes <= max_bytes:
                break
            if obj_name in protected:
                continue
            try:
                p.unlink()
            except OSError:
                continue
            total_bytes -= size
            evicted_names.add(obj_name)

        if evicted_names:
            self._prune_stale_requests_unlocked(evicted_names)

        return len(evicted_names)

    def evict_lru(
        self,
        *,
        max_bytes: int,
        protected_object_names: Iterable[str] = (),
    ) -> int:
        """Evict least-recently-used unleased objects until total bytes <= max_bytes."""
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        with self._locked():
            return self._evict_lru_unlocked(max_bytes=max_bytes, extra_protected_names=protected_object_names)

    def _prune_stale_requests_unlocked(self, missing_object_names: set[str] | None = None) -> int:
        removed = 0
        for req_path in self.requests_dir.glob("*.json"):
            try:
                record = json.loads(req_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                req_path.unlink(missing_ok=True)
                removed += 1
                continue

            if not isinstance(record, dict):
                req_path.unlink(missing_ok=True)
                removed += 1
                continue

            outputs = record.get("outputs")
            if not isinstance(outputs, dict) or not outputs:
                req_path.unlink(missing_ok=True)
                removed += 1
                continue

            stale = False
            for meta in outputs.values():
                if not isinstance(meta, dict):
                    stale = True
                    break
                cid = meta.get("content_id")
                ext = meta.get("ext")
                if not _validate_hex64(cid) or not isinstance(ext, str):
                    stale = True
                    break
                try:
                    _validate_ext(ext)
                except ValueError:
                    stale = True
                    break
                obj_name = f"{cid}{ext}"
                if missing_object_names is not None:
                    if obj_name in missing_object_names:
                        stale = True
                        break
                elif not (self.objects_dir / obj_name).is_file():
                    stale = True
                    break

            if stale:
                req_path.unlink(missing_ok=True)
                removed += 1
        return removed

    def _clean_abandoned_tmp_unlocked(self, *, force: bool = False) -> int:
        removed = 0
        local_host = socket.gethostname().replace("/", "_")
        now = time.time()
        for p in self.tmp_dir.iterdir():
            if not p.is_file():
                continue
            if not force:
                parts = p.name.split("_", 2)
                if len(parts) >= 2 and parts[0] == local_host and parts[1].isdigit():
                    if _pid_is_alive(int(parts[1])):
                        continue
                else:
                    # Unknown host or naming: only prune temp files older than 1 hour.
                    try:
                        if now - p.stat().st_mtime < 3600:
                            continue
                    except OSError:
                        continue
            p.unlink(missing_ok=True)
            removed += 1
        return removed

    def _clean_ephemeral_unlocked(self, *, force: bool = False) -> int:
        removed = 0
        now = time.time()
        if not self.ephemeral_dir.is_dir():
            return 0
        for p in self.ephemeral_dir.iterdir():
            if not p.is_file():
                continue
            if not force:
                parts = p.name.split("_", 1)
                if parts and parts[0].isdigit():
                    if _pid_is_alive(int(parts[0])):
                        continue
                else:
                    try:
                        if now - p.stat().st_mtime < 3600:
                            continue
                    except OSError:
                        continue
            p.unlink(missing_ok=True)
            removed += 1
        return removed

    def stats(self) -> CacheStats:
        object_count = 0
        object_bytes = 0
        for p in self.objects_dir.iterdir():
            if not p.is_file():
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            object_count += 1
            object_bytes += sz

        request_count = 0
        metadata_bytes = 0
        for p in self.requests_dir.glob("*.json"):
            if not p.is_file():
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            request_count += 1
            metadata_bytes += sz

        temp_count = sum(1 for p in self.tmp_dir.iterdir() if p.is_file())
        ephemeral_count = 0
        ephemeral_bytes = 0
        if self.ephemeral_dir.is_dir():
            for p in self.ephemeral_dir.iterdir():
                if not p.is_file():
                    continue
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                ephemeral_count += 1
                ephemeral_bytes += sz
        return CacheStats(
            root=self.root,
            object_count=object_count,
            request_count=request_count,
            object_bytes=object_bytes,
            metadata_bytes=metadata_bytes,
            temp_count=temp_count,
            max_bytes=self.max_bytes,
            ephemeral_count=ephemeral_count,
            ephemeral_bytes=ephemeral_bytes,
        )

    def clean(self, *, max_bytes: int | None = None, all_entries: bool = False) -> dict[str, int]:
        """Clean abandoned temp/ephemeral files and either wipe unleased entries or trim via LRU."""
        if max_bytes is not None and max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        with self._locked():
            removed_tmp = self._clean_abandoned_tmp_unlocked(force=all_entries)
            removed_ephemeral = self._clean_ephemeral_unlocked(force=all_entries)

            if all_entries:
                protected, all_readable = self._active_leased_objects()
                removed_objects = 0
                if all_readable:
                    for p in self.objects_dir.iterdir():
                        if p.is_file() and p.name not in protected:
                            p.unlink(missing_ok=True)
                            removed_objects += 1
                removed_requests = self._prune_stale_requests_unlocked()
                if all_readable and not protected:
                    for p in self.requests_dir.glob("*.json"):
                        p.unlink(missing_ok=True)
                        removed_requests += 1
                self._stat_hash_memo.clear()
                return {
                    "removed_objects": removed_objects,
                    "removed_requests": removed_requests,
                    "removed_tmp": removed_tmp,
                    "removed_ephemeral": removed_ephemeral,
                }

            removed_objects = 0
            if max_bytes is not None:
                removed_objects = self._evict_lru_unlocked(max_bytes=max_bytes)
            removed_requests = self._prune_stale_requests_unlocked()
            return {
                "removed_objects": removed_objects,
                "removed_requests": removed_requests,
                "removed_tmp": removed_tmp,
                "removed_ephemeral": removed_ephemeral,
            }

    def inspect(self, prefix: str = "") -> list[dict[str, Any]]:
        """List request metadata records whose request_key or output content_id starts with `prefix`."""
        matches: list[dict[str, Any]] = []
        for req_path in sorted(self.requests_dir.glob("*.json")):
            try:
                record = json.loads(req_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            if not prefix:
                matches.append(record)
                continue
            rk = str(record.get("request_key", ""))
            outputs = record.get("outputs")
            out_cids = (
                [str(m.get("content_id", "")) for m in outputs.values() if isinstance(m, dict)]
                if isinstance(outputs, dict)
                else []
            )
            if rk.startswith(prefix) or any(cid.startswith(prefix) for cid in out_cids):
                matches.append(record)
        return matches
