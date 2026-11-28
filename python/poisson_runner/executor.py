"""Pluggable stage execution backends (local subprocess now, SLURM-ready interface)."""

from __future__ import annotations

import abc
import functools
import os
import platform
import socket
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)

_SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK",
    "SLURM_NNODES",
    "SLURM_NODELIST",
)


@functools.lru_cache(maxsize=1)
def detect_cpu_model() -> str | None:
    """Detect host CPU model name from `/proc/cpuinfo` (Linux) or `platform.processor()`."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in ("model name", "hardware", "cpu model"):
                    val = value.strip()
                    if val:
                        return val
        except OSError:
            pass
    proc = platform.processor().strip()
    return proc or None


def _usable_cpu_count() -> int | None:
    """CPUs this process may run on: the SLURM/cgroup allocation, not the whole node."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def collect_host_environment(env_override: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Capture CPU, threading, and SLURM metadata for reproducible HPC benchmarking."""
    merged = os.environ.copy()
    if env_override:
        merged.update(env_override)

    meta: dict[str, Any] = {
        "cpu_model": detect_cpu_model(),
        "cpu_count": _usable_cpu_count(),
    }
    threads = {k: merged[k] for k in _THREAD_ENV_KEYS if k in merged and merged[k] != ""}
    if threads:
        meta["threading"] = threads
    slurm = {k: merged[k] for k in _SLURM_ENV_KEYS if k in merged and merged[k] != ""}
    if slurm:
        meta["slurm"] = slurm
    return meta


@dataclass(frozen=True)
class ExecutionRequest:
    """Specification for executing a single stage process."""

    stage_name: str
    command: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    # Set by a parallel local run (`run_experiment(cores=N)`): `Executor.cancel(token)` stops the
    # processes started for requests carrying this token, and none start once it is set.
    cancel_token: threading.Event | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome and isolated OS resource measurements for one executed stage."""

    exit_code: int
    wall_time_s: float
    max_rss_kb: int | None
    stdout: str
    stderr: str
    hostname: str
    cpu_user_s: float | None = None
    cpu_sys_s: float | None = None
    cpu_efficiency: float | None = None
    minor_page_faults: int | None = None
    major_page_faults: int | None = None
    voluntary_ctx_switches: int | None = None
    involuntary_ctx_switches: int | None = None
    environment: dict[str, Any] = field(default_factory=dict)


class StageCancelled(RuntimeError):
    """A stage was not started (or was stopped) because its run was cancelled."""


class Executor(abc.ABC):
    """Abstract execution backend. Subclass to add DelftBlue SLURM (srun/sbatch).

    `run` may be called from several threads at once (a parallel local run)."""

    @abc.abstractmethod
    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a stage command and return its exit status and resource metrics."""

    def cancel(self, token: threading.Event) -> None:
        """Stop every running stage whose request carries `token` (already set by the caller).

        The default does nothing; a backend that can stop its stages overrides it."""


_EXECUTOR_INIT_LOCK = threading.Lock()


class LocalExecutor(Executor):
    """Runs each stage in an isolated local subprocess and measures wall time, CPU, and RSS.

    Thread-safe: it tracks the children started for a cancellable request (one carrying a
    `cancel_token`), so `cancel(token)` can kill them. Each of those starts in its own session,
    and `cancel` kills its whole process group, so whatever the stage started dies with it.
    Other children stay in this process group, so a terminal Ctrl-C reaches them directly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, tuple[subprocess.Popen[bytes], threading.Event]] = {}

    def _state(self) -> tuple[threading.Lock, dict[int, Any]]:
        # Subclasses that don't call `__init__` still work.
        if "_lock" not in self.__dict__:
            with _EXECUTOR_INIT_LOCK:
                if "_lock" not in self.__dict__:
                    self._active = {}
                    self._lock = threading.Lock()
        return self._lock, self._active

    @staticmethod
    def _kill_group(proc: subprocess.Popen[bytes]) -> None:
        """SIGKILL the process group a cancellable child leads (it is not reaped yet, so its pid
        and group id are still its own), or the child alone if the group is gone."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _forget(self, proc: subprocess.Popen[bytes]) -> None:
        """Stop tracking `proc` (caller holds the lock); a reused pid another run registered stays."""
        _, active = self._state()
        if active.get(proc.pid, (None, None))[0] is proc:
            del active[proc.pid]

    def cancel(self, token: threading.Event) -> None:
        lock, active = self._state()
        with lock:
            for proc, owner in list(active.values()):
                if owner is token:
                    self._kill_group(proc)

    def _reap(self, proc: subprocess.Popen[bytes], tracked: bool) -> tuple[int, Any]:
        """Wait for `proc` to exit and reap it: `(exit code, rusage or None)`. A tracked child
        leaves the tracking table before its pid is freed, so `cancel` never signals a reused pid."""
        lock, _ = self._state()
        if tracked and hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)  # exited, not reaped yet
            with lock:
                self._forget(proc)
        elif tracked:
            # No WNOWAIT: reap without blocking under the lock, so `cancel` can't come in between.
            delay = 0.001
            while True:
                with lock:
                    if hasattr(os, "wait4"):
                        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
                        done = pid != 0
                        if done:
                            proc.returncode = os.waitstatus_to_exitcode(status)
                    else:
                        rusage = None
                        done = proc.poll() is not None
                    if done:
                        self._forget(proc)
                        assert proc.returncode is not None
                        return proc.returncode, rusage
                time.sleep(delay)
                delay = min(2 * delay, 0.01)
        if hasattr(os, "wait4"):
            _, status, rusage = os.wait4(proc.pid, 0)
            # Prevent subprocess.Popen.__del__ from attempting a second wait on the reaped pid.
            proc.returncode = os.waitstatus_to_exitcode(status)
            return proc.returncode, rusage
        return proc.wait(), None

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        env = None
        if request.env is not None:
            env = os.environ.copy()
            env.update(request.env)

        hostname = socket.gethostname()
        max_rss_kb: int | None = None
        cpu_user_s: float | None = None
        cpu_sys_s: float | None = None
        minor_page_faults: int | None = None
        major_page_faults: int | None = None
        voluntary_ctx_switches: int | None = None
        involuntary_ctx_switches: int | None = None

        lock, active = self._state()
        token = request.cancel_token
        tracked = token is not None
        with tempfile.TemporaryFile() as out_fp, tempfile.TemporaryFile() as err_fp:
            # Start under the lock, so a `cancel` either sees this child or comes first and
            # leaves the token set, which is checked here.
            with lock:
                if token is not None and token.is_set():
                    raise StageCancelled(f"Stage '{request.stage_name}' not started: the run was cancelled")
                start = time.perf_counter()
                proc = subprocess.Popen(
                    request.command,
                    cwd=str(request.cwd) if request.cwd is not None else None,
                    env=env,
                    stdout=out_fp,
                    stderr=err_fp,
                    start_new_session=tracked,
                )
                if token is not None:
                    active[proc.pid] = (proc, token)
            try:
                exit_code, rusage = self._reap(proc, tracked)
                if rusage is not None:
                    # On Linux ru_maxrss is in KiB (on macOS it is bytes).
                    max_rss_kb = int(rusage.ru_maxrss)
                    cpu_user_s = float(rusage.ru_utime)
                    cpu_sys_s = float(rusage.ru_stime)
                    minor_page_faults = int(rusage.ru_minflt)
                    major_page_faults = int(rusage.ru_majflt)
                    voluntary_ctx_switches = int(rusage.ru_nvcsw)
                    involuntary_ctx_switches = int(rusage.ru_nivcsw)
            except BaseException:
                # Interrupted (e.g. SIGTERM turned into SystemExit): don't leave the child running.
                with lock:
                    if proc.returncode is None:
                        if tracked:
                            self._kill_group(proc)
                        else:
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                pass
                    self._forget(proc)  # before the pid is freed
                proc.wait()
                raise
            finally:
                with lock:
                    self._forget(proc)
            if token is not None and token.is_set() and exit_code != 0:
                raise StageCancelled(f"Stage '{request.stage_name}' was stopped: the run was cancelled")

            wall_time_s = time.perf_counter() - start
            cpu_efficiency: float | None = None
            if cpu_user_s is not None and cpu_sys_s is not None and wall_time_s > 0.0:
                cpu_efficiency = (cpu_user_s + cpu_sys_s) / wall_time_s

            out_fp.seek(0)
            err_fp.seek(0)
            stdout = out_fp.read().decode("utf-8", errors="replace")
            stderr = err_fp.read().decode("utf-8", errors="replace")

        return ExecutionResult(
            exit_code=exit_code,
            wall_time_s=wall_time_s,
            max_rss_kb=max_rss_kb,
            stdout=stdout,
            stderr=stderr,
            hostname=hostname,
            cpu_user_s=cpu_user_s,
            cpu_sys_s=cpu_sys_s,
            cpu_efficiency=cpu_efficiency,
            minor_page_faults=minor_page_faults,
            major_page_faults=major_page_faults,
            voluntary_ctx_switches=voluntary_ctx_switches,
            involuntary_ctx_switches=involuntary_ctx_switches,
            environment=collect_host_environment(request.env),
        )
