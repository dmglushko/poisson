"""Pluggable stage execution backends (local subprocess now, SLURM-ready interface)."""

from __future__ import annotations

import abc
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionRequest:
    """Specification for executing a single stage process."""

    stage_name: str
    command: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    resources: dict[str, Any] = field(default_factory=dict)


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


class Executor(abc.ABC):
    """Abstract execution backend. Subclass to add DelftBlue SLURM (srun/sbatch)."""

    @abc.abstractmethod
    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a stage command and return its exit status and resource metrics."""


class LocalExecutor(Executor):
    """Runs each stage in an isolated local subprocess and measures wall time, CPU, and RSS."""

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        env = None
        if request.env is not None:
            env = os.environ.copy()
            env.update(request.env)

        hostname = socket.gethostname()
        with tempfile.TemporaryFile() as out_fp, tempfile.TemporaryFile() as err_fp:
            start = time.perf_counter()
            proc = subprocess.Popen(
                request.command,
                cwd=str(request.cwd) if request.cwd is not None else None,
                env=env,
                stdout=out_fp,
                stderr=err_fp,
            )

            max_rss_kb: int | None = None
            cpu_user_s: float | None = None
            cpu_sys_s: float | None = None
            minor_page_faults: int | None = None
            major_page_faults: int | None = None
            voluntary_ctx_switches: int | None = None
            involuntary_ctx_switches: int | None = None

            if hasattr(os, "wait4"):
                _, status, rusage = os.wait4(proc.pid, 0)
                exit_code = os.waitstatus_to_exitcode(status)
                # Prevent subprocess.Popen.__del__ from attempting a second wait on the reaped pid.
                proc.returncode = exit_code
                # On Linux ru_maxrss is in KiB (on macOS it is bytes).
                max_rss_kb = int(rusage.ru_maxrss)
                cpu_user_s = float(rusage.ru_utime)
                cpu_sys_s = float(rusage.ru_stime)
                minor_page_faults = int(rusage.ru_minflt)
                major_page_faults = int(rusage.ru_majflt)
                voluntary_ctx_switches = int(rusage.ru_nvcsw)
                involuntary_ctx_switches = int(rusage.ru_nivcsw)
            else:
                exit_code = proc.wait()

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
        )
