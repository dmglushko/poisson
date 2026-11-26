"""Strict scalar validation shared by the Python API and the CLI JSON config."""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np


def require_int(value: Any, name: str) -> int:
    """Return `value` as an `int`, rejecting `bool`, floats such as `2.9`, and other types."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def require_finite_float(value: Any, name: str) -> float:
    """Return `value` as a finite `float`, rejecting `bool`, non-numbers, `nan`, and `inf`."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


# Supported magnitude ranges for nonzero floating-point values (see architecture.md):
# - mesh / assembly inputs (eps_r, screening_length, BC values and alphas, material voltages,
#   `constant_source`, `source(...)` output): [1e-30, 1e30], so any product or ratio of two or
#   three of them (e.g. `eps_r / sl^2`, `eps_r * V / L`) stays well inside the problem range;
# - coordinates (nodes, charge positions): |v| <= 1e30 only. They enter only through differences
#   (edge vectors), and refined meshes legitimately hold rounding-noise coordinates such as
#   -7.5e-32 next to zero, so there is no lower bound;
# - `AssembledProblem` (`A.data`, `f`, `known_value`): [1e-100, 1e100], so the solver's RHS
#   normalization and residual (squares of normalized values, `A * x`) stay far from float64
#   overflow (~1.8e308) and the subnormal range (< 2.2e-308). `assemble_problem` checks its own
#   `A.data` and `f` against this range, so it never writes a problem `solve` rejects.
# `SolverResult::x` and `NodalField::u` only need to be finite (the field stage copies them).
MESH_MAGNITUDE_MIN: float = 1e-30
MESH_MAGNITUDE_MAX: float = 1e30
PROBLEM_MAGNITUDE_MIN: float = 1e-100
PROBLEM_MAGNITUDE_MAX: float = 1e100


def _fmt_bound(x: float) -> str:
    return f"{x:.0e}".replace("e+", "e")   # 1e-30, 1e30, 1e100


def _fmt_range(lo: float, hi: float) -> str:
    if lo == 0.0:
        return f"|v| <= {_fmt_bound(hi)}"
    return f"[{_fmt_bound(lo)}, {_fmt_bound(hi)}] for nonzero values"


def magnitude_range_error(
    values: Any, name: str, lo: float = PROBLEM_MAGNITUDE_MIN, hi: float = PROBLEM_MAGNITUDE_MAX
) -> str | None:
    """Why `values` violates the supported range, or `None`: every value must be finite and
    every nonzero value must satisfy `lo <= |v| <= hi`."""
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    if arr.dtype.kind not in "fiu":
        return f"{name} must be a real floating-point array, got dtype {arr.dtype}"
    arr = arr.astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(arr)
    if not np.all(finite):
        bad = int(np.flatnonzero(~finite)[0])
        return f"{name} contains a non-finite value {float(arr[bad])!r} at flat index {bad}"
    mag = np.abs(arr)
    out_of_range = (mag != 0.0) & ((mag < lo) | (mag > hi))
    if np.any(out_of_range):
        bad = int(np.flatnonzero(out_of_range)[0])
        return (
            f"{name} value {float(arr[bad])!r} at flat index {bad} is outside the supported magnitude range "
            f"{_fmt_range(lo, hi)}"
        )
    return None


def require_supported_magnitudes(
    values: Any, name: str, lo: float = PROBLEM_MAGNITUDE_MIN, hi: float = PROBLEM_MAGNITUDE_MAX
) -> None:
    """Raise `ValueError` unless `values` is finite with nonzero magnitudes in `[lo, hi]`
    (default: the `AssembledProblem` range [1e-100, 1e100])."""
    why = magnitude_range_error(values, name, lo, hi)
    if why is not None:
        raise ValueError(why)


def require_mesh_magnitudes(values: Any, name: str) -> None:
    """Raise `ValueError` unless `values` is finite with nonzero magnitudes in [1e-30, 1e30]."""
    require_supported_magnitudes(values, name, MESH_MAGNITUDE_MIN, MESH_MAGNITUDE_MAX)


def require_coordinate_magnitudes(values: Any, name: str) -> None:
    """Raise `ValueError` unless `values` (node or charge coordinates) is finite with
    `|v| <= 1e30` (no lower bound: only coordinate differences enter the assembly)."""
    require_supported_magnitudes(values, name, 0.0, MESH_MAGNITUDE_MAX)
