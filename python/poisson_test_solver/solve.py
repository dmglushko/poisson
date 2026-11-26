"""Reference linear solvers (SciPy SuperLU direct and PyAMG smoothed-aggregation CG)."""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.sparse.linalg import cg, spsolve

from .h5_io import AssembledProblemData, SolverInfo, SolverResultData
from .validation import require_finite_float, require_int, require_supported_magnitudes

DEFAULT_DIRECT_THRESHOLD: int = 100_000

# Serializes the save-seed-build-restore section around NumPy's *global* RNG between our own
# concurrent `solve_problem` calls (other code touching np.random meanwhile is not covered).
_AMG_RNG_LOCK = threading.Lock()


def _relative_residual(A, x: np.ndarray, f_max: float, b: np.ndarray) -> float:
    """||A (x / f_max) - b||_2 / ||b||_2: the residual of the returned `x` in the RHS-normalized
    system (`b = f / f_max`, ||b||_inf == 1, so ||b||_2 >= 1), which never overflows."""
    return float(np.linalg.norm(A @ (x / f_max) - b)) / float(np.linalg.norm(b))


def solve_problem(
    problem: AssembledProblemData,
    *,
    method: str = "auto",
    rtol: float = 1e-12,
    maxiter: int = 2000,
    direct_threshold: int = DEFAULT_DIRECT_THRESHOLD,
    seed: int = 0,
) -> tuple[SolverResultData, SolverInfo]:
    """Solve `A x = f` from an `AssembledProblemData`.

    Both methods solve the RHS-normalized system `A y = b`, `b = f / max|f|`, and return
    `x = max|f| * y`, so tiny (`1e-90`) or huge (`1e90`) right-hand sides neither underflow
    nor overflow. The reported `residual` is the true relative residual of the *returned* `x`
    in that normalized system, `||A (x / max|f|) - b||_2 / ||b||_2`. Nonzero entries of `A`
    and `f` must lie in the supported magnitude range `[1e-100, 1e100]`.

    Parameters
    ----------
    method:
        - `"direct"`: SciPy SuperLU sparse LU (`spsolve`). Exact to machine precision;
          best for unit tests and small-to-medium 3D meshes (up to ~100k-200k DOFs).
        - `"amg"`: PyAMG smoothed-aggregation V-cycle preconditioner + SciPy CG.
        - `"auto"`: `"direct"` when `n <= direct_threshold`, else `"amg"`.
    rtol:
        Relative residual stopping tolerance for iterative `"amg"` CG (default `1e-12`).
        For either method a warning is recorded iff the true relative residual of the returned
        `x` exceeds `rtol`.
    maxiter:
        Maximum CG iterations for `"amg"` (default `2000`).
    direct_threshold:
        DOF cutoff used when `method == "auto"`.
    seed:
        Seed for NumPy's global RNG while PyAMG builds its hierarchy (its spectral-radius
        estimates draw random start vectors), so `"amg"` output is byte-for-byte reproducible.
        The caller's global RNG state is restored afterwards.
    """
    if method not in ("auto", "direct", "amg"):
        raise ValueError(f"Unknown solver method '{method}'; expected 'auto', 'direct', or 'amg'")
    rtol = require_finite_float(rtol, "rtol")
    if rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    maxiter = require_int(maxiter, "maxiter")
    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}")
    direct_threshold = require_int(direct_threshold, "direct_threshold")
    seed = require_int(seed, "seed")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")

    t_start = time.perf_counter()
    warnings: list[str] = []

    A = problem.A
    f = np.asarray(problem.f, dtype=np.float64)
    n = int(f.shape[0])
    if A.shape != (n, n):
        raise ValueError(f"A shape {A.shape} does not match f length {n}")
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains non-finite values")
    require_supported_magnitudes(f, "f")
    require_supported_magnitudes(A.data, "A.data")

    if n == 0:
        return (
            SolverResultData(x=np.zeros((0,), dtype=np.float64)),
            SolverInfo(time=time.perf_counter() - t_start, iterations=0, residual=0.0, warnings=warnings),
        )

    f_max = float(np.max(np.abs(f)))
    if f_max == 0.0:
        return (
            SolverResultData(x=np.zeros((n,), dtype=np.float64)),
            SolverInfo(time=time.perf_counter() - t_start, iterations=0, residual=0.0, warnings=warnings),
        )
    b = f / f_max

    chosen = ("direct" if n <= direct_threshold else "amg") if method == "auto" else method

    if chosen == "direct":
        A_csc = A.tocsc()
        y = np.asarray(spsolve(A_csc, b), dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(y)):
            raise RuntimeError("Direct solver produced non-finite values (singular or ill-conditioned matrix)")
        A_res = A_csc
        iterations = 1
    else:
        try:
            import pyamg
        except ImportError as exc:
            raise ImportError(
                "pyamg is required for method='amg'; install it with `pip install pyamg`"
            ) from exc

        A_csr = A.tocsr()
        with _AMG_RNG_LOCK:
            saved_rng = np.random.get_state()
            try:
                np.random.seed(seed)
                ml = pyamg.smoothed_aggregation_solver(A_csr)
            finally:
                np.random.set_state(saved_rng)
        M = ml.aspreconditioner(cycle="V")

        iters = 0

        def _on_iter(_xk: np.ndarray) -> None:
            nonlocal iters
            iters += 1

        y, info_code = cg(A_csr, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=_on_iter)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if info_code < 0:
            raise RuntimeError(f"AMG-CG failed with breakdown / illegal input (code {info_code})")
        if not np.all(np.isfinite(y)):
            raise RuntimeError("AMG-CG produced non-finite values")
        A_res = A_csr
        iterations = iters

    x = f_max * y
    if not np.all(np.isfinite(x)):
        raise RuntimeError("Solution overflows float64 after undoing the RHS normalization")
    res = _relative_residual(A_res, x, f_max, b)
    # Judge convergence by the true residual of the returned x, not CG's exit code: SciPy
    # reports info == maxiter even when the last iteration converged, and its recursive
    # residual can drift from the true one.
    if res > rtol:
        if chosen == "amg":
            warnings.append(
                f"AMG-CG did not reach rtol={rtol:g} within {maxiter} iterations (final relative residual {res:.3e})"
            )
        else:
            warnings.append(f"direct solve relative residual {res:.3e} exceeds rtol={rtol:g} (ill-conditioned A?)")

    elapsed = time.perf_counter() - t_start
    return SolverResultData(x=x), SolverInfo(
        time=elapsed,
        iterations=iterations,
        residual=res,
        warnings=warnings,
    )
