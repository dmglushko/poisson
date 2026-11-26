"""Permutation-invariant comparison of two AssembledProblem instances.

Two independent assemblers (e.g. this Python test solver and a teammate's C++ Assembly)
will generally number unknown rows in different orders. Canonicalizing rows by the smallest
mesh node index belonging to each row makes `A`, `f`, `row`, and `known` directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .h5_io import NONE_U32, AssembledProblemData, normalize_csc


@dataclass(frozen=True)
class AssembledComparison:
    equal: bool
    reason: str
    a_max_abs_diff: float
    a_rel_diff: float
    f_max_abs_diff: float
    f_rel_diff: float
    known_max_abs_diff: float


def canonicalize_assembled_problem(problem: AssembledProblemData) -> AssembledProblemData:
    """Permute unknown rows/columns so row k has the k-th smallest minimum mesh node index."""
    n_unknowns = int(problem.f.shape[0])
    row = problem.row.astype(np.uint32, copy=True)

    # Sort known entries by node index.
    if problem.known_node.size > 0:
        k_order = np.argsort(problem.known_node, kind="mergesort")
        known_node = problem.known_node[k_order].astype(np.uint32, copy=False)
        known_value = problem.known_value[k_order].astype(np.float64, copy=False)
    else:
        known_node = np.zeros((0,), dtype=np.uint32)
        known_value = np.zeros((0,), dtype=np.float64)

    if n_unknowns == 0:
        return AssembledProblemData(
            A=normalize_csc(problem.A),
            f=problem.f.astype(np.float64, copy=True),
            row=row,
            known_node=known_node,
            known_value=known_value,
        )

    # Compute min node index per row.
    min_node = np.full((n_unknowns,), np.iinfo(np.uint64).max, dtype=np.uint64)
    has_row = row != NONE_U32
    node_ids = np.flatnonzero(has_row).astype(np.uint64)
    row_ids = row[has_row].astype(np.intp)
    np.minimum.at(min_node, row_ids, node_ids)

    if np.any(min_node == np.iinfo(np.uint64).max):
        unused = int(np.flatnonzero(min_node == np.iinfo(np.uint64).max)[0])
        raise ValueError(f"Row {unused} in AssembledProblem is not referenced by any mesh node")

    # old_of_new[new_r] = old_r; new_of_old[old_r] = new_r
    old_of_new = np.argsort(min_node, kind="mergesort")
    new_of_old = np.empty_like(old_of_new)
    new_of_old[old_of_new] = np.arange(n_unknowns, dtype=old_of_new.dtype)

    # Permute A: A_canon = Q A Q^T where Q[new_r, old_of_new[new_r]] = 1 -> A[old_of_new][:, old_of_new]
    Q = sparse.csc_matrix(
        (np.ones(n_unknowns, dtype=np.float64), (np.arange(n_unknowns), old_of_new)),
        shape=(n_unknowns, n_unknowns),
    )
    A_canon = normalize_csc(Q @ problem.A @ Q.T)
    f_canon = problem.f[old_of_new].astype(np.float64, copy=True)

    row_canon = row.copy()
    row_canon[has_row] = new_of_old[row_ids].astype(np.uint32)

    return AssembledProblemData(
        A=A_canon,
        f=f_canon,
        row=row_canon,
        known_node=known_node,
        known_value=known_value,
    )


def _worst(ratio: np.ndarray) -> tuple[int, float]:
    """Index and value of the largest entry of `ratio` (-1, 0.0 when empty)."""
    if ratio.size == 0:
        return -1, 0.0
    k = int(np.argmax(ratio))
    return k, float(ratio[k])


def _ratio(diff: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """diff / tol, with 0/0 -> 0 and x/0 -> inf, so `ratio > 1` iff `diff > tol`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tol > 0.0, diff / tol, np.where(diff > 0.0, np.inf, 0.0))


def compare_assembled_problems(
    first: AssembledProblemData,
    second: AssembledProblemData,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> AssembledComparison:
    """Compare two AssembledProblemData objects up to unknown-row permutation.

    The test is symmetric in its arguments and scaled row by row, so a large entry in one row
    cannot hide a wrong entry in another and the comparison is invariant under scaling the
    whole system (a `1e-14` system is judged exactly like a `1` system):

    - `A`: over the union of both sparsity patterns, with
      `s_i = max(||A1[i, :]||_inf, ||A2[i, :]||_inf)`, require
      `|A1[i, j] - A2[i, j]| <= atol + rtol * s_i`.
    - `f`: with `U = max |known_value|` over both problems (the scale of the Dirichlet lifting
      `-K u_known` that `f` contains), require
      `|f1[i] - f2[i]| <= atol + rtol * max(|f1[i]|, |f2[i]|, s_i * U)`, evaluated divided
      by `U` when `U > 0` so `s_i * U` never overflows (an overflowing `|diff| / U` is unequal).
    - `known_value`: `|v1[k] - v2[k]| <= atol + rtol * max(|v1[k]|, |v2[k]|)`.

    `reason` names the worst offending (canonical) row or known node. The `*_rel_diff` fields
    are the worst `|diff| / scale` ratios under the same per-row scales.
    """
    def _fail(reason: str, known_diff: float = float("inf")) -> AssembledComparison:
        inf = float("inf")
        return AssembledComparison(False, reason, inf, inf, inf, inf, known_diff)

    if first.row.shape != second.row.shape:
        return _fail(f"node count mismatch: {first.row.shape[0]} vs {second.row.shape[0]}")
    if first.f.shape != second.f.shape:
        return _fail(f"unknown count mismatch: {first.f.shape[0]} vs {second.f.shape[0]}")
    # NaN compares false against every tolerance and inf - inf is NaN, so non-finite values
    # would otherwise slip through as "equal"; they are never valid AssembledProblem data.
    for label, prob in (("first", first), ("second", second)):
        for field_name, values in (("A", prob.A.data), ("f", prob.f), ("known_value", prob.known_value)):
            if not np.all(np.isfinite(values)):
                return _fail(f"{label} problem has non-finite values in {field_name}")

    ca = canonicalize_assembled_problem(first)
    cb = canonicalize_assembled_problem(second)

    if not np.array_equal(ca.known_node, cb.known_node):
        return _fail("known node sets differ")
    if not np.array_equal(ca.row, cb.row):
        return _fail("node-to-row grouping (free/floating-metal partition) differs", 0.0)

    # Canonical row r is named by its smallest mesh node, which identifies it across assemblers.
    n_rows = int(ca.f.shape[0])
    has_row = ca.row != NONE_U32
    row_min_node = np.zeros((n_rows,), dtype=np.int64)
    if n_rows > 0:
        nodes_with_row = np.flatnonzero(has_row)
        row_min_node[:] = np.iinfo(np.int64).max
        np.minimum.at(row_min_node, ca.row[has_row].astype(np.intp), nodes_with_row)

    reasons: list[str] = []

    # ---- known values (per node, symmetric)
    v1, v2 = ca.known_value, cb.known_value
    k_diff = np.abs(v1 - v2)
    k_tol = atol + rtol * np.maximum(np.abs(v1), np.abs(v2))
    known_diff = float(np.max(k_diff)) if k_diff.size > 0 else 0.0
    k_worst, k_ratio = _worst(_ratio(k_diff, k_tol))
    known_ok = k_ratio <= 1.0
    if not known_ok:
        reasons.append(
            f"known values differ (worst at node {int(ca.known_node[k_worst])}: "
            f"{v1[k_worst]!r} vs {v2[k_worst]!r})"
        )

    # ---- A (row-scaled over the union of both patterns, symmetric)
    A1 = ca.A.tocsr()
    A2 = cb.A.tocsr()
    s = np.zeros((n_rows,), dtype=np.float64)
    for M in (A1, A2):
        if M.nnz > 0:
            s = np.maximum(s, np.asarray(abs(M).max(axis=1).todense(), dtype=np.float64).reshape(-1))
    D = (A1 - A2).tocsr()
    D.sum_duplicates()
    d_rows = np.repeat(np.arange(n_rows), np.diff(D.indptr))
    d_abs = np.abs(D.data)
    a_max_abs = float(np.max(d_abs)) if d_abs.size > 0 else 0.0
    a_worst, a_ratio = _worst(_ratio(d_abs, atol + rtol * s[d_rows]))
    a_rel = _worst(_ratio(d_abs, s[d_rows]))[1]
    a_ok = a_ratio <= 1.0
    if not a_ok:
        r, c = int(d_rows[a_worst]), int(D.indices[a_worst])
        reasons.append(
            f"A differs (worst at canonical row {r} (mesh node {int(row_min_node[r])}), column {c}: "
            f"|diff| {d_abs[a_worst]:.3e} vs row scale {s[r]:.3e}; max abs {a_max_abs:.3e})"
        )

    # ---- f (per row, symmetric, with the Dirichlet-lifting scale s_i * U)
    U = 0.0
    for kv in (v1, v2):
        if kv.size > 0:
            U = max(U, float(np.max(np.abs(kv))))
    f1, f2 = ca.f, cb.f
    with np.errstate(over="ignore"):
        f_diff = np.abs(f1 - f2)          # may overflow to inf: then it is unequal, never equal
        if U > 0.0:
            # Everything divided by U, so the lifting scale is s_i itself (s_i * U may overflow).
            f_diff_u = f_diff / U
            f_scale_u = np.maximum(np.maximum(np.abs(f1) / U, np.abs(f2) / U), s)
            f_ratios = _ratio(f_diff_u, atol / U + rtol * f_scale_u)
            f_rel_ratios = _ratio(f_diff_u, f_scale_u)
            f_scale = f_scale_u * U       # for the message only (may be inf)
        else:
            f_scale = np.maximum(np.abs(f1), np.abs(f2))
            f_ratios = _ratio(f_diff, atol + rtol * f_scale)
            f_rel_ratios = _ratio(f_diff, f_scale)
    f_max_abs = float(np.max(f_diff)) if f_diff.size > 0 else 0.0
    f_worst, f_ratio = _worst(f_ratios)
    f_rel = _worst(f_rel_ratios)[1]
    f_ok = f_ratio <= 1.0
    if not f_ok:
        reasons.append(
            f"f differs (worst at canonical row {f_worst} (mesh node {int(row_min_node[f_worst])}): "
            f"{f1[f_worst]!r} vs {f2[f_worst]!r}, scale {f_scale[f_worst]:.3e})"
        )

    return AssembledComparison(
        equal=(known_ok and a_ok and f_ok),
        reason="; ".join(reasons) if reasons else "ok",
        a_max_abs_diff=a_max_abs,
        a_rel_diff=a_rel,
        f_max_abs_diff=f_max_abs,
        f_rel_diff=f_rel,
        known_max_abs_diff=known_diff,
    )
