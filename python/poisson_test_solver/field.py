"""Reconstruct a NodalField (u at every node + point_sources) for AMR stages."""

from __future__ import annotations

import time

import numpy as np

from .constants import E_OVER_EPS0_NM
from .h5_io import (
    NONE_U32,
    AssembledProblemData,
    FieldInfo,
    MeshData,
    NodalFieldData,
    SolverResultData,
)
from .locate import locate_charges


def build_nodal_field(
    mesh: MeshData,
    problem: AssembledProblemData,
    solution: SolverResultData,
) -> tuple[NodalFieldData, FieldInfo]:
    """Reconstruct nodal potential `u` from `AssembledProblem` + `SolverResult` and attach `point_sources`."""
    t_start = time.perf_counter()
    warnings: list[str] = []

    n_nodes = int(mesh.nodes.shape[0])
    if problem.row.shape[0] != n_nodes:
        raise ValueError(
            f"AssembledProblem row length ({problem.row.shape[0]}) does not match mesh node count ({n_nodes})"
        )
    n_unknowns = int(problem.f.shape[0])
    if solution.x.shape[0] != n_unknowns:
        raise ValueError(
            f"SolverResult x length ({solution.x.shape[0]}) does not match AssembledProblem rows ({n_unknowns})"
        )

    # SolverResult::x only needs to be finite: it is copied into u without arithmetic.
    if not np.all(np.isfinite(solution.x)):
        raise ValueError("SolverResult x contains non-finite values")

    u = np.empty((n_nodes,), dtype=np.float64)
    assigned = np.zeros((n_nodes,), dtype=bool)

    has_row = problem.row != NONE_U32
    if np.any(has_row):
        row_ids = problem.row[has_row]
        if int(np.max(row_ids)) >= n_unknowns:
            raise ValueError("AssembledProblem row index exceeds SolverResult x length")
        u[has_row] = solution.x[row_ids]
        assigned[has_row] = True

    if problem.known_node.size > 0:
        if int(np.max(problem.known_node)) >= n_nodes:
            raise ValueError("AssembledProblem known node index exceeds mesh node count")
        if np.any(assigned[problem.known_node]):
            raise ValueError("AssembledProblem has node(s) present in both `row` and `known`")
        if np.unique(problem.known_node).size != problem.known_node.size:
            raise ValueError("AssembledProblem has duplicate entries in `known`")
        u[problem.known_node] = problem.known_value
        assigned[problem.known_node] = True

    if not np.all(assigned):
        n_missing = int(np.sum(~assigned))
        raise ValueError(f"{n_missing} mesh node(s) appear in neither `row` nor `known`")

    # Convert trapped charges to RefinementInput::PointSource records with deterministic tet hints.
    n_charges = int(mesh.charge_positions.shape[0])
    if n_charges > 0:
        elem_is_metal = mesh.material_is_metal[mesh.element_materials]
        elem_ids, _ = locate_charges(
            mesh.nodes, mesh.elements, elem_is_metal, mesh.charge_positions, allow_metal=False,
            neighbours=mesh.neighbours,
        )
        ps_pos = mesh.charge_positions.astype(np.float64, copy=True)
        ps_val = mesh.charge_values.astype(np.float64) * E_OVER_EPS0_NM
        ps_elem = elem_ids.astype(np.uint32, copy=False)
    else:
        ps_pos = np.zeros((0, 3), dtype=np.float64)
        ps_val = np.zeros((0,), dtype=np.float64)
        ps_elem = np.zeros((0,), dtype=np.uint32)

    elapsed = time.perf_counter() - t_start
    return (
        NodalFieldData(
            u=u,
            point_source_position=ps_pos,
            point_source_value=ps_val,
            point_source_element=ps_elem,
        ),
        FieldInfo(
            time=elapsed,
            n_nodes=n_nodes,
            n_point_sources=n_charges,
            warnings=warnings,
        ),
    )
