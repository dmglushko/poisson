"""External-library Python test solver for AMR testing and independent verification.

NOT a production C++ pipeline stage. Wraps scikit-fem, SciPy, and PyAMG behind the
project's HDF5 contracts (`MeshProperties`, `AssembledProblem`, `SolverResult`, `NodalField`).
"""

from __future__ import annotations

from .assemble import assemble_problem
from .compare import AssembledComparison, canonicalize_assembled_problem, compare_assembled_problems
from .constants import COULOMB_CONSTANT_NM, E_OVER_EPS0_NM
from .field import build_nodal_field
from .h5_io import (
    NONE_U32,
    AssembledProblemData,
    AssemblyInfo,
    FieldInfo,
    MeshData,
    NodalFieldData,
    SolverInfo,
    SolverResultData,
    read_assembled_problem,
    read_mesh_properties,
    read_nodal_field,
    read_solver_result,
    write_assembled_problem,
    write_mesh_properties,
    write_nodal_field,
    write_solver_result,
)
from .solve import solve_problem

__all__ = [
    "AssembledComparison",
    "AssembledProblemData",
    "AssemblyInfo",
    "COULOMB_CONSTANT_NM",
    "E_OVER_EPS0_NM",
    "FieldInfo",
    "MeshData",
    "NONE_U32",
    "NodalFieldData",
    "SolverInfo",
    "SolverResultData",
    "assemble_problem",
    "build_nodal_field",
    "canonicalize_assembled_problem",
    "compare_assembled_problems",
    "read_assembled_problem",
    "read_mesh_properties",
    "read_nodal_field",
    "read_solver_result",
    "solve_problem",
    "write_assembled_problem",
    "write_mesh_properties",
    "write_nodal_field",
    "write_solver_result",
]
