"""P1 tetrahedral assembly using scikit-fem + exact floating-metal reduction (A = P^T K P).

This is an external-library test assembler used for AMR testing and independent verification,
NOT a production C++ pipeline stage.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .constants import E_OVER_EPS0_NM
from .h5_io import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_ROBIN,
    NONE_U32,
    AssembledProblemData,
    AssemblyInfo,
    MeshData,
    normalize_csc,
)
from .locate import locate_charges
from .validation import (
    require_coordinate_magnitudes,
    require_finite_float,
    require_int,
    require_mesh_magnitudes,
    require_supported_magnitudes,
)

_VOLTAGE_RTOL: float = 1e-12
_VOLTAGE_ATOL: float = 1e-12


def _voltages_match(a: float, b: float) -> bool:
    return abs(a - b) <= _VOLTAGE_ATOL + _VOLTAGE_RTOL * max(abs(a), abs(b))


def _pack_sorted_faces(faces: np.ndarray) -> np.ndarray:
    """Sort each triangle's 3 vertex indices ascending -> (F, 3) uint64."""
    return np.sort(np.asarray(faces, dtype=np.uint64), axis=1)


def _map_boundary_triangles_to_facets(mesh_facets: np.ndarray, boundary_elements: np.ndarray) -> np.ndarray:
    """Map each boundary triangle (B, 3) to its column index in skfem `mesh.facets` (3, F)."""
    if boundary_elements.shape[0] == 0:
        return np.zeros((0,), dtype=np.intp)
    if mesh_facets.shape[1] == 0:
        raise ValueError("Mesh has boundary elements but no tetrahedron facets")

    mf_sorted = _pack_sorted_faces(mesh_facets.T)          # (F, 3)
    be_sorted = _pack_sorted_faces(boundary_elements)      # (B, 3)

    # Lexicographic sort of mesh facets using structured void view for fast vectorized searchsorted.
    row_dtype = np.dtype([("a", "<u8"), ("b", "<u8"), ("c", "<u8")])
    mf_view = np.ascontiguousarray(mf_sorted).view(row_dtype).ravel()
    be_view = np.ascontiguousarray(be_sorted).view(row_dtype).ravel()

    order = np.argsort(mf_view)
    mf_ordered = mf_view[order]
    pos = np.searchsorted(mf_ordered, be_view)

    valid = (pos < mf_ordered.shape[0]) & (mf_ordered[np.minimum(pos, mf_ordered.shape[0] - 1)] == be_view)
    if not np.all(valid):
        bad = int(np.flatnonzero(~valid)[0])
        tri = boundary_elements[bad]
        raise ValueError(f"Boundary element {bad} ({tri[0]}, {tri[1]}, {tri[2]}) is not a face of any tetrahedron")

    return order[pos]


def _classify_nodes(
    mesh: MeshData,
    dirichlet_nodes: dict[int, float],
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Classify every mesh node into `row` and sorted `known` arrays.

    Rules:
    - Nodes belonging to Dirichlet boundary faces are known at their BC value.
    - Metal nodes are grouped into connected components through shared nodes of metal tets.
      * If a metal component has a prescribed voltage (from any metal tet material in the
        component) or touches a Dirichlet boundary node, the entire component is known at
        that voltage (conflicting voltages raise ValueError).
      * Otherwise, all nodes in the floating metal component share a single row index in `A`.
    - Remaining free dielectric nodes each receive a unique row index.
    """
    n_nodes = int(mesh.nodes.shape[0])
    known_map: dict[int, float] = dict(dirichlet_nodes)

    # Check for unused nodes (not belonging to any tetrahedron).
    used = np.zeros(n_nodes, dtype=bool)
    if mesh.elements.size > 0:
        used[mesh.elements.ravel()] = True
    if not np.all(used):
        n_unused = int(np.sum(~used))
        raise ValueError(f"Mesh contains {n_unused} unused node(s) not belonging to any tetrahedron")

    # Warn if a non-metal material carries a voltage (importer warns; not treated as a metal).
    for m_idx, is_m in enumerate(mesh.material_is_metal):
        if not bool(is_m) and bool(mesh.material_has_voltage[m_idx]):
            warnings.append(
                f"material '{mesh.material_names[m_idx]}' has a voltage but is not a metal; voltage ignored"
            )

    elem_is_metal = mesh.material_is_metal[mesh.element_materials]
    metal_tets = np.flatnonzero(elem_is_metal)

    floating_components: list[np.ndarray] = []

    if metal_tets.size > 0:
        metal_conn = mesh.elements[metal_tets]             # (Em, 4)
        metal_nodes = np.unique(metal_conn.ravel())
        # Map global node -> local index 0..Nm-1 for connected_components.
        local_id = np.full(n_nodes, -1, dtype=np.int32)
        local_id[metal_nodes] = np.arange(metal_nodes.size, dtype=np.int32)
        loc_tets = local_id[metal_conn]                    # (Em, 4)

        # Connect all 6 edges of each metal tet (connecting v0 to v1,v2,v3 suffices for connectivity).
        u_edges = np.concatenate([loc_tets[:, 0], loc_tets[:, 0], loc_tets[:, 0]])
        v_edges = np.concatenate([loc_tets[:, 1], loc_tets[:, 2], loc_tets[:, 3]])
        n_m = metal_nodes.size
        adj = sparse.coo_matrix(
            (np.ones(u_edges.size, dtype=np.int8), (u_edges, v_edges)),
            shape=(n_m, n_m),
        )
        n_comp, labels = connected_components(adj, directed=False)

        # Gather prescribed voltages per metal component from metal materials and Dirichlet faces.
        comp_voltage: list[float | None] = [None] * n_comp

        # 1. Material voltages on metal tets
        tet_mats = mesh.element_materials[metal_tets]
        tet_has_v = mesh.material_has_voltage[tet_mats]
        if np.any(tet_has_v):
            prescribed_tets = np.flatnonzero(tet_has_v)
            tet_comps = labels[loc_tets[prescribed_tets, 0]]
            tet_volts = mesh.material_voltage[tet_mats[prescribed_tets]]
            for c_id, volt in zip(tet_comps.tolist(), tet_volts.tolist()):
                prev = comp_voltage[c_id]
                if prev is None:
                    comp_voltage[c_id] = volt
                elif not _voltages_match(prev, volt):
                    raise ValueError(
                        f"Conflicting prescribed voltages ({prev} V and {volt} V) on connected metal component {c_id}"
                    )

        # 2. Dirichlet boundary nodes touching a metal component
        for node_id, volt in dirichlet_nodes.items():
            lid = int(local_id[node_id])
            if lid < 0:
                continue
            c_id = int(labels[lid])
            prev = comp_voltage[c_id]
            if prev is None:
                comp_voltage[c_id] = volt
            elif not _voltages_match(prev, volt):
                raise ValueError(
                    f"Metal component {c_id} at {prev} V conflicts with Dirichlet boundary node {node_id} at {volt} V"
                )

        # Group global metal nodes by component (ordered deterministically by smallest node index).
        comp_nodes: list[list[int]] = [[] for _ in range(n_comp)]
        for lid, gnode in enumerate(metal_nodes.tolist()):
            comp_nodes[int(labels[lid])].append(gnode)

        for c_id in range(n_comp):
            nodes_c = comp_nodes[c_id]
            volt = comp_voltage[c_id]
            if volt is not None:
                for gnode in nodes_c:
                    prev = known_map.get(gnode)
                    if prev is not None and not _voltages_match(prev, volt):
                        raise ValueError(
                            f"Conflicting known voltages ({prev} V and {volt} V) at node {gnode}"
                        )
                    known_map[gnode] = volt
            else:
                floating_components.append(np.asarray(nodes_c, dtype=np.uint32))

        # Order floating components deterministically by their smallest global node index.
        floating_components.sort(key=lambda arr: int(arr[0]))

    row = np.full(n_nodes, NONE_U32, dtype=np.uint32)
    is_known = np.zeros(n_nodes, dtype=bool)
    for gnode in known_map:
        is_known[gnode] = True

    next_row = 0
    assigned = is_known.copy()

    # Assign one shared row per floating metal component.
    for comp in floating_components:
        row[comp] = np.uint32(next_row)
        assigned[comp] = True
        next_row += 1

    # Assign one unique row per remaining free dielectric node in ascending node order.
    free_nodes = np.flatnonzero(~assigned)
    if free_nodes.size > 0:
        row[free_nodes] = np.arange(next_row, next_row + free_nodes.size, dtype=np.uint32)
        next_row += int(free_nodes.size)

    # Sort known entries by node index for deterministic output.
    sorted_known = sorted(known_map.items())
    known_node = np.asarray([k for k, _ in sorted_known], dtype=np.uint32)
    known_value = np.asarray([v for _, v in sorted_known], dtype=np.float64)

    return row, known_node, known_value, next_row


def _check_nonsingular_components(A: sparse.csc_matrix, pinned_rows: np.ndarray, warnings: list[str]) -> None:
    """Require every connected component of `A` to be structurally pinned.

    A component is pinned when one of its rows couples (nonzero `K_full` entry) to a known
    node, touches a Robin face with `alpha > 0`, or touches a non-metal tet with active
    screening `c > 0`; otherwise its constant mode lies in the kernel of `A` and the system is
    singular whatever the magnitudes. Being structural, the test is invariant under scaling
    (an `eps_r = 1e-14` box pinned by Dirichlet is accepted). The scale-invariant energy ratio
    `v_C^T A v_C / sum |diag(A)[C]|` of the constant indicator `v_C` is then only reported as an
    ill-conditioning warning.
    """
    n = A.shape[0]
    if n == 0:
        return
    n_comp, labels = connected_components(A, directed=False)
    comp_pinned = np.zeros(n_comp, dtype=bool)
    comp_pinned[labels[pinned_rows]] = True
    if not np.all(comp_pinned):
        c = int(np.flatnonzero(~comp_pinned)[0])
        raise ValueError(
            f"Assembled system is singular on free connected component {c} "
            f"({int(np.sum(labels == c))} unknowns): no Dirichlet node, prescribed metal, "
            "Robin alpha > 0, or active screening pins its potential"
        )

    # v_C^T A v_C = sum of the entries of A inside component C (components share no entries).
    coo = A.tocoo()
    energy = np.bincount(labels[coo.row], weights=coo.data, minlength=n_comp)
    scale = np.bincount(labels, weights=np.abs(A.diagonal()), minlength=n_comp)
    for c in range(n_comp):
        if scale[c] == 0.0:
            raise ValueError(f"Assembled system has an all-zero diagonal on free connected component {c}")
        if energy[c] <= 1e-12 * scale[c]:
            warnings.append(
                f"free connected component {c} ({int(np.sum(labels == c))} unknowns) is nearly singular: "
                f"constant-mode energy ratio {energy[c] / scale[c]:.3e} <= 1e-12"
            )


def assemble_problem(
    mesh: MeshData,
    *,
    constant_source: float | None = None,
    source: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    intorder: int = 2,
) -> tuple[AssembledProblemData, AssemblyInfo]:
    """Assemble the linear system A x = f from MeshProperties using scikit-fem.

    Parameters
    ----------
    mesh:
        Imported or adapted `MeshProperties`.
    constant_source:
        Optional constant smooth RHS f [V/nm^2] applied on all non-metal tetrahedra.
    source:
        Optional callable `source(x, elem_indices)` taking quadrature coordinates `x` of shape
        `(3, n_nonmetal, n_qp)` and 1D global element indices `elem_indices` of shape `(n_nonmetal,)`,
        returning values of shape `(n_nonmetal, n_qp)`. Mutually exclusive with `constant_source`.
    intorder:
        Quadrature order for volume assembly (default 2, exact for P1 stiffness and mass).
    """
    try:
        from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet
        from skfem.helpers import dot, grad
    except ImportError as exc:
        raise ImportError(
            "scikit-fem is required for poisson_test_solver.assemble; install it with `pip install scikit-fem`"
        ) from exc

    if constant_source is not None and source is not None:
        raise ValueError("Specify at most one of `constant_source` and `source`")
    if constant_source is not None:
        constant_source = require_finite_float(constant_source, "constant_source")
        require_mesh_magnitudes(constant_source, "constant_source")
    intorder = require_int(intorder, "intorder")
    if intorder < 2:
        raise ValueError("intorder must be at least 2 for exact P1 mass integration")
    if mesh.nodes.shape[0] == 0 or mesh.elements.shape[0] == 0:
        raise ValueError("Mesh must contain at least one node and one tetrahedron")
    # Defensive invariants (read_mesh_properties already enforces them for files, but an
    # in-memory MeshData may be built by hand).
    if mesh.boundary_element_bcs.shape[0] != mesh.boundary_elements.shape[0]:
        raise ValueError("boundary_element_bcs length must match boundary_elements")
    if mesh.boundary_element_bcs.size > 0 and int(np.max(mesh.boundary_element_bcs)) >= mesh.bc_type.shape[0]:
        raise ValueError(
            f"boundary element references boundary condition {int(np.max(mesh.boundary_element_bcs))}, "
            f"but the mesh has only {mesh.bc_type.shape[0]}"
        )

    # Supported mesh-input magnitude ranges (see architecture.md): read_mesh_properties enforces
    # them for files; an in-memory MeshData is checked here.
    require_coordinate_magnitudes(mesh.nodes, "nodes")
    require_coordinate_magnitudes(mesh.charge_positions, "charge_positions")
    for name, values in (
        ("material_eps_r", mesh.material_eps_r),
        ("material_screening_length", mesh.material_screening_length),
        ("material_voltage", mesh.material_voltage),
        ("bc_value", mesh.bc_value),
        ("bc_alpha", mesh.bc_alpha),
    ):
        require_mesh_magnitudes(values, name)

    t_start = time.perf_counter()
    warnings: list[str] = []

    n_nodes = int(mesh.nodes.shape[0])
    nodes_t = np.ascontiguousarray(mesh.nodes.T, dtype=np.float64)
    elems_t = np.ascontiguousarray(mesh.elements.T, dtype=np.int64)
    sk_mesh = MeshTet(nodes_t, elems_t)
    if sk_mesh.nvertices != n_nodes or not np.array_equal(sk_mesh.p, nodes_t):
        raise RuntimeError("scikit-fem MeshTet unexpectedly modified vertex ordering")
    elem_fe = ElementTetP1()

    # Validate boundary_elements against mesh facets and reject interior boundary faces.
    be_facets = _map_boundary_triangles_to_facets(sk_mesh.facets, mesh.boundary_elements)
    if be_facets.size > 0:
        interior = sk_mesh.f2t[1, be_facets] >= 0
        if np.any(interior):
            bad_idx = int(np.flatnonzero(interior)[0])
            tri = mesh.boundary_elements[bad_idx]
            raise ValueError(
                f"Boundary element {bad_idx} ({tri[0]}, {tri[1]}, {tri[2]}) is an interior face shared by two tetrahedra"
            )
        # Also verify no duplicate boundary faces.
        if np.unique(be_facets).size != be_facets.size:
            raise ValueError("Mesh contains duplicate boundary elements on the same face")

    elem_is_metal = mesh.material_is_metal[mesh.element_materials]
    nonmetal_elems = np.flatnonzero(~elem_is_metal)

    # ---- Assemble K_full (volume dielectric + Robin boundary terms)
    t_a0 = time.perf_counter()
    K_full = sparse.csc_matrix((n_nodes, n_nodes), dtype=np.float64)

    if nonmetal_elems.size > 0:
        vol_basis = Basis(sk_mesh, elem_fe, elements=nonmetal_elems, intorder=intorder)
        mats_nm = mesh.element_materials[nonmetal_elems]
        eps_nm = mesh.material_eps_r[mats_nm]
        if np.any(eps_nm <= 0.0) or not np.all(np.isfinite(eps_nm)):
            raise ValueError("Non-metal relative permittivity eps_r must be finite and positive")

        if mesh.screened:
            sl = mesh.material_screening_length[mats_nm]
            if np.any(sl < 0.0) or not np.all(np.isfinite(sl)):
                raise ValueError("screening_length must be finite and non-negative")
            # (eps / sl) / sl rather than eps / sl^2, so sl^2 itself never under- or overflows.
            sl_safe = np.where(sl > 0.0, sl, 1.0)
            c_nm = np.where(sl > 0.0, (eps_nm / sl_safe) / sl_safe, 0.0)
        else:
            c_nm = np.zeros_like(eps_nm)

        @BilinearForm
        def _poisson_form(u, v, w):
            return w.eps * dot(grad(u), grad(v)) + w.c * u * v

        K_full = K_full + _poisson_form.assemble(vol_basis, eps=eps_nm[:, None], c=c_nm[:, None]).tocsc()
    else:
        vol_basis = None
        c_nm = np.zeros((0,), dtype=np.float64)

    # Classify boundary elements by BC type.
    dirichlet_nodes: dict[int, float] = {}
    robin_facets_list: list[np.ndarray] = []
    robin_alphas_list: list[np.ndarray] = []
    flux_facets_list: list[np.ndarray] = []
    flux_gs_list: list[np.ndarray] = []

    if be_facets.size > 0:
        be_bc_ids = mesh.boundary_element_bcs
        adj_tets = sk_mesh.f2t[0, be_facets]
        adj_is_metal = elem_is_metal[adj_tets]

        for bc_id in range(mesh.bc_type.shape[0]):
            sel = np.flatnonzero(be_bc_ids == bc_id)
            if sel.size == 0:
                continue
            btype = int(mesh.bc_type[bc_id])
            bval = float(mesh.bc_value[bc_id])

            if btype == BC_DIRICHLET:
                for node_id in np.unique(mesh.boundary_elements[sel].ravel()).tolist():
                    prev = dirichlet_nodes.get(node_id)
                    if prev is not None and not _voltages_match(prev, bval):
                        raise ValueError(
                            f"Conflicting Dirichlet boundary values ({prev} V and {bval} V) at node {node_id}"
                        )
                    dirichlet_nodes[node_id] = bval
            elif btype == BC_NEUMANN:
                # Neumann g = 0 on a metal outer face (e.g. planar_layers metal end) is a valid no-op.
                if np.any(adj_is_metal[sel]) and bval != 0.0:
                    raise ValueError(
                        f"Non-zero Neumann condition (g={bval}) on a metal outer face is unsupported"
                    )
                diel_sel = sel[~adj_is_metal[sel]]
                if diel_sel.size > 0 and bval != 0.0:
                    flux_facets_list.append(be_facets[diel_sel])
                    flux_gs_list.append(np.full(diel_sel.size, bval, dtype=np.float64))
            elif btype == BC_ROBIN:
                if not bool(mesh.bc_has_alpha[bc_id]):
                    raise ValueError(f"Robin boundary condition {bc_id} is missing alpha")
                alpha = float(mesh.bc_alpha[bc_id])
                if alpha < 0.0 or not np.isfinite(alpha):
                    raise ValueError(f"Robin alpha must be finite and non-negative, got {alpha}")
                if np.any(adj_is_metal[sel]) and (alpha != 0.0 or bval != 0.0):
                    raise ValueError(
                        f"Non-trivial Robin condition (alpha={alpha}, g={bval}) on a metal outer face is unsupported"
                    )
                diel_sel = sel[~adj_is_metal[sel]]
                if diel_sel.size > 0:
                    if alpha > 0.0:
                        robin_facets_list.append(be_facets[diel_sel])
                        robin_alphas_list.append(np.full(diel_sel.size, alpha, dtype=np.float64))
                    if bval != 0.0:
                        flux_facets_list.append(be_facets[diel_sel])
                        flux_gs_list.append(np.full(diel_sel.size, bval, dtype=np.float64))
            else:
                raise ValueError(f"Unknown boundary condition type {btype}")

    all_robin_facets = np.zeros((0,), dtype=np.intp)
    if robin_facets_list:
        all_robin_facets = np.concatenate(robin_facets_list)
        all_robin_alphas = np.concatenate(robin_alphas_list)
        robin_basis = FacetBasis(sk_mesh, elem_fe, facets=all_robin_facets, intorder=2)

        @BilinearForm
        def _robin_form(u, v, w):
            return w.alpha * u * v

        K_full = K_full + _robin_form.assemble(robin_basis, alpha=all_robin_alphas[:, None]).tocsc()

    # Classify nodes into `row` and `known` and build prolongation matrix P.
    row, known_node, known_value, n_unknowns = _classify_nodes(mesh, dirichlet_nodes, warnings)

    free_mask = row != NONE_U32
    free_global = np.flatnonzero(free_mask)
    P = sparse.csc_matrix(
        (np.ones(free_global.size, dtype=np.float64), (free_global, row[free_global].astype(np.int64))),
        shape=(n_nodes, n_unknowns),
    )

    A = normalize_csc(P.T @ K_full @ P)
    t_a1 = time.perf_counter()

    # ---- Assemble b_full (smooth volume source + Neumann/Robin boundary flux + point charges)
    t_f0 = time.perf_counter()
    b_full = np.zeros(n_nodes, dtype=np.float64)

    if vol_basis is not None and (constant_source is not None or source is not None):
        if constant_source is not None:
            f_vals = np.full((nonmetal_elems.size, 1), float(constant_source), dtype=np.float64)
        else:
            assert source is not None
            coords = vol_basis.global_coordinates()
            raw = np.asarray(source(coords, nonmetal_elems))
            if np.iscomplexobj(raw) or raw.dtype == object:
                raise ValueError(f"source(...) must return real floating-point values, got dtype {raw.dtype}")
            f_vals = raw.astype(np.float64)
            expected = (nonmetal_elems.size, np.asarray(coords).shape[2])
            if f_vals.shape != expected:
                raise ValueError(f"source(...) must return shape {expected}, got {f_vals.shape}")
            if not np.all(np.isfinite(f_vals)):
                raise ValueError("source(...) returned non-finite values")
            require_mesh_magnitudes(f_vals, "source(...) output")

        @LinearForm
        def _rhs_form(v, w):
            return w.f * v

        b_full += _rhs_form.assemble(vol_basis, f=f_vals)

    if flux_facets_list:
        all_flux_facets = np.concatenate(flux_facets_list)
        all_flux_gs = np.concatenate(flux_gs_list)
        flux_basis = FacetBasis(sk_mesh, elem_fe, facets=all_flux_facets, intorder=2)

        @LinearForm
        def _flux_form(v, w):
            return w.g * v

        b_full += _flux_form.assemble(flux_basis, g=all_flux_gs[:, None])

    if mesh.charge_positions.shape[0] > 0:
        charge_elems, charge_weights = locate_charges(
            mesh.nodes, mesh.elements, elem_is_metal, mesh.charge_positions, allow_metal=False,
            neighbours=mesh.neighbours,
        )
        zero_q = int(np.sum(mesh.charge_values == 0))
        if zero_q > 0:
            warnings.append(f"{zero_q} trapped charge(s) have zero value")
        q_rhs = mesh.charge_values.astype(np.float64) * E_OVER_EPS0_NM
        charge_nodes = mesh.elements[charge_elems]         # (Q, 4)
        contrib = q_rhs[:, None] * charge_weights          # (Q, 4)
        np.add.at(b_full, charge_nodes.ravel(), contrib.ravel())

    u_known = np.zeros(n_nodes, dtype=np.float64)
    if known_node.size > 0:
        u_known[known_node] = known_value

    f_vec = np.asarray(P.T @ (b_full - K_full @ u_known), dtype=np.float64).reshape(-1)
    t_f1 = time.perf_counter()
    # Within the supported input range these cannot overflow; guard against it anyway.
    if not np.all(np.isfinite(A.data)):
        raise RuntimeError("Assembled matrix A contains non-finite values")
    if not np.all(np.isfinite(f_vec)):
        raise RuntimeError("Assembled right-hand side f contains non-finite values")
    # Never emit a problem that read_assembled_problem / solve would reject: extreme (but
    # individually supported) combinations such as a huge eps_r with a huge Dirichlet voltage can
    # push A or f outside the AssembledProblem range [1e-100, 1e100].
    require_supported_magnitudes(A.data, "assembled A.data")
    require_supported_magnitudes(f_vec, "assembled f")

    # Structural pins per free row (see _check_nonsingular_components).
    pinned_node = np.zeros(n_nodes, dtype=bool)
    if known_node.size > 0:
        is_known = np.zeros(n_nodes, dtype=np.float64)
        is_known[known_node] = 1.0
        K_pattern = K_full.copy()
        K_pattern.data = (K_pattern.data != 0.0).astype(np.float64)
        pinned_node |= np.asarray(K_pattern @ is_known).reshape(-1) > 0.0
    if all_robin_facets.size > 0:
        pinned_node[sk_mesh.facets[:, all_robin_facets].ravel()] = True
    if nonmetal_elems.size > 0 and np.any(c_nm > 0.0):
        pinned_node[mesh.elements[nonmetal_elems[c_nm > 0.0]].ravel()] = True
    pinned_rows = np.unique(row[pinned_node & free_mask]).astype(np.intp)
    _check_nonsingular_components(A, pinned_rows, warnings)

    t_end = time.perf_counter()
    problem = AssembledProblemData(
        A=A,
        f=f_vec,
        row=row,
        known_node=known_node,
        known_value=known_value,
    )
    info = AssemblyInfo(
        time=t_end - t_start,
        time_assemble_A=t_a1 - t_a0,
        time_assemble_f=t_f1 - t_f0,
        nnz=int(A.nnz),
        n_dirichlet=int(known_node.size),
        warnings=warnings,
    )
    return problem, info
