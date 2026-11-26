"""Deterministic AABB + signed-distance point-charge locator safe on graded AMR meshes."""

from __future__ import annotations

import numpy as np

# Membership tolerance in length units: delta = _DIST_ULPS * eps_mach * (||p||_inf + tet extent).
_DIST_ULPS: float = 64.0
_EPS: float = float(np.finfo(np.float64).eps)

# Face i of a tet is the face opposite local vertex i.
_FACES: tuple[tuple[int, int, int], ...] = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
_NONE_U32: int = 0xFFFFFFFF


def compute_face_neighbours(elements: np.ndarray) -> np.ndarray:
    """`(E, 4)` uint32 face neighbours: entry `[e, i]` is the tet across the face opposite local
    vertex `i` of tet `e`, or `0xFFFFFFFF` on the outside (the `MeshProperties::neighbours`
    convention). A face shared by more than two tets is rejected as non-manifold."""
    n_elem = int(elements.shape[0])
    out = np.full((n_elem, 4), _NONE_U32, dtype=np.uint32)
    if n_elem == 0:
        return out
    elems = np.asarray(elements, dtype=np.uint64)
    keys = np.concatenate([np.sort(elems[:, list(face)], axis=1) for face in _FACES])  # (4E, 3)
    owner = np.tile(np.arange(n_elem, dtype=np.int64), 4)
    local = np.repeat(np.arange(4, dtype=np.int64), n_elem)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    same = np.all(k[1:] == k[:-1], axis=1)
    if np.any(same[1:] & same[:-1]):
        raise ValueError("Mesh has a face shared by more than two tetrahedra")
    pair = np.flatnonzero(same)
    a, b = order[pair], order[pair + 1]
    out[owner[a], local[a]] = owner[b].astype(np.uint32)
    out[owner[b], local[b]] = owner[a].astype(np.uint32)
    return out


def _signed_face_distances(tet_coords: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distances from `point` (3,) to the four faces of each candidate tet `(K, 4, 3)`.

    Returns `(d, lam)`, both `(K, 4)`: `d[:, i]` is the perpendicular distance to the plane of
    the face opposite vertex `i` (positive on the vertex's side, i.e. inside), and
    `lam[:, i] = d[:, i] / h_i` is the barycentric coordinate, `h_i = 3|K| / |F_i|` being the
    height of vertex `i` above that face.

    Each tet is shifted by its vertex 0 and scaled by `scale = max |tet - v0|` (per tet) before
    the cross products, so they only see O(1) entries and cannot overflow or underflow whatever
    the coordinate magnitude (e.g. edge length 1e80); `d = scale * d_hat` and
    `h = scale * h_hat`, `d_hat`/`h_hat` being the distances in the normalized tet. The shift
    only costs the rounding already present in the mesh coordinates themselves.
    """
    k = tet_coords.shape[0]
    v0 = tet_coords[:, 0:1, :]                                              # (K, 1, 3)
    rel = tet_coords - v0                                                   # (K, 4, 3)
    scale = np.max(np.abs(rel).reshape(k, -1), axis=1)                      # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / scale
        t_hat = rel * inv[:, None, None]                                    # entries in [-1, 1]
        p_hat = (point[None, :] - v0[:, 0, :]) * inv[:, None]               # (K, 3)
    d_hat = np.empty((k, 4), dtype=np.float64)
    h_hat = np.empty((k, 4), dtype=np.float64)
    for i, (a, b, c) in enumerate(_FACES):
        va = t_hat[:, a, :]
        n = np.cross(t_hat[:, b, :] - va, t_hat[:, c, :] - va)             # |n| = 2 |F_i_hat|
        n_norm = np.linalg.norm(n, axis=1)
        height = np.einsum("kj,kj->k", n, t_hat[:, i, :] - va)             # = +-6|K_hat|
        sign = np.where(height < 0.0, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d_hat[:, i] = sign * np.einsum("kj,kj->k", n, p_hat - va) / n_norm
            h_hat[:, i] = np.abs(height) / n_norm
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d = scale[:, None] * d_hat
        lam = d_hat / h_hat
    return d, lam


def locate_charges(
    nodes: np.ndarray,
    elements: np.ndarray,
    is_metal_elem: np.ndarray,
    charge_positions: np.ndarray,
    *,
    allow_metal: bool = False,
    neighbours: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate each charge deterministically in the tetrahedral mesh.

    Uses a vectorized AABB filter (immune to graded AMR meshes where a centroid KD-tree can
    miss large tets surrounded by fine tets) followed by a physical signed-distance test: tet
    `K` admits `p` iff `min_i d_i >= -delta`, with `d_i` the signed distance (in nm) from `p`
    to the face opposite vertex `i` and `delta = 64 eps_mach (||p||_inf + extent(K))`. A
    length-based tolerance does not scale with the tet size, so a huge dielectric tet next to a
    tiny metal tet cannot swallow a charge sitting inside the metal.

    Raises ValueError if a charge falls outside the mesh or (unless `allow_metal=True`) inside
    the metal, judged by the metal tets' own (small) tolerances so a huge neighbouring
    dielectric tet cannot swallow it. A charge admitted by metal tets is inside the metal iff
      - some admitted metal tet `K` holds it strictly (`min_i d_i > delta_K` over all four
        faces), or
      - every admitted metal tet `K` keeps it off its metal-dielectric interface:
        `min_{i: neighbours[K, i] dielectric} d_i > delta_K` (`+inf` when `K` has no
        dielectric neighbour), e.g. on a metal-metal face away from any dielectric,
    or when no dielectric tet admits it (a face, edge, or vertex inside a metal region or on
    its outer boundary). A charge on the interface (face, edge, or vertex) is admitted by a
    metal tet whose interface face passes through it, so it is not rejected. Otherwise the
    pick prefers:
      1. dielectric tetrahedra over metal tetrahedra, then
      2. the lowest tetrahedron index,
    so a charge on a metal-dielectric interface lands in the dielectric.

    `neighbours` is the `MeshProperties::neighbours` table (face opposite vertex `i`); when
    `None` it is computed from `elements`. `.cha` trapped-charge positions are float32
    (~1e-7 ||p|| resolution), so placement finer than float32 is below the input resolution.
    """
    n_charges = int(charge_positions.shape[0])
    if n_charges == 0:
        return np.zeros((0,), dtype=np.uint32), np.zeros((0, 4), dtype=np.float64)
    if elements.shape[0] == 0:
        raise ValueError("Cannot locate charges in a mesh with no tetrahedra")
    if not np.all(np.isfinite(charge_positions)):
        raise ValueError("Trapped charge positions must be finite")

    n_elem = int(elements.shape[0])
    if neighbours is None:
        neighbours = compute_face_neighbours(elements)
    else:
        neighbours = np.asarray(neighbours)
        if neighbours.shape != (n_elem, 4):
            raise ValueError(f"neighbours must have shape ({n_elem}, 4), got {neighbours.shape}")
        real = neighbours[neighbours != _NONE_U32]
        if real.size > 0 and int(np.max(real)) >= n_elem:
            raise ValueError(f"neighbours references out-of-range tetrahedron {int(np.max(real))}")
    # dielectric_face[e, i]: the face opposite vertex i of tet e borders a dielectric tet.
    has_nb = neighbours != _NONE_U32
    dielectric_face = np.zeros((n_elem, 4), dtype=bool)
    dielectric_face[has_nb] = ~is_metal_elem[neighbours[has_nb].astype(np.intp)]

    tet_coords = nodes[elements]                  # (E, 4, 3)
    bb_min = np.min(tet_coords, axis=1)           # (E, 3)
    bb_max = np.max(tet_coords, axis=1)           # (E, 3)
    extent = np.max(bb_max - bb_min, axis=1)      # (E,)
    coord_max = np.maximum(np.max(np.abs(bb_min), axis=1), np.max(np.abs(bb_max), axis=1))
    # Any admitted p satisfies ||p||_inf <= coord_max + delta, so twice the nominal delta
    # bounds delta(p) and the AABB filter never drops a tet the distance test would admit.
    pad = (2.0 * _DIST_ULPS * _EPS) * (coord_max + extent)
    bb_min_pad = bb_min - pad[:, None]
    bb_max_pad = bb_max + pad[:, None]

    elem_ids = np.empty((n_charges,), dtype=np.uint32)
    weights = np.empty((n_charges, 4), dtype=np.float64)

    for q_idx in range(n_charges):
        p = np.asarray(charge_positions[q_idx], dtype=np.float64)
        where = f"Trapped charge {q_idx} at ({p[0]:.17g}, {p[1]:.17g}, {p[2]:.17g})"
        candidates = np.flatnonzero(np.all((p >= bb_min_pad) & (p <= bb_max_pad), axis=1))
        if candidates.size == 0:
            raise ValueError(f"{where} lies outside the mesh")

        d, lam = _signed_face_distances(tet_coords[candidates], p)
        delta = (_DIST_ULPS * _EPS) * (float(np.max(np.abs(p))) + extent[candidates])
        min_d = np.min(d, axis=1)
        admit = np.flatnonzero(np.isfinite(min_d) & (min_d >= -delta))
        if admit.size == 0:
            raise ValueError(f"{where} lies outside the mesh")

        matched_tets = candidates[admit]           # ascending (from np.flatnonzero)
        matched_metal = is_metal_elem[matched_tets]
        dielectric_pos = np.flatnonzero(~matched_metal)
        if not allow_metal and np.any(matched_metal):
            metal_pos = np.flatnonzero(matched_metal)
            d_metal = d[admit[metal_pos]]                           # (Km, 4)
            delta_metal = delta[admit[metal_pos]]
            strictly_inside = np.min(d_metal, axis=1) > delta_metal
            if np.any(strictly_inside):
                tet = int(matched_tets[metal_pos[np.flatnonzero(strictly_inside)[0]]])
                raise ValueError(f"{where} lies inside metal tetrahedron {tet}")
            d_interface = np.where(dielectric_face[matched_tets[metal_pos]], d_metal, np.inf)
            if np.all(np.min(d_interface, axis=1) > delta_metal):
                tet = int(matched_tets[metal_pos[0]])
                raise ValueError(
                    f"{where} lies inside metal tetrahedron {tet} (off every metal-dielectric interface face)"
                )
        if not allow_metal:
            if dielectric_pos.size == 0:
                # On a face, edge, or vertex shared only by metal tets (e.g. inside a metal
                # region, on its outer boundary): no dielectric tet can receive it.
                raise ValueError(
                    f"{where} lies inside metal tetrahedron {int(matched_tets[0])} (touches no dielectric tetrahedron)"
                )

        pick = int(dielectric_pos[0]) if dielectric_pos.size > 0 else 0
        chosen_tet = int(matched_tets[pick])

        w = np.clip(lam[admit[pick]], 0.0, 1.0)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError(f"Degenerate barycentric weights for charge {q_idx} in tetrahedron {chosen_tet}")
        elem_ids[q_idx] = np.uint32(chosen_tet)
        weights[q_idx] = w / w_sum

    return elem_ids, weights
