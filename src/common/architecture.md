# Common primitives

Stage-agnostic numerical and geometric primitives shared across multiple pipeline stages
(`include/common/`, `src/common/`).

Dependency rule: every pipeline stage (`importer`, `amr`, `assembly`, `solver`,
`ray_tracing`) may depend on `common/`; `common/` depends only on the core data structs
(such as `MeshProperties`) and never on any pipeline stage.

## Components

| Header | Purpose | Used by |
|---|---|---|
| `include/common/constants.h` | Physical constants (`elementary_charge`, `vacuum_permittivity`, `electron_mass`, `speed_of_light`) and derived solver-unit factors (`e_over_eps0_nm`, `coulomb_constant_nm`, `e_over_me`) in `poisson::constants`. | AnalyticalMaker, Assembly, Ray Tracing |
| `include/common/quadrature.h` | Strictly positive, strictly interior quadrature rules on tetrahedra and triangles up to degree 20 (weights summing to 1): fully symmetric Witherden–Vincent rules (via PyFR, BSD-3) for degrees 0–8, with collapsed Gauss–Legendre (Stroud conical product) fallback for degrees 9–20. | AMR estimators (the source terms), Assembly |
| `include/common/mesh_locator.h` | Non-owning spatial index (`MeshLocator`) over `MeshProperties`: flat-array AABB BVH (`O(log N)` cold `locate`), BVH-free neighbour walks (`walk_to_point`, `walk_along_segment`, also as standalone free functions), warm `locate_near` (walk, then BVH fallback), exact closed-ball tet intersection (`query_sphere`), and tet-AABB box queries (`query_box`). See [MeshLocator](#meshlocator). | AMR point-source location, AnalyticalMaker (P3M), Ray Tracing |

## MeshLocator

`MeshLocator` holds a pointer to the mesh (which must outlive it and stay unchanged), its
`Settings { max_walk_steps = 64 }` (validated `>= 1`), the BVH node array, and a permutation
`order_` of tet indices grouped by leaf. It stores nothing per tet besides `order_`; tet
geometry is always read back from `mesh.nodes`. The constructor validates the whole mesh
(finite nodes, node/neighbour indices in range, `neighbours.size() == elements.size()`,
positive volumes), so the member walks skip per-tet validation.

### Barycentric coordinates

For local node `i`, `V_i(x)` is 6x the signed volume of the tet with node `i` replaced by
`x`. Each face's three vertices are taken in ascending node-index order and the permutation
sign is folded into the face normal, so `V_0..V_3 > 0` inside a positively oriented tet,
and two tets sharing a face evaluate bit-identical values of opposite sign for it: a point
can never be "outside" both, and walks cannot flip-flop across a face because of rounding.
`V = V_0(p_0)` (6x the tet's signed volume, computed directly from the four vertices and
checked finite and `> 0`) is the query-independent denominator, so coordinates sum to 1 up
to rounding. Containment tests are division-free (`V_i >= -1e-9 * V`, so NaN from extreme
coordinates counts as outside); coordinates are only divided out when a result is returned.

### Walks

Both walks cross faces through `mesh.neighbours`, never use the BVH, take at most
`max_steps` hops (`steps` counts hops), and report `found`, `exited` (with the local
boundary `exit_face`), or `step_limit` with the last visited tet and its barycentric
coordinates. The free functions `walk_to_point(mesh, ...)` / `walk_along_segment(mesh, ...)`
need no `MeshLocator` and validate the mesh as they go: up front the neighbours size, finite
points, `start_element`, and `max_steps >= 1`; for every visited tet its node indices,
finite coordinates, positive `V`, and neighbour indices (`std::invalid_argument`).

- `walk_to_point` is interior-first: it only crosses faces with `V_i < -tol * V` that have a
  neighbour, and returns `exited` only when every such face is a boundary face (then
  `exit_face` is the most negative one). Steps `0..15` use steepest descent (most negative
  interior face); from step 16 a deterministic remembering walk skips the entry face and
  scans from face `splitmix(cur, step) & 3` to break deterministic cycles (termination is
  always guaranteed by `max_steps`). Interior-first walks around notches of non-convex meshes
  that would stop a pure steepest descent walk.
- `walk_along_segment` follows `p(t) = x0 + t (x1 - x0)` from the tet holding `x0`. All four
  faces with `V_i(x1) < -tol * V` are candidates (no entry-face skip is needed because the
  shared face has `V_i(x1) > 0` in the new tet on a consistent mesh, and checking all four
  faces prevents malformed disjoint neighbours from falsely returning `found`). A face is
  crossed only when the remaining segment goes more than `tol` beyond it; for nearly tangent
  segments this prevents false `t = 0` exits on wall-sliding rays, while the reported exit
  can shift along the tolerance-thickened wall. Candidate faces are ordered by crossing
  fraction `t_i = V_i(x0) <= 0 ? 0 : V_i(x0) / (V_i(x0) - V_i(x1))` (ties: most negative
  `V_i(x1)`), clamped monotonically into `[t_cur, 1]`. Before exiting through a boundary
  face at `p(t)`, any interior face with `V_g(x1) < 0` and `V_g(p(t)) <= tol * V` is crossed
  instead (most negative `V_g(x1)`), preventing false exits when grazing reflex boundary
  edges even under floating-point near-ties. A true boundary exit returns `exited` at `p(t)`,
  even when `x1` lies back inside the mesh beyond a cavity (ray tracing needs the wall hit;
  `locate_near` finds the point itself). `step_limit` reports the last crossing fraction
  reached. A slightly-outside `x0` (integration drift) crosses or exits at `t = 0`, and
  `x0 == x1` needs no special case: it walks interior-first at `t = 0`, returning `found`
  (`t = 1`) or `exited` at `t = 0` only when `x0` lies beyond boundary faces alone.
- `locate_near(x, hint)` returns empty immediately when `x` lies outside the padded root
  box; otherwise it runs `walk_to_point` from `hint` and falls back to `locate(x)` when the
  walk does not end `found`.

### BVH

Nodes are 64-byte (`alignas(64)`) `{ Aabb box; first; count; right; }` in depth-first
pre-order: the left child of node `idx` is `idx + 1`, `right == 0` marks a leaf, and every
node covers the contiguous range `order_[first, first + count)`. The build partitions
`order_` top-down with median splits (`nth_element`, ties by tet index) along the axis of
largest `double` spread of temporary `float` split keys (tet-box midpoints shifted relative
to vertex 0 of tet 0), stopping at `<= 8` tets per leaf; leaves therefore hold `>= 4` tets
(except for meshes with `< 4` tets total) and the node count is `<= n_tets / 2`, which is
reserved up front. Exact `double` leaf boxes are then computed from `mesh.nodes` and merged
bottom-up into internal nodes in `O(N)`. Point-location box tests pad each axis by
`4 * bary_tol * max(span, |lo|, |hi|)` to cover the worst-case `3 * bary_tol` protrusion of
three `-bary_tol` barycentric coordinates plus rounding without an absolute floor. Memory is
about `4 + 64 * nodes / n_tets` bytes per tet (~25 B/tet at the median leaf fill of ~6),
versus ~170 B/tet for the previous layout with per-tet barycentric gradients and boxes.

### Range queries

- `query_sphere(center, radius)` returns the tets whose closed tetrahedron meets the closed
  ball (radius padded by `1e-12 * max(radius, root_scale)` where `root_scale` is the max
  axis span/coordinate magnitude of the root box; throws `std::invalid_argument` if the
  padded radius squared overflows). A node whose farthest box corner lies in the ball is
  accepted in `O(1)` and its `k` tet indices are appended in `O(k)`; in leaves a tet is
  accepted when any vertex lies in the ball, else rejected by its box distance, else
  accepted if `V_0..V_3(center) >= 0` (center inside) or a face triangle is within the
  radius. The overload taking `out` clears it and fills unsorted unique indices reusing its
  capacity; the vector-returning overload calls it and sorts.
- `query_box(lo, hi)` returns the tets whose AABB overlaps `[lo, hi]` (a conservative
  superset of exact tet-box intersection), sorted; a node box inside `[lo, hi]` accepts its
  whole range, and leaf tets' boxes are recomputed exactly from their four vertices.

## Tests

- `tests/test_constants.cpp`: checks `poisson::constants` identities and solver-unit magnitudes.
- `tests/test_quadrature.cpp`: positivity, strict interiority, point counts, $S_4$/$S_3$ permutation symmetry (degrees 0–8), monomial exactness up to degree 20 on tets and triangles, and out-of-range degree validation.
- `tests/test_locator.cpp`: cold and warm point location (interior, vertices, edges, faces,
  outside, non-convex cavity/hole fallback, missing neighbours); barycentric coordinates with
  permuted local node orders and shared-face consistency; `walk_to_point` and
  `walk_along_segment` (free and member) statuses, step counts, step limits, an L-shaped mesh
  where interior-first avoids the BVH, a U-shaped cavity where the segment walk exits at the
  wall while `locate_near` finds the end point, boundary/drifted/degenerate segments,
  edge-grazing segments; sphere (both overloads) and box queries against brute force on a
  shuffled mesh; and malformed mesh/query validation.
