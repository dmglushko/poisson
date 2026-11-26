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
| `include/common/quadrature.h` | Collapsed Gauss–Legendre (Stroud conical product) rules on tetrahedra and triangles up to degree 20, strictly positive weights summing to 1. | AMR estimators (the source terms), Assembly |
| `include/common/mesh_locator.h` | Non-owning spatial index (`MeshLocator`) over `MeshProperties`: flat-array AABB BVH with median centroid splits (`O(log N)` cold lookup), `mesh.neighbours` walking with BVH fallback (`locate_near`), exact closed-ball tet intersection (`query_sphere`), and box queries (`query_box`). | AMR point-source location, AnalyticalMaker (P3M), Ray Tracing |

## Tests

- `tests/test_constants.cpp`: checks `poisson::constants` identities and solver-unit magnitudes.
- `tests/test_quadrature.cpp`: monomial exactness up to degree 20 on tets and triangles.
- `tests/test_locator.cpp`: cold and warm point location (interior, vertices, edges, faces,
  outside, non-convex cavity/hole fallback, missing neighbours), exact sphere queries, box
  queries, and malformed mesh/query validation.
