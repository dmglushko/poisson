# Verification

Problems with a known exact solution, to check the pipeline against
(`include/verification/`, `src/verification/`, library `poisson_verification`). A case
defines everything about its problem in one C++ class: the geometry (as a gmsh `.geo`
script), the materials, the boundary conditions, the trapped charges and the exact potential
with its gradient. So the mesh a case produces and the formula it is checked against cannot
drift apart. Python only names a case and passes its parameters
(`python/poisson_runner/cases.py`).

Nothing here is part of the solve. Only the `poisson_case` CLI links this library, so adding
or changing a case does not change the code hash of the import, AMR or solver stages.

## Flow

```text
Case(name, params) --poisson_case--> case.geo + case.yaml + case.cha
                   --gmsh--> case.msh --poisson_import--> mesh.h5
mesh_mismatches(mesh, case)          checks that a mesh is the mesh of the case
case.u(x, material), case.grad_u(...)  the exact solution, as plain functions
```

**Not here yet:** comparing a computed solution with the exact one (error norms, values at
chosen points, the exact field on a mesh). It depends on the `Solution` format, which is not
defined (with P3M it is the FEM field plus an analytical correction near each charge). See
[issue #6](https://github.com/dmglushko/poisson/issues/6); a first version that assumed a P1
nodal field is in commit `63a0427`.

## Interface

```cpp
struct VerificationProblem {
    std::string geometry;                                        // gmsh .geo (OpenCASCADE)
    std::vector<MeshProperties::Material>          materials;    // one per physical volume
    std::vector<MeshProperties::BoundaryCondition> boundary_conditions;
    std::vector<MeshProperties::TrappedCharge>     charges;      // float32-exact positions
    bool   screened = true;
    double default_exclusion_radius = 0;   // nm
    double boundary_error_bound = 0;       // V; 0 when every boundary condition is exact
};

class VerificationCase {
public:
    virtual std::string_view name() const = 0;
    const VerificationProblem& problem() const;
    double                u(const std::array<double, 3>& x, std::size_t material) const;       // NVI
    std::array<double, 3> grad_u(const std::array<double, 3>& x, std::size_t material) const;  // NVI
    std::size_t material_index(std::string_view name) const;
};

std::vector<std::string> mesh_mismatches(const MeshProperties&, const VerificationCase&);
```

- `u` and `grad_u` take the **material** of the point, not just its position: at an
  interface the potential is continuous but its gradient is not, and a faceted mesh of a
  curved interface puts some tets slightly on the other side. The formula of the tet's
  material is used. In the spherical case it is evaluated at the radius clamped into the
  layer (extending a strongly screened layer's exponentials past its edge would overflow).
- Each case has a nested `Settings` validated in its constructor (`std::invalid_argument`,
  including a problem without a unique solution).
- Charge positions are rounded to float32 in the constructor, so the formula, the `.cha`
  file and the imported mesh agree bit for bit.
- `mesh_mismatches` compares an imported mesh with the case: the screened flag, every
  mesh material against the case material of its name (metal, voltage, and for dielectrics
  `eps_r` and `screening_length`; a name can occur twice, e.g. `"gate 5V"` and `"gate 7V"`),
  the set of boundary conditions, the trapped charges, and where things are (each case's
  private `geometry_mismatches`), plus: every boundary triangle is an outer face and every
  outer face has a boundary condition.
  - **Flat cases** (planar layers, the image boxes) know exactly where each face and region
    is, to `1e-9` of the case size: every boundary triangle has the condition the case puts
    there (one off the case's boundary is reported too), every tet lies in its material's
    region. Swapped end conditions, other layer thicknesses and box sizes are caught.
  - **Spheres.** A meshed sphere is a polyhedron inside the sphere: some vertices lie on it,
    others (after AMR, on flat facets) inside. So: every tet vertex within its layer's outer
    radius, and at most half the layer below under its inner radius; some vertex of every
    layer on its outer radius (a too small domain is caught); every boundary triangle carries
    a condition the case puts on a sphere it does not extend past, and some triangle with
    each sphere's condition reaches that sphere (swapped inner/outer conditions are caught).
    No triangle is classified by its own radius, so refined meshes whose new boundary nodes
    lie inside the sphere still match.

## Cases

Lengths in nm. `K = constants::coulomb_constant_nm`, `G(r) = exp(-r/lambda)/r` (`1/r`
without screening). Boundary conditions follow `MeshProperties`:
$\varepsilon_r \partial_n u + \alpha u = g$.

| Name (`name()`) | Class | Problem | Exact? |
|---|---|---|---|
| `planar_layers` | `PlanarLayersCase` | Box `[0, L] x [0, width] x [0, height]`, layers along x (dielectrics with their own `eps_r` and screening length, metals at a voltage or floating), `left` / `right` conditions on `x = 0` / `x = L`, Neumann 0 on the sides | yes |
| `spherical_layers` | `SphericalLayersCase` | Concentric layers from `inner_radius` (a ball if 0, else a shell with an `inner` condition), `outer` condition, optional central `charge` in a dielectric ball | yes |
| `grounded_ball_image` | `GroundedBallImageCase` | Ball, one dielectric, no screening, `u = 0` on the sphere, a charge anywhere inside (Kelvin image `-qR/d` at `p R^2/d^2`) | yes |
| `half_space_image` | `HalfSpaceImageCase` | Box `[-W, W]^2 x [0, H]`, charge at `(0, 0, d)`, plane `z = 0` Dirichlet 0 (image `-q`) or Neumann 0 (image `+q`), other faces `u = 0` | far faces approximate |
| `dielectric_interface` | `DielectricInterfaceCase` | Box `[-W, W]^2 x [-H, H]`, `upper` (z > 0) and `lower` media with the same screening length, charge at `(0, 0, d)` in `upper`, all faces `u = 0` | far faces approximate |

The layered cases cover the slab and shell cases of the verification plan: a straight line
between two voltages, Neumann or Robin ends, two dielectrics, floating and prescribed metals,
screened slabs (`cosh`/`sinh` profiles), a Robin end that acts as an infinitely long slab,
two screened layers, shells, and a central charge with Dirichlet, Neumann or Robin outer
conditions and dielectric shells around it.

### How the layered cases are solved

In a dielectric layer with `k = 1/lambda` (0 without screening, or when `screened` is false),
the potential is a combination of two basis functions:

| | planar (`s = x`) | spherical (`s = r`) | ball (`r` from 0) |
|---|---|---|---|
| `k = 0` | `1`, `s - mid` | the same divided by `r` (`1/r`, `1 - mid/r`) | `1` |
| `0 < k L < 1` | `cosh(k(s - mid))`, `sinh(k(s - mid))/k` | the same divided by `r` | `sinh(k r)/r` (scaled) |
| `k L >= 1` | `exp(-k(s - start))`, `exp(-k(end - s))` | the same divided by `r` | `sinh(k r)/r` (scaled) |

The decaying exponentials keep every basis function at most 1 in its layer, so thick and
strongly screened layers do not overflow. A ball with a central charge adds
`q K / eps_r * G(r)`. A metal layer is one potential: given, or one unknown when floating.

The coefficients follow from one small dense system (Eigen full-pivot LU, rows equilibrated):

- between two dielectrics: `u` and the flux `eps_r du/ds` (times `r^2` for spheres) are continuous;
- next to a metal: `u` equals the metal potential;
- a floating metal carries no net charge: the flux into it equals the flux out of it;
- the two end conditions (Dirichlet, Neumann or Robin, with the outward normal).

Uniqueness is decided structurally: two solutions differ by a field of zero energy, so by a
constant, which a Dirichlet end, a Robin end with `alpha > 0`, a metal at a voltage or a
screened dielectric forces to 0. Without any of them (for example Neumann everywhere and no
screening) the case is rejected as having no unique solution. When screening alone fixes it,
the constant is set by terms of order `(L / lambda)^2` (L the whole problem's size); below
`1e-10` they are lost to rounding, so such a case is rejected as too ill-conditioned rather
than solved wrongly. Otherwise the system is solved, and a large residual or an estimated
reciprocal condition number below `1e-12` is reported as too ill-conditioned. A layer with `k L < 1` uses the
centred pair `cosh(k t)`, `sinh(k t) / k` (`t` from the layer middle; `1`, `t` at `k = 0`)
instead of the two exponentials, which become indistinguishable as `k -> 0`; this keeps
very weak screening (screening length far above the size) accurate. A metal layer at an end needs Neumann 0 on that end face, and two
adjacent metals must be merged into one layer.

### Parameters (JSON)

`poisson_case` reads a case as
`{"name": <name>, "params": {...}}` (`cli/verification/case_config.h`, the only place the
parameter names appear). Unknown names, unknown or missing parameters and wrong types are
errors. A boundary condition is `{"type": "dirichlet" | "neumann" | "robin", "value": v,
"alpha": a}` (`alpha` only for Robin); a layer is `{"name", "thickness", "eps_r" = 1,
"screening_length" = 0, "metal" = false, "voltage"}` (`voltage` only for a metal; without it
the metal floats).

| Case | Parameters (defaults) |
|---|---|
| `planar_layers` | `layers` (required), `left` (Dirichlet 0), `right` (Dirichlet 1), `width` (1), `height` (1), `screened` (true) |
| `spherical_layers` | `layers` (required), `inner_radius` (0), `inner` (required iff `inner_radius > 0`), `outer` (Dirichlet 0), `charge` (0), `screened` (true) |
| `grounded_ball_image` | `radius` (10), `eps_r` (1), `charge` (-1), `position` ([3, 0, 0]) |
| `half_space_image` | `eps_r` (1), `screening_length` (1), `charge` (-1), `distance` (2), `half_width` (20), `height` (20), `plane` ("dirichlet" or "neumann"), `screened` (true) |
| `dielectric_interface` | `eps_upper` (1), `eps_lower` (4), `screening_length` (1), `charge` (-1), `distance` (2), `half_width` (20), `half_height` (20), `screened` (true) |

Material names are a letter followed by letters, digits or `_`; `vacuum` and names starting
with `detector` are rejected, since the importer gives them a special meaning. The image cases name their materials
`medium`, or `upper` and `lower`.

### Approximate far faces

The half-space and interface formulas hold in an infinite domain; on the box, the far faces
are set to `u = 0`, which the formula is not. `problem().boundary_error_bound` bounds
`|u_exact|` there (from the nearest far face, e.g. `|q| K / eps_r * G(min(W, H - d))` for a
Dirichlet plane). With screening and a box many screening lengths wide it is negligible
(`e^-20`); without screening it is not, and errors below it mean nothing. The bound and a
warning are in the `poisson_case` info.

## Generated input files

`make_case_inputs(case)` returns a `CaseInputsResult { geometry, materials, charges, info }`:

- **`.geo`**: OpenCASCADE boxes or spheres, fragmented so that every interface is a mesh
  face. Volumes and faces are picked by bounding boxes (thin boxes around each face of a box;
  nested boxes for concentric spheres), grown by a quarter of the smallest size of the case
  (well above OpenCASCADE's tolerance, well below the distance to any other face) and named for the importer: a volume after its
  material, with `" <V>V"` for a prescribed metal; a surface after its condition (`"0V"`,
  `"D0"`, `"R0.5,0.1"`), with one physical group per distinct name. Mesh sizes are the `.geo`
  constants `h_max`, `h_min` and (cases with charges) `h_charge`, `r_fine`, `r_coarse`:
  `h_charge > 0` adds a Distance + Threshold field around the charges. The caller sets them
  with `gmsh -setnumber`.
- **`.yaml`**: one `---` document per material (`epsilon_r`, `screening_length` in nm, or
  `band_structure: model: metal`). Numbers are written in their shortest exact form, so the
  importer reads back exactly the same doubles.
- **`.cha`**: the charges (float32 position, int32 value); empty without charges.

## Tests

`tests/test_verification.cpp`:

- every layered and image case matches its closed form from the verification plan
  (straight lines, two dielectrics, floating and prescribed metals, screened slabs, the Robin
  "infinite slab", two screened layers with the closed-form interface potential, shells,
  Coulomb and Yukawa balls with Dirichlet, Neumann and Robin ends, a dielectric ball in a
  shell, a floating metal cage);
- finite-difference checks that each formula satisfies $-\varepsilon_r \Delta u + c u = 0$,
  that `grad_u` is its gradient, and the interface and boundary conditions;
- invalid settings and problems without a unique solution are rejected;
- the generated `.yaml`, `.geo` names and `.cha` bytes;
- with gmsh installed: each case is meshed and imported and matches its case
  (`mesh_mismatches` is empty), including thin layers, charges with a refined mesh around
  them, and a thin shell refined twice with AMR; meshes of another case are caught (other
  `eps_r`, swapped end or shell conditions, other layer thickness or box size, a too small
  ball, a missing boundary triangle, a second volume of a material with another voltage,
  screening off at import).

`tests/test_verification_runner.py` runs the chain through the Python stages (meshes of
several cases and sizes, caching, saved runs keeping their case, the gmsh `.geo` guard, the
CLI not publishing part of its outputs).
