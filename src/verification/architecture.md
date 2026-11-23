# Verification

Problems with a known exact solution, to check the pipeline against
(`include/verification/`, `src/verification/`, library `poisson_verification`). A case
defines everything about its problem in one C++ class: the geometry (as a gmsh `.geo`
script), the materials, the boundary conditions, the trapped charges and the exact potential
with its gradient. So the mesh a case produces and the formula it is checked against cannot
drift apart. Python only names a case and passes its parameters
(`python/poisson_runner/cases.py`).

Nothing here is part of the solve. Only the three verification CLIs link this library, so
adding or changing a case does not change the code hash of the import, AMR or solver stages.

## Flow

```text
Case(name, params) --poisson_case--> case.geo + case.yaml + case.cha
                   --gmsh--> case.msh --poisson_import--> mesh.h5
mesh.h5 + solution (NodalField)                  --poisson_diff--> per-tet/per-node errors + norms (info)
mesh.h5 + solution + points [+ case]             --poisson_sample--> PointSamples
mesh.h5 + case                                   --poisson_case--> exact NodalField (interpolant)
```

The exact interpolant stands in for a solution until Assembly and the Solver exist, and is
the interpolation-error baseline after that.

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

std::optional<MeshProperties::BoundaryCondition> expected_condition(triangle) const;   // NVI
bool fits(tet, material) const;                                                        // NVI

std::vector<std::string> mesh_mismatches(const MeshProperties&, const VerificationCase&);
```

- `u` and `grad_u` take the **material** of the point, not just its position: at an
  interface the potential is continuous but its gradient is not, and a faceted mesh of a
  curved interface puts some tets slightly on the other side. The formula of the tet's
  material is used, evaluated where the point is.
- Each case has a nested `Settings` validated in its constructor (`std::invalid_argument`,
  including a problem without a unique solution).
- Charge positions are rounded to float32 in the constructor, so the formula, the `.cha`
  file and the imported mesh agree bit for bit.
- `mesh_mismatches` compares an imported mesh with the case: the screened flag, every
  material by name (metal, voltage, and for dielectrics `eps_r` and `screening_length`), the
  set of boundary conditions, the trapped charges, and where things are:
  - every boundary triangle has the condition the case puts there (`expected_condition`;
    a triangle that is not on the case's boundary at all is reported too), so swapped end
    conditions are caught;
  - every tet lies in its material's region (`fits`), so a different layer thickness or
    box size is caught;
  - every outer tet face has a boundary condition.

  Flat faces and interfaces are checked to `1e-9` of the case size. A meshed sphere is a
  polyhedron inside the sphere whose vertices lie on it (or, after refinement, on its flat
  facets), so radii are bounded exactly from above and only within half the neighbouring
  layer from below; a triangle belongs to the nearer of the inner and outer spheres.

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
screening) the case is rejected as having no unique solution. Otherwise the system is solved
and a large residual is reported as too ill-conditioned. A layer with `k L < 1` uses the
centred pair `cosh(k t)`, `sinh(k t) / k` (`t` from the layer middle; `1`, `t` at `k = 0`)
instead of the two exponentials, which become indistinguishable as `k -> 0`; this keeps
very weak screening (screening length far above the size) accurate. A metal layer at an end needs Neumann 0 on that end face, and two
adjacent metals must be merged into one layer.

### Parameters (JSON)

`poisson_case`, `poisson_sample` and `poisson_diff` read a case as
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
(`e^-20`); without screening it is not, and errors below it mean nothing. The bound is in the
`poisson_case` info and a warning of `poisson_diff`.

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

`interpolate_exact(mesh, case)` gives the exact potential at every node (`ExactInterpolationResult`).
A node takes the formula of one of its tets' materials, a metal one if it has any. A node on
a charge (within `1e-9` of the mesh's bounding box diagonal) is set to 0 with a warning.

## Error evaluation

`ErrorEvaluator(Settings{exclusion_radius, quadrature_degree})::evaluate(mesh, u, case)`
returns `ErrorEvaluationResult`:

- outputs: `element_l2_squared` and `element_h1_squared` (per tet, the integrals over its
  measured part) and `node_error` (`u_h - u` per node, NaN where not measured);
- `Info`: `l2_error`, `h1_error` (seminorm), `max_error` (nodes), `l2_exact` and `h1_exact`
  over the same region (for relative errors), `metal_potential_error`, `measured_volume`,
  `excluded_points`, `metal_elements`, `exclusion_radius`. The norms are sums of the outputs.

Rules:

- `u_h` is the P1 field of the nodal values; the exact values come from the case with the
  tet's material at tet quadrature points of the chosen degree (default 4).
- Metal tets are skipped: no equation is solved inside a metal. The potential of metal nodes
  is checked separately as `metal_potential_error` (this is where a floating metal's
  potential is tested).
- Quadrature points and nodes closer than the exclusion radius to a charge are not measured
  (P1 cannot follow `1/r`), and so is a quadrature point exactly on a charge. The radius is
  in nm, not in elements, so meshes of different sizes measure the same region. The default
  is the case's `default_exclusion_radius`: `min(1, half the distance to the nearest
  boundary or interface)`. With radius 0 in a case with charges, a warning says that the
  H1 and nodal numbers depend on where points fall (the exact H1 seminorm is infinite).
- A field with a non-finite value is rejected (`std::invalid_argument`), as by the sampler.

## Point sampling

`FieldSampler::sample(mesh, u, points, case = nullptr)` locates each point with `MeshLocator`
(walking from the previous point's tet) and returns `PointSamples`: `found`, `element`,
`material`, `u`, the P1 gradient `grad`, and with a case `u_exact` and `grad_exact` from the
material of the same located tet, so a plot of both never mixes the two sides of an
interface. Points outside the mesh get NaN and are counted in the info.
`FieldSampler::line(from, to, n)` gives `n` equally spaced points.

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
  (`mesh_mismatches` is empty); the interpolant of a piecewise linear case with a floating
  metal has zero error, including the metal potential; a constant offset gives exactly that
  error; the screened slab and a charged ball converge at second order in L2; sampling
  returns the exact values; meshes of another case, or imported unscreened, are caught.

`tests/test_verification_runner.py` runs the same chain through the Python stages, including
verification of a saved run in a separate evaluation.
