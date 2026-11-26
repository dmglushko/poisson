# Adaptive mesh refinement

The module that improves the mesh after a solve. It is three steps, each an interface
with interchangeable methods:

```mermaid
flowchart LR
    M([MeshProperties])
    IN([RefinementInput])
    EST[ErrorEstimator]
    E([ErrorEstimate])
    MARK[MarkingStrategy]
    K([marked tets])
    REF[Refiner]
    H([RefinementHistory])
    R([RefinementResult])

    M --> EST
    IN --> EST
    EST --> E
    E --> MARK
    M --> MARK
    MARK --> K
    K --> REF
    M --> REF
    H --> REF
    REF --> R
    R -. next cycle: mesh, history .-> M
```

| Step | Interface | Methods | Header |
|---|---|---|---|
| adapt (top-level) | `MeshAdapter::start()` → `Session::adapt(mesh, input)` → `AdaptationResult` | `ErrorBasedAdapter` (extensible to metric-based, etc.) | `include/amr/mesh_adapter.h`, `include/amr/error_based_adapter.h` |
| estimate | `ErrorEstimator::estimate(mesh, input)` → `ErrorEstimate` | `ResidualEstimator`, `KellyEstimator`, `RecoveryEstimator`, `TrajectoryEstimator` | `include/amr/error_estimator.h` and one header per method |
| mark | `MarkingStrategy::mark(mesh, eta, sum)` → sorted tet indices | `DorflerMarking`, `MaximumMarking`, `FixedFractionMarking`, `EquidistributionMarking` | `include/amr/marking.h` |
| refine | `Refiner::refine(mesh, marked, history)` → `RefinementResult` | `LongestEdgeBisection`, `RedGreenRefinement` | `include/amr/refiner.h` and one header per method |

The top-level solver loop (`PoissonDriver`) is sketched as a placeholder in
`include/poisson_driver.h` until `AnalyticalMaker`, `Assembly`, `Solver`, and `Solution`
contracts exist.

## Top-level `MeshAdapter` and `Session`

`MeshAdapter` is the virtual boundary between the outer solver driver and any mesh
adaptation method. A configured `MeshAdapter` is `const` and reusable; calling `start()`
returns a `MeshAdapter::Session` that holds per-run state across cycles and releases it on
destruction.

`ErrorBasedAdapter` composes `(ErrorEstimator, MarkingStrategy, Refiner)` with its own
stopping settings (`absolute_tolerance`, `relative_tolerance`, `max_elements`,
`estimate_at_limit`). A future metric-based adapter implements `MeshAdapter` directly
without using `ErrorEstimator`, `MarkingStrategy`, or `RefinementHistory`.

### What one `adapt()` call returns

```cpp
struct ErrorBasedExtras {
    std::vector<double> eta;      // per tet of the INPUT mesh
};
using AdaptationExtras = std::variant<std::monostate, ErrorBasedExtras>;

struct AdaptationResult {
    AdaptationStatus              status;   // adapted / converged / limit_reached / unchanged
    std::optional<MeshProperties> mesh;     // iff adapted
    std::optional<Parents>        parents;  // iff adapted (new mesh -> input mesh)
    AdaptationExtras              extras;   // adapter-specific per-cycle outputs, if any
    Info                          info;     // metadata only: time, counts, warnings, stages
};
```

Adapter-specific per-cycle data goes in the `extras` variant so top-level `AdaptationResult`
fields stay strictly common across all adapters. For `ErrorBasedAdapter`, `extras` holds
`ErrorBasedExtras{eta}` (computed on the input mesh before any refinement; its norm
$\sqrt{\sum_K \eta_K^2}$, or $\sum_K \eta_K$ for an estimator with `IndicatorSum::values`, is the `total` metric of the `estimate` entry in `info.stages`) on every outcome where the estimator ran (`adapted`,
`converged`, `unchanged`, and `limit_reached` with `estimate_at_limit = true`), and
`std::monostate` when estimation was skipped. Mesh-sized arrays (`mesh`, `parents`, `eta`)
are result data, never `Info`, so CLIs write them as `.h5` and `Info` as `.info.json`.

### Cross-cycle state: `ErrorBasedAdapter::State`

```cpp
struct ErrorBasedAdapter::State {
    RefinementHistory     history;        // red-green green families (empty otherwise)
    std::optional<double> initial_error;  // first estimated cycle's total, base of relative_tolerance
};
auto session = adapter.start();        // fresh: State{}
auto session = adapter.start(state);   // resume; returns the typed ErrorBasedAdapter::Session
session->state();                      // state after the latest successful adapt()
```

The session updates its state by itself, so an in-process loop never touches it. The state
is a plain value so that cycles can also run in separate processes (`poisson_adapt` runs one
cycle per call): save `session->state()` after a cycle and `start(state)` the next one; the
sequence is identical to one long session (`test_adapter` checks this). After each call the
state belongs to the mesh the caller continues with: the refined mesh after `adapted`,
otherwise the unchanged input mesh. `State` lives on `ErrorBasedAdapter`, not on the generic
`MeshAdapter`, because cross-cycle state is encapsulated inside `Session` in-process and only
leaves `Session` for concrete HDF5 serialization across CLI invocations.

## Shared shape of the interfaces

As in pde2d: the public function of each interface (`estimate`, `mark`, `refine`,
`Session::adapt`) is **not virtual**. It checks the input/postconditions, times the work,
applies the rules every method shares, and then calls a private virtual function with the
method's own work (`indicators`, `select`, `refine_marked`, `adapt_step`). So the checks
are written once and no method can skip them.

Each method has a nested `Settings` struct, given to its constructor and checked there
(`std::invalid_argument`). The default constructor (where meaningful) uses default settings.

Each result has a nested `Info` with typed fields for the common numbers (`time`, element
and node counts, `warnings`), plus a `details` map for
method-specific diagnostics (`element_part`, `red`, `passes`, ...). `AdaptationResult::Info`
also carries a `std::vector<StageInfo> stages` recording every internal stage that actually
ran (`estimate`, `mark`, `refine`, or whatever stages a future adapter uses). Following the
root convention, `Info` never repeats a result value: the error estimate is
`ErrorEstimate::total` (not in `ErrorEstimate::Info`), and in an `AdaptationResult` it is the
`total` metric of the `estimate` stage.

## Estimation

`RefinementInput` is the boundary to the rest of the code: `u` (one value per node,
Dirichlet nodes included), `source` (the smooth right-hand side f(x, element), the same f
that Assembly integrates) and `point_sources` (deltas that are still deltas). The
estimators do not know how the analytical solution is represented.

The equation they assume, per tet with its material's `eps_r` and `screening_length`:
`-div(eps grad u) + c u = f`, `eps = eps_r`, `c = eps_r / screening_length^2` when
`mesh.screened && screening_length > 0` (and `0` otherwise).

| | element term | face jumps | Neumann / Robin faces | uses `source` |
|---|---|---|---|---|
| residual | `h_K^2/eps_K ‖f − c u_h‖²` | `h_F/(2 eps_F) ‖J_F‖²` | `h_F/eps_K ‖R_F‖²` | yes |
| Kelly | — | same | same | no |
| recovery | `∫ eps |G − ∇u_h|²` (or the flux form) | — | — | no |
| trajectory | `|∫_K ∇e_K|`, `e_K` from a local problem (below) | load `−J_F/2` | load `R_F` | yes |

Shared rules (in `ErrorEstimator::estimate`):

- **Metal** tets get `eta = 0`; faces that touch a metal are skipped like Dirichlet faces.
- **Point sources**: `PointSourceRule::exclude_patch` (default) sets `eta = 0` on every tet
  that shares a node with the tet containing a non-zero delta, because near a delta every
  estimator keeps growing under refinement. `ignore` leaves them out. A source without an
  element hint is located with `MeshLocator` (`include/common/mesh_locator.h`).
- An outer face without a boundary element is natural (`g = 0`).

Recovery averages per material by default (`per_material = true`): neither `grad u` nor
`eps grad u` is continuous across an interface, so averaging across it would report error
on every interface tet forever.

### Indicator sum

Each estimator declares how its indicators add up (`IndicatorSum`, `include/amr/indicator_sum.h`):
`squares` ($\text{total}^2 = \sum_K \eta_K^2$, the norm estimators) or `values`
($\text{total} = \sum_K \eta_K$). `ErrorEstimate::sum` carries it, `ErrorBasedAdapter` passes it to
`MarkingStrategy::mark`, and the strategies that compare sums follow it: Dörfler accumulates
$\eta_K$ instead of $\eta_K^2$, equidistribution uses the bound `tolerance / N` instead of
`tolerance / sqrt(N)`. Maximum and fixed-fraction marking only rank, so they ignore it.

### Trajectory estimator

Mitchell and Villarrubia, *SEM simulation with adaptive mesh refinement*, Secs. 3.3–3.4. It
estimates each tet's contribution to an electron's landing-position error, which is linear in
the field error, so it uses `IndicatorSum::values`. On every tet the Bank–Weiser local problem
is solved in the six quadratic edge bubbles (a 6×6 system, no global solve): element load
`f − c u_h`, half the flux mismatch on interior faces, the full residual on Neumann/Robin faces,
`e_K = 0` on Dirichlet faces. Then $\eta_K = \text{scale}\,|\int_K \nabla e_K|$, the paper's
$h_K^3 |\langle\nabla e\rangle_K|$ with $h_K = |K|^{1/3}$. `scale` is the paper's $eR/(4TV)$. It
does not change what the relative rules do (Dörfler, maximum, fixed fraction,
`relative_tolerance`); with it, `total` estimates the relative landing-position error, so
`absolute_tolerance` and equidistribution's `tolerance` get a meaning. A face to a metal with
a prescribed voltage is a Dirichlet face for the local problem. The paper validates the indicator only without screening.

Every integral of a polynomial in the barycentric coordinates is computed in closed form,
$\int_K \prod_i \lambda_i^{a_i} = 3!\,|K| \prod_i a_i! / (3 + \sum_i a_i)!$ (and $2!\,|F|\cdots$ on a
face): the flux jumps and the Neumann/Robin residuals (constant and linear on a face), the
recovery difference (linear on a tet), and the trajectory bubble matrix, Robin block and face
loads. Only the integrals that contain the source `f` (an arbitrary function) use quadrature,
from `include/common/quadrature.h` (see [src/common/architecture.md](../common/architecture.md);
in `common/` because Assembly needs it too), with the degree set by the estimator's
`tet_degree`.

## Marking

The rules of pde2d unchanged: indicators must be finite and non-negative, zeros are never
marked, the result is sorted and unique, ties go by tet index.

## Refinement

Both refiners share one operation (`src/amr/refinement_common.*`): replace a tet by
children given in its local terms (nodes and edge midpoints). From that local description
the children's faces inherit the boundary elements they lie in, so boundary triangles are
split exactly like the tets and keep their condition and orientation. A child keeps its
parent's material; children are turned to positive volume; `neighbours` is rebuilt.

`Parents` gives, per new tet, the input tet that contains it, and per new node the two
nodes it is the midpoint of.

| | longest-edge bisection | red-green |
|---|---|---|
| unit step | bisect at the longest edge (2 children) | red: 8 children; green: 2, 3 or 4 |
| conformity | longest-edge propagation, in passes | green closure, red where no green pattern fits |
| state between cycles | none | `RefinementHistory`: the green families |
| shape | bounded in practice; not proven in 3D | bounded (green tets are never refined further) |

`RefinementHistory` exists only for red-green: green tets must not be subdivided again, so
the next refinement has to know which tets are green and what their parent was. A direct
`Refiner::refine` caller passes the previous result's history back with its mesh;
`ErrorBasedAdapter` keeps it in its `State`.

## Tests

`tests/test_estimators.cpp`, `test_marking.cpp`, `test_refiners.cpp`, `test_adapter.cpp`,
`test_quadrature.cpp`. With no solver yet, the estimators and `ErrorBasedAdapter` are tested
on interpolants: exact piecewise linear functions (including a kink at a material interface
with continuous flux) must give 0, and a smooth function must give O(h). The refiners are
checked after every cycle for positive volumes, conformity, boundary conditions per area,
orientation, neighbours and parents, on a structured and on a jittered mesh. `test_adapter`
verifies session-scoped red-green history, that resuming every cycle from a saved `State`
reproduces one long session (meshes, statuses, reopened green families, `relative_tolerance`
convergence), `eta` on every outcome, absolute/relative tolerances, `max_elements` with and
without `estimate_at_limit`, `StageInfo` collection, and the `PoissonDriver` placeholder.

## Open points

- The weights `1/eps` and `eps_F = max(eps_K, eps_L)` in the residual and Kelly
  estimators, and `eps = eps_r` without `eps_0`: they must match Assembly's units.
- Faces between a dielectric and a floating metal (no voltage) are skipped like Dirichlet
  faces.
- An interior face with a Neumann or Robin boundary element is treated as an ordinary
  interior face, with a warning. Only meshes built through the C++ API can reach this, since
  the importer rejects boundary elements on interior faces.
- On a face shared between an excluded point-source tet and a non-excluded neighbour, the
  non-excluded neighbour still receives half of that face's flux jump.
- No coarsening, no maximum refinement level.
