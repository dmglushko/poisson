# Architecture

## Data flow

Rectangles are steps, rounded nodes are data.

```mermaid
flowchart LR
    IMP[Importer]
    AMR[Adaptive Mesh Refining]
    AM[AnalyticalMaker]
    ASM[Assembly]
    SOL[Solver]
    SB[Solution Builder]
    RT[Ray Tracing]

    M([MeshProperties])
    CH([FreeCharges])
    AS([Analytical Solution])
    AP([AssembledProblem])
    SR([SolverResult])
    S([Solution])
    OUT([Output])

    IMP --> M
    IMP --> CH
    CH --> RT
    AMR --> M
    M --> ASM
    M --> AM
    AM --> AS
    AS --> ASM
    ASM --> AP
    AP --> SOL
    SOL --> SR
    SR --> SB
    AP --> SB
    AS --> SB
    SB --> S
    S --> AMR
    S --> RT
    RT --> OUT
```

## Conventions

- Classes are CamelCase, variables are snake_case.
- **Data classes** (`MeshProperties`, `FreeCharges`, `AssembledProblem`, and later
  `Solution`) hold only problem data, never an `Info`. The same data class can come from
  different stages (a `MeshProperties` from the Importer or from AMR) or from disk, so how
  one run went cannot be part of it.
- **Stage results** (`MeshImportResult`, `FreeChargesImportResult`, `AssemblyResult`,
  `SolverResult`, `ErrorEstimate`, `RefinementResult`, `AdaptationResult`) are
  `{ outputs...; Info info; }`:
  - **outputs** are everything a later stage or the driver reads: the data classes, plain
    arrays such as `x` or `eta`, helpers such as `parents`, and control values such as
    `AdaptationResult::status`. CLIs write them as `.h5`.
  - **`Info`** is only for people and logs: at least `time` and `warnings`, plus counts,
    timings, a `details` map or `stages`. Summaries derived from outputs (`nnz`,
    `max_indicator`, the refined mesh size) are fine, but `Info` never repeats a value that
    is already an output, and never reports the same quantity twice (under two names, or
    in both a composite's own fields and its `stages`). CLIs write it as `.info.json`.

  A quick test: if deleting a field would break a later stage, it is an output; if only
  the `.info.json` would change, it belongs in `Info`. One exception in the CLIs: a small
  control output such as `AdaptationResult::status` may be written into the `.info.json`
  next to the `Info` instead of into an `.h5`, so that an orchestrator can read it without
  opening one. In C++ it stays an output. Such a value is then not covered by the runner's
  `.h5` determinism check.
- Drivers and composite stages are preferably built by composing smaller interchangeable
  module interfaces.
- When a module or driver needs configuration, the preferred convention is a nested
  `Settings` struct passed to its constructor (with driver-level settings kept separate
  from sub-module settings).

## Data classes

### MeshProperties

The mesh with its materials, boundary conditions and trapped charges. Elements and boundary
elements store only indices; coordinates, materials and boundary conditions are looked up
in the other vectors.

```cpp
struct MeshProperties {
    struct Element {                              // volume element (tet)
        std::array<uint32_t, 4> node;             // indices into nodes
        uint16_t material;                        // index into materials
    };

    struct BoundaryElement {                      // surface element (triangle)
        std::array<uint32_t, 3> node;             // indices into nodes
        uint16_t boundary_condition;              // index into boundary_conditions
    };

    enum class BcType { Dirichlet, Neumann, Robin };
    struct BoundaryCondition {
        BcType type;
        double value;                             // Dirichlet: u [V]; Neumann and Robin: g [V/nm]
        std::optional<double> alpha;              // Robin only [1/nm]: eps_r * du/dn + alpha * u = g
    };

    struct Material {
        std::string name;                         // e.g. "silicondioxide"
        double eps_r = 1;                         // relative permittivity
        double screening_length = 0;              // nm; 0 = no screening, else c = eps_r / screening_length^2
        bool is_metal = false;                    // conductor: one potential everywhere inside
        std::optional<double> voltage;            // prescribed potential [V]; empty if not prescribed
    };

    struct TrappedCharge {
        std::array<double, 3> position;           // x, y, z in nm
        int32_t               value;              // in elementary charges: +1 hole, -1 electron
    };

    // neighbours entry for a face on the outside of the mesh.
    static constexpr uint32_t no_neighbour = std::numeric_limits<uint32_t>::max();
    static constexpr std::array<uint32_t, 4> no_neighbours{no_neighbour, no_neighbour, no_neighbour, no_neighbour};

    std::vector<std::array<double, 3>> nodes;               // nodes[id] = {x, y, z} in nm
    std::vector<Element>               elements;
    std::vector<BoundaryElement>       boundary_elements;
    std::vector<BoundaryCondition>     boundary_conditions;
    std::vector<Material>              materials;
    std::vector<TrappedCharge>         charges;             // all trapped charges; grows after every batch
    bool                               screened = true;     // true: solve screened Poisson (-div(eps_r grad u) + c u = f); false: unscreened (c = 0)

    // The tets around each tet. neighbours[e][i] is the tet on the other side of the face
    // opposite node i of element e, or no_neighbour if that face is on the outside of the
    // mesh. Same index as elements; always filled (by the Importer and every refiner).
    std::vector<std::array<uint32_t, 4>> neighbours;
};
```

#### Units and PDE convention

All stages (`Importer`, `AnalyticalMaker`, `Assembly`, `Solver`, `AMR`, `Ray Tracing`) share one unit system (matching Nebula):

- **Length:** nanometers (`nm`). Coordinates in `.msh`, `.cha`, and `.pri` are read as-is in `nm`; `.mat` `screening_length` values are converted to `nm`.
- **Potential:** volts (`V`).
- **Permittivity:** dimensionless relative permittivity $\varepsilon_r$ (without $\varepsilon_0$ on the left-hand side).
- **Equation:** inside each non-metal tet, with $c = \varepsilon_r / \lambda^2$ when `mesh.screened && screening_length > 0` (and $c = 0$ otherwise):
  $$-\nabla \cdot (\varepsilon_r \nabla u) + c u = f = \frac{\rho}{\varepsilon_0}$$
  so $f$ has units $\text{V}/\text{nm}^2$, and a point charge of $q$ elementary charges contributes $q \cdot \frac{e}{\varepsilon_0\,\mathrm{nm}} \approx 18.0951\,q\;\text{V}\cdot\text{nm}$ to the integrated right-hand side.
- **Boundary conditions:** Dirichlet prescribes $u\;\text{[V]}$; Neumann and Robin satisfy $\varepsilon_r \frac{\partial u}{\partial n} + \alpha u = g$ with $g$ in $\text{V}/\text{nm}$ and $\alpha$ in $\text{nm}^{-1}$ ($\alpha = 0$ for Neumann).

### FreeCharges

The primary electrons, passed to the ray tracer. One entry holds what one record of a
`.pri` file holds.

```cpp
struct FreeCharges {
    struct Electron {
        std::array<double, 3>  position;    // x, y, z in nm
        std::array<double, 3>  direction;   // direction of motion, not normalised
        double                 energy;      // kinetic energy in eV
        std::array<int32_t, 2> pixel;       // image pixel this electron belongs to
    };

    std::vector<Electron> electrons;
};
```

### AssembledProblem

The linear system A x = f.

A has one row per unknown, not one per node. `row` gives the row of each node; several
nodes may share a row (all nodes of one connected floating metal share one row). Nodes
whose value is known in advance (Dirichlet boundaries and prescribed-voltage metals) have
no row and are listed in `known`. The potential at every node follows from the solution `x`
of the system (`SolverResult::x`):

- node with a row: `u[i] = x[*row[i]]`
- every `(node, value)` in `known`: `u[node] = value`

```cpp
struct AssembledProblem {
    Eigen::SparseMatrix<double>              A;
    std::vector<double>                      f;       // right-hand side, one entry per row of A
    std::vector<std::optional<uint32_t>>     row;     // per node: its row in A, or empty if its value is known
    std::vector<std::pair<uint32_t, double>> known;   // (node, value) for every node whose value is known
};
```

## Stage results

Results that cross a stage boundary through the root headers. The others are defined in
their module's header (`include/importer/importer.h`, `include/amr/*.h`).

### AssemblyResult

What Assembly returns (`include/assembled_problem.h`).

```cpp
struct AssemblyResult {
    struct Info {
        double   time = 0;                      // total, in seconds
        double   time_assemble_A = 0;
        double   time_assemble_f = 0;
        uint64_t nnz = 0;                       // nonzeros in A
        uint32_t n_dirichlet = 0;               // eliminated nodes
        std::vector<std::string> warnings;
    };

    AssembledProblem problem;
    Info             info;
};
```

### SolverResult

What the solver returns: the solution `x` of A x = f. `x` is a plain array, so it needs no
data class of its own.

```cpp
struct SolverResult {
    struct Info {
        double   time = 0;                      // total, in seconds
        uint32_t iterations = 0;
        double   residual = 0;                  // final relative residual
        std::vector<std::string> warnings;      // e.g. no convergence within the iteration limit
    };

    std::vector<double> x;                      // one value per row of A
    Info                info;
};
```

## Steps

### Importer

Reads the input files (`.msh`/`.msh2`, material `.yaml`/`.mat`, `.cha`, `.pri`) and returns
`MeshImportResult { mesh, info }` and `FreeChargesImportResult { charges, info }`
(`include/importer/importer.h`, `src/importer/importer.cpp`). Described in
[src/importer/architecture.md](src/importer/architecture.md).

### Solver

Solves A x = f. It takes `A` and `f` from `AssembledProblem` (Assembly's
`AssemblyResult::problem`) and returns a `SolverResult`.

### Solution Builder

Builds `Solution` from `SolverResult::x`, using `row` and `known` from `AssembledProblem`.

### Adaptive Mesh Refining

Top-level virtual `MeshAdapter` (`include/amr/mesh_adapter.h`, session-based) and its
composite `ErrorBasedAdapter` (`include/amr/error_based_adapter.h`), built from three
interchangeable step interfaces (`ErrorEstimator`, `MarkingStrategy`, `Refiner`). Its input
is `MeshProperties` and a `RefinementInput` (u at every node, the source f as a function,
point sources); its output is an `AdaptationResult`. Described in
[src/amr/architecture.md](src/amr/architecture.md).

### Common primitives

Stage-agnostic numerical and geometric helpers (`include/common/`, `src/common/`):
quadrature rules on tetrahedra and triangles (`quadrature.h`) and a BVH + neighbour-walk
spatial lookup index over `MeshProperties` (`mesh_locator.h`). Described in
[src/common/architecture.md](src/common/architecture.md).

### Verification

Not a pipeline stage: cases with a known exact solution (geometry, materials, boundary
conditions, charges and formula in one C++ class) and the CLI that turns a case into input
files (`include/verification/`, `src/verification/`). Comparing a solution with them waits
for `Solution`. Described in [src/verification/architecture.md](src/verification/architecture.md).

### Driver (placeholder)

`PoissonDriver` (`include/poisson_driver.h`) is a placeholder header sketching the top-level
solve loop over `AnalyticalMaker`, `Assembly`, `Solver`, `SolutionBuilder`, and an optional
`MeshAdapter`. To be implemented once those contracts are defined.

### Not defined yet

Assembly, AnalyticalMaker and Ray Tracing.

## Open points

- `Analytical Solution`: contents not decided. Planned as local singularity subtraction (P3M): a smooth cloud per charge in f, plus an analytical correction within r_c of each charge.
  analytical functions around the trapped charges.
- `Solution`: contents not decided; it depends on the shape of `Analytical Solution`. It
  is a data class, so the Solution Builder returns it wrapped with its `Info`
  (`SolutionBuildResult { solution, info }`).
- `Output`: contents not decided.
