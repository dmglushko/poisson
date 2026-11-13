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
- Every data class has a nested `Info` describing the run that produced it. Each `Info` has
  at least `time` and `warnings`; the other fields differ per class.

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
        double value;                             // Dirichlet: u [V]; Neumann and Robin: g
        std::optional<double> alpha;              // Robin only: eps * du/dn + alpha * u = g
    };

    struct Material {
        std::string name;                         // e.g. "silicondioxide"
        double eps_r = 1;                         // relative permittivity
        double screening_length = 0;              // nm; 0 = no screening, else c = eps_r / screening_length^2
        bool is_metal = false;                    // conductor: one potential everywhere inside
        std::optional<double> voltage;            // prescribed potential [V]; empty if not prescribed
    };

    struct Charge {                               // trapped charge
        std::array<double, 3> position;           // x, y, z in nm
        int32_t               value;              // in elementary charges: +1 hole, -1 electron
    };

    struct Info {
        double time = 0;                          // total, in seconds
        double time_read = 0;                     // reading the .msh
        double time_geometry = 0;                 // checking the geometry
        std::vector<std::string> warnings;        // e.g. a volume without a material file
    };

    std::vector<std::array<double, 3>> nodes;               // nodes[id] = {x, y, z} in nm
    std::vector<Element>               elements;
    std::vector<BoundaryElement>       boundary_elements;
    std::vector<BoundaryCondition>     boundary_conditions;
    std::vector<Material>              materials;
    std::vector<Charge>                charges;             // all trapped charges; grows after every batch

    // PROPOSED, not fully decided: the tets around each tet. neighbours[e][i] is the tet on
    // the other side of the face opposite node i of element e, or empty if that face is on
    // the outside of the mesh. Same index as elements.
    // std::optional doubles the memory here (32 instead of 16 bytes per tet). If that turns
    // out to matter, a plain uint32_t with a reserved "no neighbour" value may be better.
    std::vector<std::array<std::optional<uint32_t>, 4>> neighbours;
    Info                               info;
};
```

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

    struct Info {
        double time = 0;                        // total, in seconds
        std::vector<std::string> warnings;
    };

    std::vector<Electron> electrons;
    Info                  info;
};
```

### AssembledProblem

The linear system A x = f.

A has one row per unknown, not one per node. `row` gives the row of each node; several
nodes may share a row. Nodes whose value is known in advance have no row and are listed in
`known`. The potential at every node follows from the solution `x` of the system
(`SolverResult::x`):

- node with a row: `u[i] = x[*row[i]]`
- every `(node, value)` in `known`: `u[node] = value`

```cpp
struct AssembledProblem {
    struct Info {
        double   time = 0;                      // total, in seconds
        double   time_assemble_A = 0;
        double   time_assemble_f = 0;
        uint64_t nnz = 0;                       // nonzeros in A
        uint32_t n_dirichlet = 0;               // eliminated nodes
        std::vector<std::string> warnings;
    };

    SparseMatrix                             A;       // storage format still open
    std::vector<double>                      f;       // right-hand side, one entry per row of A
    std::vector<std::optional<uint32_t>>     row;     // per node: its row in A, or empty if its value is known
    std::vector<std::pair<uint32_t, double>> known;   // (node, value) for every node whose value is known
    Info                                     info;
};
```

### SolverResult

What the solver returns: the solution `x` of A x = f.

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

Reads the input files and builds `MeshProperties` and `FreeCharges`.

```cpp
class Importer {
public:
    MeshProperties import_mesh(const std::filesystem::path& mesh_file,
                               const std::vector<std::filesystem::path>& material_files = {},
                               const std::optional<std::filesystem::path>& charges_file = std::nullopt) const;

    FreeCharges import_free_charges(const std::filesystem::path& primaries_file) const;
};
```

| Input | Goes into |
|---|---|
| `.msh` / `.msh2` (Gmsh 2.2 ASCII) | `nodes`, `elements`, `boundary_elements`, `neighbours` |
| surface names in the mesh file | `boundary_conditions`: `"0V"` is Dirichlet, `"D0"` is Neumann |
| volume names in the mesh file | `materials`; a voltage in the name (`"silicon 5V"`) gives `voltage` |
| material `.yaml` | for the material with the same name: `eps_r` (`dielectric: epsilon_r`) and `is_metal` (`band_structure: model: metal`) |
| `.cha` | `MeshProperties::charges` |
| `.pri` | `FreeCharges::electrons` |

- Invalid input throws `std::runtime_error`. Input that is suspicious but usable is
  reported in `info.warnings`.
- Gmsh node ids and physical tags are renumbered to indices starting at 0.
- A tet with a negative volume gets two of its nodes swapped.
- A detector with a voltage (`"detector 10V"`) is a metal, as in Nebula.
- The `.tri` file is not read; it is generated from the mesh.

### Solver

Solves A x = f. It takes `A` and `f` from `AssembledProblem` and returns a `SolverResult`.

### Solution Builder

Builds `Solution` from `SolverResult::x`, using `row` and `known` from `AssembledProblem`.

### Not defined yet

Assembly, AnalyticalMaker, Adaptive Mesh Refining and Ray Tracing.

## Open points

- `Analytical Solution`: contents not decided. Planned as singularity subtraction:
  analytical functions around the trapped charges.
- `Solution`: contents not decided; it depends on the shape of `Analytical Solution`.
- `Output`: contents not decided.
- `SparseMatrix`: storage format not decided; an empty placeholder struct in the code.
- `neighbours` in `MeshProperties`: proposed, not decided.
