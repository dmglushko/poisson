# Importer

Reads the input files and returns `MeshImportResult { mesh, info }` and
`FreeChargesImportResult { charges, info }` (`include/importer/importer.h`,
`src/importer/importer.cpp`).

## Interface

```cpp
struct MeshImportResult {
    struct Info {
        double                   time = 0;            // total, in seconds
        double                   time_read = 0;       // reading the .msh
        double                   time_geometry = 0;   // checking the geometry
        std::vector<std::string> warnings;            // e.g. a volume without a material file
    };

    MeshProperties mesh;
    Info           info;
};

struct FreeChargesImportResult {
    struct Info {
        double                   time = 0;            // total, in seconds
        std::vector<std::string> warnings;
    };

    FreeCharges charges;
    Info        info;
};

class Importer {
public:
    struct Settings {
        bool screened = true;   // copied into MeshProperties::screened
    };

    Importer() = default;
    explicit Importer(Settings settings);

    MeshImportResult import_mesh(const std::filesystem::path& mesh_file,
                                 const std::vector<std::filesystem::path>& material_files = {},
                                 const std::optional<std::filesystem::path>& charges_file = std::nullopt) const;

    FreeChargesImportResult import_free_charges(const std::filesystem::path& primaries_file) const;
};
```

## Input formats

| Input | Goes into |
|---|---|
| `.msh` / `.msh2` (Gmsh 2.2 ASCII) | `nodes`, `elements`, `boundary_elements`, `neighbours` |
| surface names in the mesh file | `boundary_conditions`, see below |
| volume names in the mesh file | `materials`; a voltage in the name (`"silicon 5V"`, `"metal 1.5e-3V"`) gives `voltage` |
| material `.yaml` | for the material with the same name: `eps_r` (`dielectric: epsilon_r`), `screening_length` (`dielectric: screening_length`, a number in nm or with a unit `nm`, `m`, `um`, `A`), and `is_metal` (`band_structure: model: metal`). One file may hold several materials, each a YAML document started by a `---` line |
| material `.mat` (Nebula, HDF5) | the same from the properties `epsilon_r` and `conductor_type`; `screening_length` from a property of that name (assumed, not in Nebula's format) |
| `.cha` | `MeshProperties::charges` |
| `.pri` | `FreeCharges::electrons` |

Surface names give the boundary condition, with the convention
$\varepsilon_r \partial_n u + \alpha u = g$ of `MeshProperties`:

| Name | Condition |
|---|---|
| `"<u>V"`, e.g. `"0V"`, `"-1.5V"` | Dirichlet, `value` = u [V] |
| `"D<g>"`, e.g. `"D0"` | Neumann, `value` = g [V/nm] |
| `"R<alpha>"`, e.g. `"R0.5"` | Robin with g = 0, `alpha` [1/nm] |
| `"R<alpha>,<g>"`, e.g. `"R0.5,0.1"` | Robin, `alpha` [1/nm] and `value` = g [V/nm] |

A negative `alpha` is an error; `alpha = 0` (the same as Neumann) is a warning.

`Settings::screened` (default `true`; `poisson_import --screening on|off`) is copied into
`MeshProperties::screened`. Turning it off drops the screened term even where a material
has a screening length.

## Rules and checks

- Invalid input throws `std::runtime_error`. Input that is suspicious but usable is
  reported in `info.warnings`.
- Gmsh node ids and physical tags are renumbered to indices starting at 0.
- A tet with a negative volume gets two of its nodes swapped.
- A detector with a voltage (`"detector 10V"`) is a metal, as in Nebula.
- The `.tri` file is not read; it is generated from the mesh.
- A warning says the solution is only fixed up to a constant when nothing prescribes the
  potential: no used Dirichlet surface, no used Robin surface with `alpha > 0`, no volume
  with a voltage, and (when screened) no non-metal volume with a screening length.
- A material given twice, in one file or in two, is an error.
