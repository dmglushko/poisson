# Importer

Reads the input files and returns `MeshImportResult { mesh, info }` and
`FreeChargesImportResult { charges, info }` (`include/importer.h`, `src/importer.cpp`).

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
| surface names in the mesh file | `boundary_conditions`: `"0V"` is Dirichlet, `"D0"` is Neumann |
| volume names in the mesh file | `materials`; a voltage in the name (`"silicon 5V"`) gives `voltage` |
| material `.yaml` | for the material with the same name: `eps_r` (`dielectric: epsilon_r`) and `is_metal` (`band_structure: model: metal`) |
| material `.mat` (Nebula, HDF5) | the same from the properties `epsilon_r` and `conductor_type`; `screening_length` from a property of that name (assumed, not in Nebula's format) |
| `.cha` | `MeshProperties::charges` |
| `.pri` | `FreeCharges::electrons` |

## Rules and checks

- Invalid input throws `std::runtime_error`. Input that is suspicious but usable is
  reported in `info.warnings`.
- Gmsh node ids and physical tags are renumbered to indices starting at 0.
- A tet with a negative volume gets two of its nodes swapped.
- A detector with a voltage (`"detector 10V"`) is a metal, as in Nebula.
- The `.tri` file is not read; it is generated from the mesh.
