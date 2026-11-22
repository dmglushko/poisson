#pragma once

#include "free_charges.h"
#include "mesh_properties.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

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

// Reads the input files and builds MeshProperties and FreeCharges.
//
// Invalid input throws std::runtime_error with the file name (and line number for text
// files). Things that are suspicious but usable are recorded in result.info.warnings.
class Importer {
public:
    // mesh_file:      Gmsh 2.2 ASCII mesh (.msh / .msh2) with tetrahedra, boundary triangles
    //                 and $PhysicalNames.
    // material_files: material .yaml or .mat files; eps_r and conductor type are assigned to
    //                 the volume with the same name.
    // charges_file:   initial trapped charges (.cha); optional.
    MeshImportResult import_mesh(const std::filesystem::path& mesh_file,
                                 const std::vector<std::filesystem::path>& material_files = {},
                                 const std::optional<std::filesystem::path>& charges_file = std::nullopt) const;

    // primaries_file: primary electrons (.pri).
    FreeChargesImportResult import_free_charges(const std::filesystem::path& primaries_file) const;
};
