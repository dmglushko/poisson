#pragma once

#include "free_charges.h"
#include "mesh_properties.h"

#include <filesystem>
#include <optional>
#include <vector>

// Reads the input files and builds MeshProperties and FreeCharges.
//
// Invalid input throws std::runtime_error with the file name (and line number for text
// files). Things that are suspicious but usable are recorded in info.warnings.
class Importer {
public:
    // mesh_file:      Gmsh 2.2 ASCII mesh (.msh / .msh2) with tetrahedra, boundary triangles
    //                 and $PhysicalNames.
    // material_files: material .yaml files; eps_r is taken from "dielectric: epsilon_r" and
    //                 assigned to the volume with the same name.
    // charges_file:   initial trapped charges (.cha); optional.
    MeshProperties import_mesh(const std::filesystem::path& mesh_file,
                               const std::vector<std::filesystem::path>& material_files = {},
                               const std::optional<std::filesystem::path>& charges_file = std::nullopt) const;

    // primaries_file: primary electrons (.pri).
    FreeCharges import_free_charges(const std::filesystem::path& primaries_file) const;
};
