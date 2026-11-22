#pragma once

#include "importer/importer.h"
#include "mesh_properties.h"

#include <filesystem>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for MeshProperties.
namespace poisson::io {

void write_mesh_properties(const std::filesystem::path& file, const MeshProperties& mesh);
MeshProperties read_mesh_properties(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const MeshImportResult::Info& info);

}   // namespace poisson::io
