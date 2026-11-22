#pragma once

#include "free_charges.h"
#include "importer/importer.h"

#include <filesystem>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for FreeCharges.
namespace poisson::io {

void write_free_charges(const std::filesystem::path& file, const FreeCharges& charges);
FreeCharges read_free_charges(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const FreeChargesImportResult::Info& info);

}   // namespace poisson::io
