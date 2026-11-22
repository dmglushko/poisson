#pragma once

#include "solver_result.h"

#include <filesystem>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for SolverResult.
namespace poisson::io {

void write_solver_result(const std::filesystem::path& file, const SolverResult& result);
SolverResult read_solver_result(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const SolverResult::Info& info);

}   // namespace poisson::io
