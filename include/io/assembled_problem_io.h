#pragma once

#include "assembled_problem.h"

#include <filesystem>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for AssembledProblem.
namespace poisson::io {

void write_assembled_problem(const std::filesystem::path& file, const AssembledProblem& problem);
AssembledProblem read_assembled_problem(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const AssembledProblem::Info& info);

}   // namespace poisson::io
