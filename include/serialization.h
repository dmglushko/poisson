#pragma once

#include "amr/mesh_adapter.h"
#include "amr/refinement_input.h"
#include "amr/refiner.h"
#include "assembled_problem.h"
#include "free_charges.h"
#include "importer.h"
#include "mesh_properties.h"
#include "solver_result.h"

#include <filesystem>
#include <vector>

// Deterministic HDF5 serialization for the pipeline data classes.
//
// Files are written without HDF5 object timestamps, padding, or run-specific timings so that
// identical data produces byte-for-byte identical .h5 files suitable for content-addressed
// caching. Run metadata (Info structs) is written separately as JSON sidecars.
namespace poisson::io {

// Owning snapshot of nodal potential u and optional point sources for CLI-based AMR stages.
struct NodalField {
    std::vector<double>                       u;
    std::vector<RefinementInput::PointSource> point_sources;
};

void write_mesh_properties(const std::filesystem::path& file, const MeshProperties& mesh);
MeshProperties read_mesh_properties(const std::filesystem::path& file);

void write_free_charges(const std::filesystem::path& file, const FreeCharges& charges);
FreeCharges read_free_charges(const std::filesystem::path& file);

void write_assembled_problem(const std::filesystem::path& file, const AssembledProblem& problem);
AssembledProblem read_assembled_problem(const std::filesystem::path& file);

void write_solver_result(const std::filesystem::path& file, const SolverResult& result);
SolverResult read_solver_result(const std::filesystem::path& file);

void write_nodal_field(const std::filesystem::path& file, const NodalField& field);
NodalField read_nodal_field(const std::filesystem::path& file);

void write_parents(const std::filesystem::path& file, const Parents& parents);
Parents read_parents(const std::filesystem::path& file);

void write_refinement_history(const std::filesystem::path& file, const RefinementHistory& history);
RefinementHistory read_refinement_history(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const MeshImportResult::Info& info);
void write_info_json(const std::filesystem::path& file, const FreeChargesImportResult::Info& info);
void write_info_json(const std::filesystem::path& file, const AssembledProblem::Info& info);
void write_info_json(const std::filesystem::path& file, const SolverResult::Info& info);
void write_info_json(const std::filesystem::path& file, AdaptationStatus status, const AdaptationResult::Info& info);

}   // namespace poisson::io
