#pragma once

#include "amr/error_based_adapter.h"
#include "amr/mesh_adapter.h"
#include "amr/refinement_input.h"
#include "amr/refiner.h"

#include <filesystem>
#include <vector>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for AMR stage artifacts.
namespace poisson::io {

// Owning snapshot of nodal potential u and optional point sources for CLI-based AMR stages.
struct NodalField {
    std::vector<double>                       u;
    std::vector<RefinementInput::PointSource> point_sources;
};

void write_nodal_field(const std::filesystem::path& file, const NodalField& field);
NodalField read_nodal_field(const std::filesystem::path& file);

void write_parents(const std::filesystem::path& file, const Parents& parents);
Parents read_parents(const std::filesystem::path& file);

void write_refinement_history(const std::filesystem::path& file, const RefinementHistory& history);
RefinementHistory read_refinement_history(const std::filesystem::path& file);

// ErrorBasedAdapter::State: group /history (same layout as a RefinementHistory file) and
// /initial_error (float64, 0 or 1 entries).
void write_adaptation_state(const std::filesystem::path& file, const ErrorBasedAdapter::State& state);
ErrorBasedAdapter::State read_adaptation_state(const std::filesystem::path& file);

// ErrorBasedExtras::eta: /eta (float64, one entry per tet of the adapted input mesh; may be empty).
void write_error_indicators(const std::filesystem::path& file, const std::vector<double>& eta);
std::vector<double> read_error_indicators(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, AdaptationStatus status, const AdaptationResult::Info& info);

}   // namespace poisson::io
