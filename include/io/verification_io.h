#pragma once

#include "verification/case_inputs.h"
#include "verification/error_evaluator.h"
#include "verification/field_sampler.h"

#include <filesystem>
#include <vector>

// Deterministic HDF5 serialization and JSON Info sidecar I/O for the verification stages.
// The exact interpolant is written as a NodalField (io/amr_io.h).
namespace poisson::io {

// PointSamples: /points (float64 N x 3), /found (uint8), /element (uint32, UINT32_MAX outside),
// /material (uint16, UINT16_MAX outside), /u, /grad (N x 3), and /u_exact, /grad_exact
// (empty without a case). NaN marks values outside the mesh.
void write_point_samples(const std::filesystem::path& file, const PointSamples& samples);
PointSamples read_point_samples(const std::filesystem::path& file);

// The fields of an ErrorEvaluationResult: /element_l2_squared and /element_h1_squared (per
// tet), /node_error (per node, NaN where not measured).
struct ErrorFields {
    std::vector<double> element_l2_squared;
    std::vector<double> element_h1_squared;
    std::vector<double> node_error;
};
void write_error_fields(const std::filesystem::path& file, const ErrorEvaluationResult& result);
ErrorFields read_error_fields(const std::filesystem::path& file);

void write_info_json(const std::filesystem::path& file, const CaseInputsResult::Info& info);
void write_info_json(const std::filesystem::path& file, const ExactInterpolationResult::Info& info);
void write_info_json(const std::filesystem::path& file, const SamplingResult::Info& info);
void write_info_json(const std::filesystem::path& file, const ErrorEvaluationResult::Info& info);

}   // namespace poisson::io
