#include "io/verification_io.h"

#include "h5_common.h"

#include <sstream>

namespace poisson::io {

using namespace detail;

void write_point_samples(const fs::path& file, const PointSamples& s) {
    const std::size_t n = s.points.size();
    if (s.found.size() != n || s.element.size() != n || s.material.size() != n || s.u.size() != n || s.grad.size() != n)
        fail(file, "point samples have inconsistent sizes");
    if ((!s.u_exact.empty() || !s.grad_exact.empty()) && (s.u_exact.size() != n || s.grad_exact.size() != n))
        fail(file, "exact point samples have inconsistent sizes");
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "PointSamples");
    write_dataset_2d(file, h5.id, "points", s.points);
    write_dataset_1d(file, h5.id, "found", s.found);
    write_dataset_1d(file, h5.id, "element", s.element);
    write_dataset_1d(file, h5.id, "material", s.material);
    write_dataset_1d(file, h5.id, "u", s.u);
    write_dataset_2d(file, h5.id, "grad", s.grad);
    write_dataset_1d(file, h5.id, "u_exact", s.u_exact);
    write_dataset_2d(file, h5.id, "grad_exact", s.grad_exact);
    h5.close_file_checked(file);
}

PointSamples read_point_samples(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "PointSamples");
    PointSamples s;
    s.points = read_dataset_2d<double, 3>(file, h5.id, "points");
    s.found = read_dataset_1d<uint8_t>(file, h5.id, "found");
    s.element = read_dataset_1d<uint32_t>(file, h5.id, "element");
    s.material = read_dataset_1d<uint16_t>(file, h5.id, "material");
    s.u = read_dataset_1d<double>(file, h5.id, "u");
    s.grad = read_dataset_2d<double, 3>(file, h5.id, "grad");
    s.u_exact = read_dataset_1d<double>(file, h5.id, "u_exact");
    s.grad_exact = read_dataset_2d<double, 3>(file, h5.id, "grad_exact");
    const std::size_t n = s.points.size();
    if (s.found.size() != n || s.element.size() != n || s.material.size() != n || s.u.size() != n || s.grad.size() != n)
        fail(file, "point sample datasets have inconsistent sizes");
    if ((!s.u_exact.empty() || !s.grad_exact.empty()) && (s.u_exact.size() != n || s.grad_exact.size() != n))
        fail(file, "exact point sample datasets have inconsistent sizes");
    for (const uint8_t f : s.found) require_bool_flag(file, f, "found");
    return s;
}

void write_error_fields(const fs::path& file, const ErrorEvaluationResult& result) {
    if (result.element_l2_squared.size() != result.element_h1_squared.size())
        fail(file, "error fields have inconsistent sizes");
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "ErrorFields");
    write_dataset_1d(file, h5.id, "element_l2_squared", result.element_l2_squared);
    write_dataset_1d(file, h5.id, "element_h1_squared", result.element_h1_squared);
    write_dataset_1d(file, h5.id, "node_error", result.node_error);
    h5.close_file_checked(file);
}

ErrorFields read_error_fields(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "ErrorFields");
    ErrorFields f;
    f.element_l2_squared = read_dataset_1d<double>(file, h5.id, "element_l2_squared");
    f.element_h1_squared = read_dataset_1d<double>(file, h5.id, "element_h1_squared");
    f.node_error = read_dataset_1d<double>(file, h5.id, "node_error");
    if (f.element_l2_squared.size() != f.element_h1_squared.size()) fail(file, "error field datasets have inconsistent sizes");
    return f;
}

void write_info_json(const fs::path& file, const CaseInputsResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"CaseInputsResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"case\": " << escape_json(info.case_name) << ",\n"
        << "  \"materials\": " << info.materials << ",\n"
        << "  \"charges\": " << info.charges << ",\n"
        << "  \"default_exclusion_radius\": " << format_double(info.default_exclusion_radius) << ",\n"
        << "  \"boundary_error_bound\": " << format_double(info.boundary_error_bound) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const ExactInterpolationResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"ExactInterpolationResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const SamplingResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"SamplingResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"outside\": " << info.outside << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const ErrorEvaluationResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"ErrorEvaluationResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"exclusion_radius\": " << format_double(info.exclusion_radius) << ",\n"
        << "  \"l2_error\": " << format_double(info.l2_error) << ",\n"
        << "  \"h1_error\": " << format_double(info.h1_error) << ",\n"
        << "  \"max_error\": " << format_double(info.max_error) << ",\n"
        << "  \"l2_exact\": " << format_double(info.l2_exact) << ",\n"
        << "  \"h1_exact\": " << format_double(info.h1_exact) << ",\n"
        << "  \"metal_potential_error\": " << format_double(info.metal_potential_error) << ",\n"
        << "  \"measured_volume\": " << format_double(info.measured_volume) << ",\n"
        << "  \"excluded_points\": " << info.excluded_points << ",\n"
        << "  \"metal_elements\": " << info.metal_elements << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
