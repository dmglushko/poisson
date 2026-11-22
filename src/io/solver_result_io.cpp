#include "h5_common.h"
#include "serialization.h"

#include <sstream>

namespace poisson::io {

using namespace detail;

void write_solver_result(const fs::path& file, const SolverResult& result) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "SolverResult");
    write_dataset_1d(file, h5.id, "x", result.x);
    h5.close_file_checked(file);
}

SolverResult read_solver_result(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "SolverResult");
    SolverResult result;
    result.x = read_dataset_1d<double>(file, h5.id, "x");
    return result;
}

void write_info_json(const fs::path& file, const SolverResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"SolverResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"iterations\": " << info.iterations << ",\n"
        << "  \"residual\": " << format_double(info.residual) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
