#include "io/verification_io.h"

#include "h5_common.h"

#include <sstream>

namespace poisson::io {

using namespace detail;

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

}   // namespace poisson::io
