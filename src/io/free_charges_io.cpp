#include "h5_common.h"
#include "serialization.h"

#include <sstream>

namespace poisson::io {

using namespace detail;

void write_free_charges(const fs::path& file, const FreeCharges& charges) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "FreeCharges");

    {
        const H5Handle g = create_group(file, h5.id, "electrons");
        const std::size_t n = charges.electrons.size();
        std::vector<std::array<double, 3>> position;
        std::vector<std::array<double, 3>> direction;
        std::vector<double> energy;
        std::vector<std::array<int32_t, 2>> pixel;
        position.reserve(n);
        direction.reserve(n);
        energy.reserve(n);
        pixel.reserve(n);
        for (const auto& e : charges.electrons) {
            position.push_back(e.position);
            direction.push_back(e.direction);
            energy.push_back(e.energy);
            pixel.push_back(e.pixel);
        }
        write_dataset_2d(file, g.id, "position", position);
        write_dataset_2d(file, g.id, "direction", direction);
        write_dataset_1d(file, g.id, "energy", energy);
        write_dataset_2d(file, g.id, "pixel", pixel);
    }

    h5.close_file_checked(file);
}

FreeCharges read_free_charges(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "FreeCharges");

    const H5Handle g = open_group(file, h5.id, "electrons");
    const auto position = read_dataset_2d<double, 3>(file, g.id, "position");
    const auto direction = read_dataset_2d<double, 3>(file, g.id, "direction");
    const auto energy = read_dataset_1d<double>(file, g.id, "energy");
    const auto pixel = read_dataset_2d<int32_t, 2>(file, g.id, "pixel");
    const std::size_t n = position.size();
    if (direction.size() != n || energy.size() != n || pixel.size() != n)
        fail(file, "electrons dataset size mismatch");

    FreeCharges result;
    result.electrons.resize(n);
    for (std::size_t i = 0; i < n; ++i) result.electrons[i] = {position[i], direction[i], energy[i], pixel[i]};
    return result;
}

void write_info_json(const fs::path& file, const FreeChargesImportResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"FreeChargesImportResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
