// poisson_case: the input files of a verification case, or its exact solution on a mesh.
//
//   poisson_case --config <case.json> --geo-out <case.geo> --materials-out <case.yaml>
//                --charges-out <case.cha> [--info-out <info.json>]
//   poisson_case --config <case.json> --mesh <mesh.h5> --out <exact.h5> [--info-out <info.json>]
//
// case.json is {"case": {"name": ..., "params": {...}}}. The second form writes the exact
// potential at every node as a NodalField, usable wherever a solution is.

#include "case_config.h"
#include "verification_cli.h"

#include "io/amr_io.h"
#include "io/mesh_io.h"
#include "io/verification_io.h"
#include "verification/case_inputs.h"

#include <cstdlib>
#include <fstream>

namespace {

namespace fs = std::filesystem;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_case --config <case.json> --geo-out <case.geo> --materials-out <case.yaml>\n"
        << "               --charges-out <case.cha> [--info-out <info.json>]\n"
        << "  poisson_case --config <case.json> --mesh <mesh.h5> --out <exact.h5> [--info-out <info.json>]\n";
}

void write_bytes(const fs::path& file, const std::string& bytes) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(file.string() + ": cannot open file for writing");
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    out.close();
    if (!out) throw std::runtime_error(file.string() + ": failed to write or close file");
}

std::unique_ptr<VerificationCase> read_case(const fs::path& config) {
    const auto root = poisson::cli::parse_json_file(config);
    const auto& obj = root.as_object("config");
    poisson::cli::reject_unknown(obj, {"case"}, "config");
    return poisson::cli::make_case(poisson::cli::require(obj, "case", "config"));
}

}   // namespace

int main(int argc, char** argv) {
    try {
        const auto flags = poisson::cli::parse_flags(
            argc, argv, {"--config", "--geo-out", "--materials-out", "--charges-out", "--mesh", "--out", "--info-out"});
        if (flags.contains("--help")) {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        const auto get = [&](const char* flag) { return flags.contains(flag) ? std::optional(flags.at(flag)) : std::nullopt; };
        const auto config = get("--config"), info_out = get("--info-out");
        const bool inputs_mode = flags.contains("--geo-out") || flags.contains("--materials-out") || flags.contains("--charges-out");
        const bool exact_mode = flags.contains("--mesh") || flags.contains("--out");
        if (!config || inputs_mode == exact_mode) {
            std::cerr << "error: give --config and either the three input outputs or --mesh and --out\n";
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }

        std::vector<fs::path> inputs{*config};
        std::vector<std::pair<std::string, fs::path>> outputs;
        if (inputs_mode) {
            for (const char* flag : {"--geo-out", "--materials-out", "--charges-out"}) {
                if (!flags.contains(flag)) throw std::runtime_error(std::string(flag) + " is required");
                outputs.emplace_back(flag, flags.at(flag));
            }
        } else {
            if (!flags.contains("--mesh") || !flags.contains("--out")) throw std::runtime_error("--mesh and --out are both required");
            inputs.push_back(flags.at("--mesh"));
            outputs.emplace_back("--out", flags.at("--out"));
        }
        if (info_out) outputs.emplace_back("--info-out", *info_out);
        poisson::cli::reject_aliasing(inputs, outputs);

        const auto c = read_case(*config);
        poisson::cli::StagedOutputs staged;
        if (inputs_mode) {
            const CaseInputsResult result = make_case_inputs(*c);
            staged.add(flags.at("--geo-out"), "geo", [&](const fs::path& p) { write_bytes(p, result.geometry); });
            staged.add(flags.at("--materials-out"), "materials", [&](const fs::path& p) { write_bytes(p, result.materials); });
            staged.add(flags.at("--charges-out"), "charges", [&](const fs::path& p) { write_bytes(p, result.charges); });
            if (info_out) staged.add(*info_out, "info", [&](const fs::path& p) { poisson::io::write_info_json(p, result.info); });
        } else {
            const MeshProperties mesh = poisson::io::read_mesh_properties(flags.at("--mesh"));
            const ExactInterpolationResult result = interpolate_exact(mesh, *c);
            staged.add(flags.at("--out"), "out", [&](const fs::path& p) { poisson::io::write_nodal_field(p, {result.u, {}}); });
            if (info_out) staged.add(*info_out, "info", [&](const fs::path& p) { poisson::io::write_info_json(p, result.info); });
        }
        staged.publish();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
