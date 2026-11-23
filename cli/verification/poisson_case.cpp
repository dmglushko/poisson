// poisson_case: the input files of a verification case.
//
//   poisson_case --config <case.json> --geo-out <case.geo> --materials-out <case.yaml>
//                --charges-out <case.cha> [--info-out <info.json>]
//
// case.json is {"case": {"name": ..., "params": {...}}}.

#include "case_config.h"
#include "verification_cli.h"

#include "io/verification_io.h"
#include "verification/case_inputs.h"

#include <cstdlib>
#include <fstream>

namespace {

namespace fs = std::filesystem;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_case --config <case.json> --geo-out <case.geo> --materials-out <case.yaml>\n"
        << "               --charges-out <case.cha> [--info-out <info.json>]\n";
}

void write_bytes(const fs::path& file, const std::string& bytes) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(file.string() + ": cannot open file for writing");
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    out.close();
    if (!out) throw std::runtime_error(file.string() + ": failed to write or close file");
}

}   // namespace

int main(int argc, char** argv) {
    using namespace poisson::cli;
    try {
        const auto flags = parse_flags(argc, argv, {"--config", "--geo-out", "--materials-out", "--charges-out", "--info-out"});
        if (flags.contains("--help")) {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        for (const char* flag : {"--config", "--geo-out", "--materials-out", "--charges-out"})
            if (!flags.contains(flag)) {
                std::cerr << "error: " << flag << " is required\n";
                print_usage(std::cerr);
                return EXIT_FAILURE;
            }
        std::vector<std::pair<std::string, fs::path>> outputs;
        for (const char* flag : {"--geo-out", "--materials-out", "--charges-out", "--info-out"})
            if (flags.contains(flag)) outputs.emplace_back(flag, flags.at(flag));
        reject_aliasing({flags.at("--config")}, outputs);

        const JsonValue root = parse_json_file(flags.at("--config"));
        const Object& cfg = root.as_object("config");
        reject_unknown(cfg, {"case"}, "config");
        const auto c = make_case(require(cfg, "case", "config"));
        const CaseInputsResult result = make_case_inputs(*c);

        StagedOutputs staged;
        staged.add(flags.at("--geo-out"), "geo", [&](const fs::path& p) { write_bytes(p, result.geometry); });
        staged.add(flags.at("--materials-out"), "materials", [&](const fs::path& p) { write_bytes(p, result.materials); });
        staged.add(flags.at("--charges-out"), "charges", [&](const fs::path& p) { write_bytes(p, result.charges); });
        if (flags.contains("--info-out"))
            staged.add(flags.at("--info-out"), "info", [&](const fs::path& p) { poisson::io::write_info_json(p, result.info); });
        staged.publish();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
