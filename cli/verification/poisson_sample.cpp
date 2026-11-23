// poisson_sample: values of a nodal field at chosen points, optionally with the exact
// solution of a verification case at the same points.
//
//   poisson_sample --mesh <mesh.h5> --field <nodal_field.h5> --config <sample.json>
//                  --out <samples.h5> [--info-out <info.json>]
//
// sample.json: {"line": {"from": [x, y, z], "to": [x, y, z], "n": N}} or
// {"points": [[x, y, z], ...]}, plus an optional "case": {"name": ..., "params": {...}}.

#include "case_config.h"
#include "verification_cli.h"

#include "io/amr_io.h"
#include "io/mesh_io.h"
#include "io/verification_io.h"
#include "verification/field_sampler.h"

#include <cstdlib>

namespace {

namespace fs = std::filesystem;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_sample --mesh <mesh.h5> --field <nodal_field.h5> --config <sample.json>\n"
        << "                 --out <samples.h5> [--info-out <info.json>]\n";
}

}   // namespace

int main(int argc, char** argv) {
    using namespace poisson::cli;
    try {
        const auto flags = parse_flags(argc, argv, {"--mesh", "--field", "--config", "--out", "--info-out"});
        if (flags.contains("--help")) {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        for (const char* flag : {"--mesh", "--field", "--config", "--out"})
            if (!flags.contains(flag)) {
                std::cerr << "error: " << flag << " is required\n";
                print_usage(std::cerr);
                return EXIT_FAILURE;
            }
        std::vector<std::pair<std::string, fs::path>> outputs{{"--out", flags.at("--out")}};
        if (flags.contains("--info-out")) outputs.emplace_back("--info-out", flags.at("--info-out"));
        reject_aliasing({flags.at("--mesh"), flags.at("--field"), flags.at("--config")}, outputs);

        const JsonValue root = parse_json_file(flags.at("--config"));
        const Object& cfg = root.as_object("config");
        reject_unknown(cfg, {"line", "points", "case"}, "config");
        if ((find(cfg, "line") != nullptr) == (find(cfg, "points") != nullptr))
            throw std::runtime_error("config: give exactly one of 'line' and 'points'");
        std::vector<std::array<double, 3>> points;
        if (const JsonValue* line = find(cfg, "line")) {
            const Object& l = line->as_object("config.line");
            reject_unknown(l, {"from", "to", "n"}, "config.line");
            const int n = require(l, "n", "config.line").as_int("config.line.n");
            if (n < 2) throw std::runtime_error("config.line.n: at least 2 points are needed");
            points = FieldSampler::line(vec3(require(l, "from", "config.line"), "config.line.from"),
                                        vec3(require(l, "to", "config.line"), "config.line.to"), static_cast<uint32_t>(n));
        } else {
            const JsonValue& list = *find(cfg, "points");
            if (!list.is_array()) throw std::runtime_error("config.points: expected a list of [x, y, z]");
            const auto& a = std::get<JsonValue::Array>(list.data);
            for (std::size_t i = 0; i < a.size(); ++i) points.push_back(vec3(a[i], "config.points[" + std::to_string(i) + "]"));
        }
        std::unique_ptr<VerificationCase> exact;
        if (const JsonValue* c = find(cfg, "case")) exact = make_case(*c);

        const MeshProperties mesh = poisson::io::read_mesh_properties(flags.at("--mesh"));
        const poisson::io::NodalField field = poisson::io::read_nodal_field(flags.at("--field"));
        const SamplingResult result = FieldSampler().sample(mesh, field.u, points, exact.get());

        StagedOutputs staged;
        staged.add(flags.at("--out"), "out", [&](const fs::path& p) { poisson::io::write_point_samples(p, result.samples); });
        if (flags.contains("--info-out"))
            staged.add(flags.at("--info-out"), "info", [&](const fs::path& p) { poisson::io::write_info_json(p, result.info); });
        staged.publish();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
