// poisson_diff: the error of a nodal field against the exact solution of a verification
// case, on the field's own mesh.
//
//   poisson_diff --mesh <mesh.h5> --field <nodal_field.h5> --config <diff.json>
//                --out <errors.h5> [--info-out <info.json>]
//
// diff.json: {"case": {"name": ..., "params": {...}}, "exclusion_radius": r (nm, optional;
// default the case's), "quadrature_degree": 4, "check_mesh": true}. With check_mesh, a mesh
// that does not match the case (materials, boundary conditions, charges, screening) is an
// error. The norms are in the info; errors.h5 holds the per-tet and per-node errors.

#include "case_config.h"
#include "verification_cli.h"

#include "io/amr_io.h"
#include "io/mesh_io.h"
#include "io/verification_io.h"
#include "verification/error_evaluator.h"

#include <cstdlib>

namespace {

namespace fs = std::filesystem;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_diff --mesh <mesh.h5> --field <nodal_field.h5> --config <diff.json>\n"
        << "               --out <errors.h5> [--info-out <info.json>]\n";
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
        reject_unknown(cfg, {"case", "exclusion_radius", "quadrature_degree", "check_mesh"}, "config");
        const auto c = make_case(require(cfg, "case", "config"));
        ErrorEvaluator::Settings settings;
        settings.exclusion_radius = number_or(cfg, "exclusion_radius", settings.exclusion_radius, "config");
        if (const JsonValue* v = find(cfg, "exclusion_radius"); v && v->as_double("config.exclusion_radius") < 0)
            throw std::runtime_error("config.exclusion_radius must not be negative");
        if (const JsonValue* v = find(cfg, "quadrature_degree")) settings.quadrature_degree = v->as_int("config.quadrature_degree");
        const bool check_mesh = bool_or(cfg, "check_mesh", true, "config");

        const MeshProperties mesh = poisson::io::read_mesh_properties(flags.at("--mesh"));
        if (check_mesh) {
            const auto mismatches = mesh_mismatches(mesh, *c);
            if (!mismatches.empty()) {
                std::string message = "the mesh does not match case '" + std::string(c->name()) + "':";
                for (const auto& m : mismatches) message += "\n  " + m;
                throw std::runtime_error(message);
            }
        }
        const poisson::io::NodalField field = poisson::io::read_nodal_field(flags.at("--field"));
        const ErrorEvaluationResult result = ErrorEvaluator(settings).evaluate(mesh, field.u, *c);

        StagedOutputs staged;
        staged.add(flags.at("--out"), "out", [&](const fs::path& p) { poisson::io::write_error_fields(p, result); });
        if (flags.contains("--info-out"))
            staged.add(flags.at("--info-out"), "info", [&](const fs::path& p) { poisson::io::write_info_json(p, result.info); });
        staged.publish();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
