#include "amr/error_based_adapter.h"
#include "amr/kelly_estimator.h"
#include "amr/longest_edge_bisection.h"
#include "amr/marking.h"
#include "amr/recovery_estimator.h"
#include "amr/red_green_refinement.h"
#include "amr/residual_estimator.h"
#include "amr/trajectory_estimator.h"
#include "cli/common/cli_utils.h"
#include "cli/common/simple_json.h"
#include "io/amr_io.h"
#include "io/mesh_io.h"

#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

namespace fs = std::filesystem;
using poisson::cli::JsonValue;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_adapt --mesh <mesh.h5> --solution <nodal_field.h5> --config <config.json>\n"
        << "                [--state <state.h5>] --out <mesh_out.h5>\n"
        << "                [--parents-out <parents.h5>] [--state-out <state.h5>]\n"
        << "                [--eta-out <eta.h5>] [--marked-out <marked.h5>] [--info-out <info.json>]\n"
        << "\n"
        << "Runs one ErrorBasedAdapter cycle. --state resumes from the --state-out of the previous\n"
        << "cycle (red-green history and the base error of relative_tolerance); omit it for an\n"
        << "imported mesh. When the status is not 'adapted', --out is the input mesh and\n"
        << "--parents-out is the identity map, so the outputs always describe the mesh to continue with.\n";
}

// Parents of a mesh that was not changed: every tet and node is its own parent.
Parents identity_parents(const MeshProperties& mesh) {
    Parents parents;
    parents.elements.resize(mesh.elements.size());
    for (std::size_t j = 0; j < parents.elements.size(); ++j) parents.elements[j] = static_cast<uint32_t>(j);
    parents.nodes.resize(mesh.nodes.size());
    for (std::size_t i = 0; i < parents.nodes.size(); ++i)
        parents.nodes[i] = {static_cast<uint32_t>(i), static_cast<uint32_t>(i)};
    return parents;
}

// One output file, written to a sibling staging path and renamed into place only after
// every output has been written successfully.
struct StagedOutput {
    fs::path                       target;
    fs::path                       staged;
    poisson::cli::TempFileCleanup  guard{{}};
};
void reject_unknown_keys(const JsonValue::Object& obj, const std::set<std::string>& allowed,
                         const std::string& section) {
    for (const auto& [k, _] : obj) {
        if (!allowed.contains(k)) {
            throw std::runtime_error(section + ": unknown config key '" + k + "'");
        }
    }
}

PointSourceRule parse_point_source_rule(const JsonValue& v, const std::string& ctx) {
    const std::string& s = v.as_string(ctx);
    if (s == "exclude_patch") return PointSourceRule::exclude_patch;
    if (s == "ignore") return PointSourceRule::ignore;
    throw std::runtime_error(ctx + ": expected 'exclude_patch' or 'ignore', got '" + s + "'");
}

struct ParsedConfig {
    std::shared_ptr<const ErrorEstimator>  estimator;
    std::shared_ptr<const MarkingStrategy> marking;
    std::shared_ptr<const Refiner>         refiner;
    ErrorBasedAdapter::Settings            adapter_settings;
    std::optional<double>                  constant_source;
};

ParsedConfig parse_config(const JsonValue& root) {
    const auto& obj = root.as_object("config");
    reject_unknown_keys(obj, {"estimator", "marker", "refiner", "adapter", "constant_source"}, "config");

    ParsedConfig cfg;
    if (const auto it = obj.find("constant_source"); it != obj.end() && !it->second.is_null()) {
        cfg.constant_source = it->second.as_double("config.constant_source");
    }

    // ---- estimator
    {
        const auto it = obj.find("estimator");
        if (it == obj.end()) throw std::runtime_error("config: missing 'estimator'");
        const auto& eo = it->second.as_object("config.estimator");
        const auto tit = eo.find("type");
        if (tit == eo.end()) throw std::runtime_error("config.estimator: missing 'type'");
        const std::string& type = tit->second.as_string("config.estimator.type");

        if (type == "kelly") {
            reject_unknown_keys(eo, {"type", "point_sources"}, "config.estimator");
            KellyEstimator::Settings s;
            if (auto p = eo.find("point_sources"); p != eo.end())
                s.point_sources = parse_point_source_rule(p->second, "point_sources");
            cfg.estimator = std::make_shared<KellyEstimator>(s);
        } else if (type == "recovery") {
            reject_unknown_keys(eo, {"type", "quantity", "per_material", "point_sources"}, "config.estimator");
            RecoveryEstimator::Settings s;
            if (auto p = eo.find("quantity"); p != eo.end()) {
                const std::string& q = p->second.as_string("quantity");
                if (q == "flux") s.quantity = RecoveryEstimator::Quantity::flux;
                else if (q == "gradient") s.quantity = RecoveryEstimator::Quantity::gradient;
                else throw std::runtime_error("config.estimator.quantity: expected 'flux' or 'gradient'");
            }
            if (auto p = eo.find("per_material"); p != eo.end()) s.per_material = p->second.as_bool("per_material");
            if (auto p = eo.find("point_sources"); p != eo.end())
                s.point_sources = parse_point_source_rule(p->second, "point_sources");
            cfg.estimator = std::make_shared<RecoveryEstimator>(s);
        } else if (type == "residual") {
            reject_unknown_keys(eo, {"type", "tet_degree", "point_sources", "constant_source"}, "config.estimator");
            ResidualEstimator::Settings s;
            if (auto p = eo.find("tet_degree"); p != eo.end()) s.tet_degree = p->second.as_int("tet_degree");
            if (auto p = eo.find("point_sources"); p != eo.end())
                s.point_sources = parse_point_source_rule(p->second, "point_sources");
            if (auto p = eo.find("constant_source"); p != eo.end() && !p->second.is_null())
                cfg.constant_source = p->second.as_double("constant_source");
            cfg.estimator = std::make_shared<ResidualEstimator>(s);
        } else if (type == "trajectory") {
            reject_unknown_keys(eo, {"type", "scale", "tet_degree", "point_sources", "constant_source"},
                                "config.estimator");
            TrajectoryEstimator::Settings s;
            if (auto p = eo.find("scale"); p != eo.end()) s.scale = p->second.as_double("scale");
            if (auto p = eo.find("tet_degree"); p != eo.end()) s.tet_degree = p->second.as_int("tet_degree");
            if (auto p = eo.find("point_sources"); p != eo.end())
                s.point_sources = parse_point_source_rule(p->second, "point_sources");
            if (auto p = eo.find("constant_source"); p != eo.end() && !p->second.is_null())
                cfg.constant_source = p->second.as_double("constant_source");
            cfg.estimator = std::make_shared<TrajectoryEstimator>(s);
        } else {
            throw std::runtime_error("config.estimator.type: unknown estimator '" + type + "'");
        }
    }

    // ---- marker
    {
        const auto it = obj.find("marker");
        if (it == obj.end()) throw std::runtime_error("config: missing 'marker'");
        const auto& mo = it->second.as_object("config.marker");
        const auto tit = mo.find("type");
        if (tit == mo.end()) throw std::runtime_error("config.marker: missing 'type'");
        const std::string& type = tit->second.as_string("config.marker.type");

        if (type == "dorfler") {
            reject_unknown_keys(mo, {"type", "theta"}, "config.marker");
            DorflerMarking::Settings s;
            if (auto p = mo.find("theta"); p != mo.end()) s.theta = p->second.as_double("theta");
            cfg.marking = std::make_shared<DorflerMarking>(s);
        } else if (type == "maximum") {
            reject_unknown_keys(mo, {"type", "theta"}, "config.marker");
            MaximumMarking::Settings s;
            if (auto p = mo.find("theta"); p != mo.end()) s.theta = p->second.as_double("theta");
            cfg.marking = std::make_shared<MaximumMarking>(s);
        } else if (type == "fixed_fraction") {
            reject_unknown_keys(mo, {"type", "fraction"}, "config.marker");
            FixedFractionMarking::Settings s;
            if (auto p = mo.find("fraction"); p != mo.end()) s.fraction = p->second.as_double("fraction");
            cfg.marking = std::make_shared<FixedFractionMarking>(s);
        } else if (type == "equidistribution") {
            reject_unknown_keys(mo, {"type", "tolerance"}, "config.marker");
            EquidistributionMarking::Settings s;
            const auto p = mo.find("tolerance");
            if (p == mo.end()) throw std::runtime_error("config.marker: equidistribution requires 'tolerance'");
            s.tolerance = p->second.as_double("tolerance");
            cfg.marking = std::make_shared<EquidistributionMarking>(s);
        } else {
            throw std::runtime_error("config.marker.type: unknown marker '" + type + "'");
        }
    }

    // ---- refiner
    {
        const auto it = obj.find("refiner");
        if (it == obj.end()) throw std::runtime_error("config: missing 'refiner'");
        const auto& ro = it->second.as_object("config.refiner");
        const auto tit = ro.find("type");
        if (tit == ro.end()) throw std::runtime_error("config.refiner: missing 'type'");
        const std::string& type = tit->second.as_string("config.refiner.type");

        if (type == "red_green") {
            reject_unknown_keys(ro, {"type", "max_new_nodes"}, "config.refiner");
            RedGreenRefinement::Settings s;
            if (auto p = ro.find("max_new_nodes"); p != ro.end()) s.max_new_nodes = p->second.as_u64("max_new_nodes");
            cfg.refiner = std::make_shared<RedGreenRefinement>(s);
        } else if (type == "longest_edge_bisection") {
            reject_unknown_keys(ro, {"type", "max_new_nodes"}, "config.refiner");
            LongestEdgeBisection::Settings s;
            if (auto p = ro.find("max_new_nodes"); p != ro.end()) s.max_new_nodes = p->second.as_u64("max_new_nodes");
            cfg.refiner = std::make_shared<LongestEdgeBisection>(s);
        } else {
            throw std::runtime_error("config.refiner.type: unknown refiner '" + type + "'");
        }
    }

    // ---- adapter settings (optional)
    if (const auto it = obj.find("adapter"); it != obj.end() && !it->second.is_null()) {
        const auto& ao = it->second.as_object("config.adapter");
        reject_unknown_keys(ao, {"absolute_tolerance", "relative_tolerance", "max_elements", "estimate_at_limit"},
                            "config.adapter");
        if (auto p = ao.find("absolute_tolerance"); p != ao.end() && !p->second.is_null())
            cfg.adapter_settings.absolute_tolerance = p->second.as_double("absolute_tolerance");
        if (auto p = ao.find("relative_tolerance"); p != ao.end() && !p->second.is_null())
            cfg.adapter_settings.relative_tolerance = p->second.as_double("relative_tolerance");
        if (auto p = ao.find("max_elements"); p != ao.end() && !p->second.is_null())
            cfg.adapter_settings.max_elements = p->second.as_u64("max_elements");
        if (auto p = ao.find("estimate_at_limit"); p != ao.end())
            cfg.adapter_settings.estimate_at_limit = p->second.as_bool("estimate_at_limit");
    }

    return cfg;
}

}   // namespace

int main(int argc, char** argv) {
    std::optional<fs::path> mesh_file;
    std::optional<fs::path> solution_file;
    std::optional<fs::path> config_file;
    std::optional<fs::path> state_in_file;
    std::optional<fs::path> out_file;
    std::optional<fs::path> parents_out_file;
    std::optional<fs::path> state_out_file;
    std::optional<fs::path> eta_out_file;
    std::optional<fs::path> marked_out_file;
    std::optional<fs::path> info_out_file;

    auto require_value = [&](int& i, std::string_view flag) -> fs::path {
        if (i + 1 >= argc) {
            std::cerr << "error: missing value for " << flag << "\n";
            std::exit(EXIT_FAILURE);
        }
        return fs::path(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        } else if (arg == "--mesh") {
            mesh_file = require_value(i, arg);
        } else if (arg == "--solution") {
            solution_file = require_value(i, arg);
        } else if (arg == "--config") {
            config_file = require_value(i, arg);
        } else if (arg == "--state") {
            state_in_file = require_value(i, arg);
        } else if (arg == "--out") {
            out_file = require_value(i, arg);
        } else if (arg == "--parents-out") {
            parents_out_file = require_value(i, arg);
        } else if (arg == "--state-out") {
            state_out_file = require_value(i, arg);
        } else if (arg == "--eta-out") {
            eta_out_file = require_value(i, arg);
        } else if (arg == "--marked-out") {
            marked_out_file = require_value(i, arg);
        } else if (arg == "--info-out") {
            info_out_file = require_value(i, arg);
        } else {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    if (!mesh_file || !solution_file || !config_file || !out_file) {
        std::cerr << "error: --mesh, --solution, --config, and --out are required\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    std::vector<fs::path> inputs = {*mesh_file, *solution_file, *config_file};
    if (state_in_file) inputs.push_back(*state_in_file);
    std::vector<std::pair<std::string, fs::path>> outputs = {{"--out", *out_file}};
    if (parents_out_file) outputs.emplace_back("--parents-out", *parents_out_file);
    if (state_out_file) outputs.emplace_back("--state-out", *state_out_file);
    if (eta_out_file) outputs.emplace_back("--eta-out", *eta_out_file);
    if (marked_out_file) outputs.emplace_back("--marked-out", *marked_out_file);
    if (info_out_file) outputs.emplace_back("--info-out", *info_out_file);

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const auto& inp : inputs) {
            if (poisson::cli::paths_alias(outputs[i].second, inp)) {
                std::cerr << "error: " << outputs[i].first << " aliases input file '" << inp.string() << "'\n";
                return EXIT_FAILURE;
            }
        }
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (poisson::cli::paths_alias(outputs[i].second, outputs[j].second)) {
                std::cerr << "error: " << outputs[i].first << " aliases " << outputs[j].first << "\n";
                return EXIT_FAILURE;
            }
        }
    }

    try {
        const ParsedConfig cfg = parse_config(poisson::cli::parse_json_file(*config_file));
        const MeshProperties mesh = poisson::io::read_mesh_properties(*mesh_file);
        const poisson::io::NodalField field = poisson::io::read_nodal_field(*solution_file);
        ErrorBasedAdapter::State state_in;
        if (state_in_file) state_in = poisson::io::read_adaptation_state(*state_in_file);

        RefinementInput ref_input;
        ref_input.u = field.u;
        ref_input.point_sources = field.point_sources;
        if (cfg.constant_source) {
            const double f_val = *cfg.constant_source;
            ref_input.source = [f_val](const std::array<double, 3>&, uint32_t) { return f_val; };
        }

        // One cycle of the standard driver, resumed from the previous cycle's state.
        const ErrorBasedAdapter adapter(cfg.estimator, cfg.marking, cfg.refiner, cfg.adapter_settings);
        auto session = adapter.start(std::move(state_in));
        AdaptationResult result = session->adapt(mesh, ref_input);
        const bool adapted = result.status == AdaptationStatus::adapted;
        const MeshProperties& out_mesh = adapted ? *result.mesh : mesh;

        std::deque<StagedOutput> staged;   // deque: emplace_back never moves (and so never deletes) earlier entries
        auto stage_output = [&](const fs::path& target, const char* tag, const auto& write) {
            StagedOutput& o = staged.emplace_back();
            o.target = target;
            o.staged = poisson::cli::make_staging_path(target, tag);
            o.guard.path = o.staged;
            write(o.staged);
        };

        stage_output(*out_file, "out", [&](const fs::path& p) { poisson::io::write_mesh_properties(p, out_mesh); });
        if (parents_out_file) {
            stage_output(*parents_out_file, "parents", [&](const fs::path& p) {
                poisson::io::write_parents(p, result.parents ? *result.parents : identity_parents(mesh));
            });
        }
        if (state_out_file) {
            stage_output(*state_out_file, "state",
                         [&](const fs::path& p) { poisson::io::write_adaptation_state(p, session->state()); });
        }
        if (eta_out_file) {
            const auto* eb = std::get_if<ErrorBasedExtras>(&result.extras);
            const std::vector<double> empty_eta;
            const std::vector<double>& eta = eb ? eb->eta : empty_eta;
            stage_output(*eta_out_file, "eta", [&](const fs::path& p) { poisson::io::write_error_indicators(p, eta); });
        }
        if (marked_out_file) {
            const auto* eb = std::get_if<ErrorBasedExtras>(&result.extras);
            const std::vector<uint32_t> empty_marked;
            const std::vector<uint32_t>& marked = eb ? eb->marked : empty_marked;
            stage_output(*marked_out_file, "marked",
                         [&](const fs::path& p) { poisson::io::write_marked_elements(p, marked); });
        }
        if (info_out_file) {
            stage_output(*info_out_file, "info",
                         [&](const fs::path& p) { poisson::io::write_info_json(p, result.status, result.info); });
        }

        for (StagedOutput& o : staged) {
            fs::rename(o.staged, o.target);
            o.guard.release();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
