#include "h5_common.h"
#include "io/amr_io.h"

#include <cmath>
#include <sstream>

namespace poisson::io {

using namespace detail;

void write_nodal_field(const fs::path& file, const NodalField& field) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "NodalField");
    write_dataset_1d(file, h5.id, "u", field.u);

    {
        const H5Handle g = create_group(file, h5.id, "point_sources");
        const std::size_t n = field.point_sources.size();
        std::vector<std::array<double, 3>> positions;
        std::vector<double> values;
        std::vector<uint32_t> elements;
        positions.reserve(n);
        values.reserve(n);
        elements.reserve(n);
        for (const auto& ps : field.point_sources) {
            if (ps.element && *ps.element == kNoneU32)
                fail(file, "point_source element collides with sentinel UINT32_MAX");
            positions.push_back(ps.position);
            values.push_back(ps.value);
            elements.push_back(ps.element.value_or(kNoneU32));
        }
        write_dataset_2d(file, g.id, "position", positions);
        write_dataset_1d(file, g.id, "value", values);
        write_dataset_1d(file, g.id, "element", elements);
    }

    h5.close_file_checked(file);
}

NodalField read_nodal_field(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "NodalField");

    NodalField field;
    field.u = read_dataset_1d<double>(file, h5.id, "u");
    {
        const H5Handle g = open_group(file, h5.id, "point_sources");
        const auto positions = read_dataset_2d<double, 3>(file, g.id, "position");
        const auto values = read_dataset_1d<double>(file, g.id, "value");
        const auto elements = read_dataset_1d<uint32_t>(file, g.id, "element");
        const std::size_t n = positions.size();
        if (values.size() != n || elements.size() != n) fail(file, "point_sources dataset size mismatch");
        field.point_sources.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            field.point_sources[i] = {
                positions[i],
                values[i],
                elements[i] == kNoneU32 ? std::nullopt : std::optional<uint32_t>(elements[i]),
            };
        }
    }
    return field;
}

void write_parents(const fs::path& file, const Parents& parents) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "Parents");
    write_dataset_1d(file, h5.id, "elements", parents.elements);
    write_dataset_2d(file, h5.id, "nodes", parents.nodes);
    h5.close_file_checked(file);
}

Parents read_parents(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "Parents");
    Parents parents;
    parents.elements = read_dataset_1d<uint32_t>(file, h5.id, "elements");
    parents.nodes = read_dataset_2d<uint32_t, 2>(file, h5.id, "nodes");
    return parents;
}

namespace {

// History datasets (families/parent, families/midpoint, family) under `parent`, shared by
// the standalone RefinementHistory file and the history group of AdaptationState.
void write_history_datasets(const fs::path& file, hid_t parent, const RefinementHistory& history) {
    {
        const H5Handle g = create_group(file, parent, "families");
        std::vector<std::array<uint32_t, 4>> parent_nodes;
        std::vector<std::array<uint32_t, 6>> midpoint;
        parent_nodes.reserve(history.families.size());
        midpoint.reserve(history.families.size());
        for (const auto& fam : history.families) {
            parent_nodes.push_back(fam.parent);
            std::array<uint32_t, 6> mp{};
            for (std::size_t k = 0; k < 6; ++k) {
                if (fam.midpoint[k] && *fam.midpoint[k] == kNoneU32)
                    fail(file, "family midpoint collides with sentinel UINT32_MAX");
                mp[k] = fam.midpoint[k].value_or(kNoneU32);
            }
            midpoint.push_back(mp);
        }
        write_dataset_2d(file, g.id, "parent", parent_nodes);
        write_dataset_2d(file, g.id, "midpoint", midpoint);
    }

    std::vector<uint32_t> family_raw;
    family_raw.reserve(history.family.size());
    for (const auto& f : history.family) {
        if (f && *f == kNoneU32) fail(file, "family index collides with sentinel UINT32_MAX");
        family_raw.push_back(f.value_or(kNoneU32));
    }
    write_dataset_1d(file, parent, "family", family_raw);
}

RefinementHistory read_history_datasets(const fs::path& file, hid_t parent) {
    RefinementHistory history;
    {
        const H5Handle g = open_group(file, parent, "families");
        const auto parent_nodes = read_dataset_2d<uint32_t, 4>(file, g.id, "parent");
        const auto midpoint = read_dataset_2d<uint32_t, 6>(file, g.id, "midpoint");
        if (parent_nodes.size() != midpoint.size()) fail(file, "families/parent and families/midpoint size mismatch");
        history.families.resize(parent_nodes.size());
        for (std::size_t i = 0; i < parent_nodes.size(); ++i) {
            history.families[i].parent = parent_nodes[i];
            for (std::size_t k = 0; k < 6; ++k) {
                history.families[i].midpoint[k] =
                    midpoint[i][k] == kNoneU32 ? std::nullopt : std::optional<uint32_t>(midpoint[i][k]);
            }
        }
    }

    const auto family_raw = read_dataset_1d<uint32_t>(file, parent, "family");
    history.family.resize(family_raw.size());
    for (std::size_t i = 0; i < family_raw.size(); ++i) {
        if (family_raw[i] == kNoneU32) {
            history.family[i] = std::nullopt;
        } else if (family_raw[i] >= history.families.size()) {
            fail(file, "family references out-of-range family index " + std::to_string(family_raw[i]));
        } else {
            history.family[i] = family_raw[i];
        }
    }
    return history;
}

}   // namespace

void write_refinement_history(const fs::path& file, const RefinementHistory& history) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "RefinementHistory");
    write_history_datasets(file, h5.id, history);
    h5.close_file_checked(file);
}

RefinementHistory read_refinement_history(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "RefinementHistory");
    return read_history_datasets(file, h5.id);
}

void write_adaptation_state(const fs::path& file, const ErrorBasedAdapter::State& state) {
    if (state.initial_error && !(std::isfinite(*state.initial_error) && *state.initial_error >= 0))
        fail(file, "initial_error must be finite and >= 0");
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "AdaptationState");
    {
        const H5Handle g = create_group(file, h5.id, "history");
        write_history_datasets(file, g.id, state.history);
    }
    std::vector<double> initial_error;
    if (state.initial_error) initial_error.push_back(*state.initial_error);
    write_dataset_1d(file, h5.id, "initial_error", initial_error);
    h5.close_file_checked(file);
}

ErrorBasedAdapter::State read_adaptation_state(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "AdaptationState");
    ErrorBasedAdapter::State state;
    {
        const H5Handle g = open_group(file, h5.id, "history");
        state.history = read_history_datasets(file, g.id);
    }
    const auto initial_error = read_dataset_1d<double>(file, h5.id, "initial_error");
    if (initial_error.size() > 1) fail(file, "initial_error must have 0 or 1 entries");
    if (!initial_error.empty()) {
        if (!(std::isfinite(initial_error[0]) && initial_error[0] >= 0))
            fail(file, "initial_error must be finite and >= 0");
        state.initial_error = initial_error[0];
    }
    return state;
}

void write_error_indicators(const fs::path& file, const std::vector<double>& eta) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "ErrorIndicators");
    write_dataset_1d(file, h5.id, "eta", eta);
    h5.close_file_checked(file);
}

std::vector<double> read_error_indicators(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "ErrorIndicators");
    return read_dataset_1d<double>(file, h5.id, "eta");
}

void write_info_json(const fs::path& file, AdaptationStatus status, const AdaptationResult::Info& info) {
    auto status_str = [](AdaptationStatus s) -> const char* {
        switch (s) {
        case AdaptationStatus::adapted: return "adapted";
        case AdaptationStatus::converged: return "converged";
        case AdaptationStatus::limit_reached: return "limit_reached";
        case AdaptationStatus::unchanged: return "unchanged";
        }
        return "unknown";
    };

    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"AdaptationResult::Info\",\n"
        << "  \"status\": " << escape_json(status_str(status)) << ",\n"
        << "  \"adapter\": " << escape_json(info.adapter) << ",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"elements_before\": " << info.elements_before << ",\n"
        << "  \"elements_after\": " << info.elements_after << ",\n"
        << "  \"error_estimate\": " << (info.error_estimate ? format_double(*info.error_estimate) : "null") << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << ",\n"
        << "  \"stages\": [";
    for (std::size_t i = 0; i < info.stages.size(); ++i) {
        const auto& st = info.stages[i];
        if (i > 0) oss << ",";
        oss << "\n    {\n"
            << "      \"stage\": " << escape_json(st.stage) << ",\n"
            << "      \"method\": " << escape_json(st.method) << ",\n"
            << "      \"time\": " << format_double(st.time) << ",\n"
            << "      \"metrics\": {";
        std::size_t m = 0;
        for (const auto& [k, v] : st.metrics) {
            if (m++ > 0) oss << ", ";
            oss << escape_json(k) << ": " << format_double(v);
        }
        oss << "},\n"
            << "      \"warnings\": " << format_warnings(st.warnings) << "\n"
            << "    }";
    }
    if (!info.stages.empty()) oss << "\n  ";
    oss << "]\n}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
