#include "h5_common.h"
#include "io/mesh_io.h"

#include <sstream>

namespace poisson::io {

using namespace detail;

void write_mesh_properties(const fs::path& file, const MeshProperties& mesh) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "MeshProperties");

    write_dataset_2d(file, h5.id, "nodes", mesh.nodes);

    {
        const H5Handle g = create_group(file, h5.id, "elements");
        std::vector<std::array<uint32_t, 4>> nodes;
        std::vector<uint16_t> materials;
        nodes.reserve(mesh.elements.size());
        materials.reserve(mesh.elements.size());
        for (const auto& e : mesh.elements) {
            nodes.push_back(e.node);
            materials.push_back(e.material);
        }
        write_dataset_2d(file, g.id, "node", nodes);
        write_dataset_1d(file, g.id, "material", materials);
    }

    {
        const H5Handle g = create_group(file, h5.id, "boundary_elements");
        std::vector<std::array<uint32_t, 3>> nodes;
        std::vector<uint16_t> bcs;
        nodes.reserve(mesh.boundary_elements.size());
        bcs.reserve(mesh.boundary_elements.size());
        for (const auto& be : mesh.boundary_elements) {
            nodes.push_back(be.node);
            bcs.push_back(be.boundary_condition);
        }
        write_dataset_2d(file, g.id, "node", nodes);
        write_dataset_1d(file, g.id, "boundary_condition", bcs);
    }

    {
        const H5Handle g = create_group(file, h5.id, "boundary_conditions");
        const std::size_t n = mesh.boundary_conditions.size();
        std::vector<uint8_t> types;
        std::vector<double> values;
        std::vector<uint8_t> has_alpha;
        std::vector<double> alphas;
        types.reserve(n);
        values.reserve(n);
        has_alpha.reserve(n);
        alphas.reserve(n);
        for (const auto& bc : mesh.boundary_conditions) {
            types.push_back(static_cast<uint8_t>(bc.type));
            values.push_back(bc.value);
            has_alpha.push_back(bc.alpha.has_value() ? 1 : 0);
            alphas.push_back(bc.alpha.value_or(0.0));
        }
        write_dataset_1d(file, g.id, "type", types);
        write_dataset_1d(file, g.id, "value", values);
        write_dataset_1d(file, g.id, "has_alpha", has_alpha);
        write_dataset_1d(file, g.id, "alpha", alphas);
    }

    {
        const H5Handle g = create_group(file, h5.id, "materials");
        const std::size_t n = mesh.materials.size();
        std::vector<uint8_t> name_bytes;
        std::vector<uint64_t> name_offsets;
        std::vector<double> eps_r;
        std::vector<double> screening_length;
        std::vector<uint8_t> is_metal;
        std::vector<uint8_t> has_voltage;
        std::vector<double> voltage;
        name_offsets.reserve(n + 1);
        eps_r.reserve(n);
        screening_length.reserve(n);
        is_metal.reserve(n);
        has_voltage.reserve(n);
        voltage.reserve(n);
        name_offsets.push_back(0);
        for (const auto& m : mesh.materials) {
            name_bytes.insert(name_bytes.end(), m.name.begin(), m.name.end());
            name_offsets.push_back(name_bytes.size());
            eps_r.push_back(m.eps_r);
            screening_length.push_back(m.screening_length);
            is_metal.push_back(m.is_metal ? 1 : 0);
            has_voltage.push_back(m.voltage.has_value() ? 1 : 0);
            voltage.push_back(m.voltage.value_or(0.0));
        }
        write_dataset_1d(file, g.id, "name_bytes", name_bytes);
        write_dataset_1d(file, g.id, "name_offsets", name_offsets);
        write_dataset_1d(file, g.id, "eps_r", eps_r);
        write_dataset_1d(file, g.id, "screening_length", screening_length);
        write_dataset_1d(file, g.id, "is_metal", is_metal);
        write_dataset_1d(file, g.id, "has_voltage", has_voltage);
        write_dataset_1d(file, g.id, "voltage", voltage);
    }

    {
        const H5Handle g = create_group(file, h5.id, "charges");
        std::vector<std::array<double, 3>> positions;
        std::vector<int32_t> values;
        positions.reserve(mesh.charges.size());
        values.reserve(mesh.charges.size());
        for (const auto& c : mesh.charges) {
            positions.push_back(c.position);
            values.push_back(c.value);
        }
        write_dataset_2d(file, g.id, "position", positions);
        write_dataset_1d(file, g.id, "value", values);
    }

    {
        std::vector<std::array<uint32_t, 4>> neighbours;
        neighbours.reserve(mesh.neighbours.size());
        for (const auto& nb : mesh.neighbours) {
            std::array<uint32_t, 4> row{};
            for (std::size_t i = 0; i < 4; ++i) {
                if (nb[i] && *nb[i] == kNoneU32)
                    fail(file, "neighbour index collides with sentinel UINT32_MAX");
                row[i] = nb[i].value_or(kNoneU32);
            }
            neighbours.push_back(row);
        }
        write_dataset_2d(file, h5.id, "neighbours", neighbours);
    }

    h5.close_file_checked(file);
}

MeshProperties read_mesh_properties(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "MeshProperties");

    MeshProperties mesh;
    mesh.nodes = read_dataset_2d<double, 3>(file, h5.id, "nodes");

    {
        const H5Handle g = open_group(file, h5.id, "boundary_conditions");
        const auto types = read_dataset_1d<uint8_t>(file, g.id, "type");
        const auto values = read_dataset_1d<double>(file, g.id, "value");
        const auto has_alpha = read_dataset_1d<uint8_t>(file, g.id, "has_alpha");
        const auto alphas = read_dataset_1d<double>(file, g.id, "alpha");
        const std::size_t n = types.size();
        if (values.size() != n || has_alpha.size() != n || alphas.size() != n)
            fail(file, "boundary_conditions dataset size mismatch");
        mesh.boundary_conditions.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (types[i] > static_cast<uint8_t>(MeshProperties::BcType::Robin))
                fail(file, "invalid BcType value " + std::to_string(types[i]));
            require_bool_flag(file, has_alpha[i], "boundary_conditions/has_alpha");
            mesh.boundary_conditions[i] = {
                static_cast<MeshProperties::BcType>(types[i]),
                values[i],
                has_alpha[i] ? std::optional<double>(alphas[i]) : std::nullopt,
            };
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "materials");
        const auto name_bytes = read_dataset_1d<uint8_t>(file, g.id, "name_bytes");
        const auto name_offsets = read_dataset_1d<uint64_t>(file, g.id, "name_offsets");
        const auto eps_r = read_dataset_1d<double>(file, g.id, "eps_r");
        const auto screening_length = read_dataset_1d<double>(file, g.id, "screening_length");
        const auto is_metal = read_dataset_1d<uint8_t>(file, g.id, "is_metal");
        const auto has_voltage = read_dataset_1d<uint8_t>(file, g.id, "has_voltage");
        const auto voltage = read_dataset_1d<double>(file, g.id, "voltage");
        const std::size_t n = eps_r.size();
        if (name_offsets.size() != n + 1 || screening_length.size() != n || is_metal.size() != n ||
            has_voltage.size() != n || voltage.size() != n)
            fail(file, "materials dataset size mismatch");
        if (name_offsets.front() != 0 || name_offsets.back() != name_bytes.size())
            fail(file, "materials/name_offsets must span materials/name_bytes from 0 to size");
        mesh.materials.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t begin = name_offsets[i], end = name_offsets[i + 1];
            if (begin > end || end > name_bytes.size()) fail(file, "invalid materials/name_offsets");
            require_bool_flag(file, is_metal[i], "materials/is_metal");
            require_bool_flag(file, has_voltage[i], "materials/has_voltage");
            mesh.materials[i] = {
                std::string(reinterpret_cast<const char*>(name_bytes.data() + begin),
                            static_cast<std::size_t>(end - begin)),
                eps_r[i],
                screening_length[i],
                is_metal[i] != 0,
                has_voltage[i] ? std::optional<double>(voltage[i]) : std::nullopt,
            };
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "elements");
        const auto nodes = read_dataset_2d<uint32_t, 4>(file, g.id, "node");
        const auto materials = read_dataset_1d<uint16_t>(file, g.id, "material");
        if (nodes.size() != materials.size()) fail(file, "elements/node and elements/material size mismatch");
        mesh.elements.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (const uint32_t n_idx : nodes[i]) {
                if (n_idx >= mesh.nodes.size())
                    fail(file, "elements/node references out-of-range node index " + std::to_string(n_idx));
            }
            if (materials[i] >= mesh.materials.size())
                fail(file, "elements/material references out-of-range material index " + std::to_string(materials[i]));
            mesh.elements[i] = {nodes[i], materials[i]};
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "boundary_elements");
        const auto nodes = read_dataset_2d<uint32_t, 3>(file, g.id, "node");
        const auto bcs = read_dataset_1d<uint16_t>(file, g.id, "boundary_condition");
        if (nodes.size() != bcs.size())
            fail(file, "boundary_elements/node and boundary_elements/boundary_condition size mismatch");
        mesh.boundary_elements.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (const uint32_t n_idx : nodes[i]) {
                if (n_idx >= mesh.nodes.size())
                    fail(file, "boundary_elements/node references out-of-range node index " + std::to_string(n_idx));
            }
            if (bcs[i] >= mesh.boundary_conditions.size())
                fail(file, "boundary_elements/boundary_condition references out-of-range index " +
                               std::to_string(bcs[i]));
            mesh.boundary_elements[i] = {nodes[i], bcs[i]};
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "charges");
        const auto positions = read_dataset_2d<double, 3>(file, g.id, "position");
        const auto values = read_dataset_1d<int32_t>(file, g.id, "value");
        if (positions.size() != values.size()) fail(file, "charges dataset size mismatch");
        mesh.charges.resize(positions.size());
        for (std::size_t i = 0; i < positions.size(); ++i) mesh.charges[i] = {positions[i], values[i]};
    }

    {
        const auto raw = read_dataset_2d<uint32_t, 4>(file, h5.id, "neighbours");
        if (!raw.empty() && raw.size() != mesh.elements.size())
            fail(file, "neighbours length must be 0 or match elements count");
        mesh.neighbours.resize(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            for (std::size_t k = 0; k < 4; ++k) {
                if (raw[i][k] == kNoneU32) {
                    mesh.neighbours[i][k] = std::nullopt;
                } else if (raw[i][k] >= mesh.elements.size()) {
                    fail(file, "neighbours references out-of-range element index " + std::to_string(raw[i][k]));
                } else {
                    mesh.neighbours[i][k] = raw[i][k];
                }
            }
        }
    }

    return mesh;
}

void write_info_json(const fs::path& file, const MeshImportResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"MeshImportResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"time_read\": " << format_double(info.time_read) << ",\n"
        << "  \"time_geometry\": " << format_double(info.time_geometry) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
