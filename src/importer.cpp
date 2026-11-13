#include "importer.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void fail(const fs::path& file, const std::string& message) {
    throw std::runtime_error(file.string() + ": " + message);
}

[[noreturn]] void fail(const fs::path& file, std::size_t line, const std::string& message) {
    throw std::runtime_error(file.string() + ":" + std::to_string(line) + ": " + message);
}

std::string read_file(const fs::path& file) {
    if (!fs::is_regular_file(file)) fail(file, "file not found");
    std::ifstream in(file, std::ios::binary);
    if (!in) fail(file, "cannot open file");
    std::string data(static_cast<std::size_t>(fs::file_size(file)), '\0');
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (static_cast<std::size_t>(in.gcount()) != data.size()) fail(file, "cannot read file");
    return data;
}

// ---------------------------------------------------------------- text helpers

bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

std::string_view trim(std::string_view s) {
    while (!s.empty() && is_blank(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_blank(s.back())) s.remove_suffix(1);
    return s;
}

// Reads one whitespace-separated number from the front of s. Returns false if the next
// token is not a complete number of type T.
template <class T>
bool take(std::string_view& s, T& value) {
    while (!s.empty() && is_blank(s.front())) s.remove_prefix(1);
    if (s.empty()) return false;
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{}) return false;
    s.remove_prefix(static_cast<std::size_t>(end - s.data()));
    return s.empty() || is_blank(s.front());
}

// Parses s as one number with nothing after it.
bool parse_number(std::string_view s, double& value) {
    s = trim(s);
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    return take(s, value) && trim(s).empty() && std::isfinite(value);
}

// Walks through a text file line by line and keeps the line number for error messages.
class TextReader {
public:
    TextReader(const fs::path& file, std::string_view text) : file_(file), text_(text) {}

    bool next(std::string_view& line) {
        if (pos_ >= text_.size()) return false;
        const std::size_t end = std::min(text_.find('\n', pos_), text_.size());
        line = trim(text_.substr(pos_, end - pos_));
        pos_ = end + 1;
        ++line_;
        return true;
    }

    std::string_view require(const std::string& what) {
        std::string_view line;
        if (!next(line)) fail("unexpected end of file, expected " + what);
        return line;
    }

    void expect(std::string_view marker) {
        const std::string_view line = require(std::string(marker));
        if (line != marker)
            fail("expected " + std::string(marker) + ", found '" + std::string(line) +
                 "' (does the count at the start of the section match its contents?)");
    }

    [[noreturn]] void fail(const std::string& message) const { ::fail(file_, line_, message); }

    std::size_t line() const { return line_; }

private:
    const fs::path& file_;
    std::string_view text_;
    std::size_t pos_ = 0;
    std::size_t line_ = 0;
};

// ---------------------------------------------------------------- Gmsh mesh

// Maps Gmsh node ids (any positive numbers, possibly with gaps) to indices 0..N-1.
class NodeIndex {
public:
    static constexpr uint32_t none = std::numeric_limits<uint32_t>::max();

    // Returns false if an id occurs twice; duplicate then holds that id.
    bool build(const std::vector<uint64_t>& ids, uint64_t& duplicate) {
        const uint64_t max_id = ids.empty() ? 0 : *std::max_element(ids.begin(), ids.end());
        use_dense_ = max_id <= 8 * ids.size() + 1024;
        if (use_dense_) dense_.assign(static_cast<std::size_t>(max_id) + 1, none);
        else sparse_.reserve(ids.size());

        for (uint32_t i = 0; i < ids.size(); ++i) {
            const bool is_new = use_dense_ ? dense_[ids[i]] == none : !sparse_.contains(ids[i]);
            if (!is_new) {
                duplicate = ids[i];
                return false;
            }
            if (use_dense_) dense_[ids[i]] = i;
            else sparse_[ids[i]] = i;
        }
        return true;
    }

    uint32_t find(uint64_t id) const {
        if (use_dense_) return id < dense_.size() ? dense_[id] : none;
        const auto it = sparse_.find(id);
        return it == sparse_.end() ? none : it->second;
    }

private:
    bool use_dense_ = true;
    std::vector<uint32_t> dense_;
    std::unordered_map<uint64_t, uint32_t> sparse_;
};

// What is needed while reading a mesh, next to the mesh itself.
struct MeshReadState {
    std::unordered_map<long long, uint16_t> volume_index;    // Gmsh physical tag -> index into materials
    std::unordered_map<long long, uint16_t> surface_index;   // Gmsh physical tag -> index into boundary_conditions
    std::vector<std::string> surface_names;                  // same order as boundary_conditions
    std::vector<uint64_t> node_ids;                          // Gmsh id of every node, same order as nodes
    NodeIndex node_index;
    std::vector<std::size_t> boundary_lines;                 // line of every boundary element
    std::vector<uint64_t> material_use;                      // tets per material
    std::vector<uint64_t> boundary_use;                      // triangles per boundary condition
};

// A surface name is "<number>V" (Dirichlet, volts) or "D<number>" (Neumann).
std::optional<MeshProperties::BoundaryCondition> parse_boundary_condition(std::string_view name) {
    name = trim(name);
    double value = 0;
    if (name.size() >= 2 && name.back() == 'V' && parse_number(name.substr(0, name.size() - 1), value))
        return MeshProperties::BoundaryCondition{MeshProperties::BcType::Dirichlet, value, std::nullopt};
    if (name.size() >= 2 && name.front() == 'D' && parse_number(name.substr(1), value))
        return MeshProperties::BoundaryCondition{MeshProperties::BcType::Neumann, value, std::nullopt};
    return std::nullopt;
}

void read_mesh_format(TextReader& reader) {
    std::string_view line = reader.require("the mesh format");
    double version = 0;
    int file_type = 0, data_size = 0;
    if (!take(line, version) || !take(line, file_type) || !take(line, data_size))
        reader.fail("invalid $MeshFormat line");
    if (version < 2.0 || version >= 3.0)
        reader.fail("unsupported Gmsh format version; only version 2.2 is supported "
                    "(in Gmsh: export with -format msh22)");
    if (file_type != 0) reader.fail("binary Gmsh files are not supported; export as ASCII");
    if (data_size != 8) reader.fail("unsupported floating-point size; expected 8");
    reader.expect("$EndMeshFormat");
}

void read_physical_names(TextReader& reader, MeshProperties& mesh, MeshReadState& state) {
    static const std::regex voltage(R"((^|\s+)(-?[\d.]+)V\b)");

    std::string_view line = reader.require("the number of physical names");
    uint64_t count = 0;
    if (!take(line, count) || !trim(line).empty()) reader.fail("invalid number of physical names");

    for (uint64_t i = 0; i < count; ++i) {
        line = reader.require("a physical name");
        int dimension = 0;
        long long tag = 0;
        if (!take(line, dimension) || !take(line, tag)) reader.fail("invalid physical name line");
        line = trim(line);
        if (line.size() < 2 || line.front() != '"' || line.back() != '"')
            reader.fail("physical name must be in double quotes");
        const std::string name(trim(line.substr(1, line.size() - 2)));
        if (name.empty()) reader.fail("empty physical name");

        if (dimension == 3) {
            if (state.volume_index.contains(tag)) reader.fail("volume tag " + std::to_string(tag) + " is named twice");
            if (mesh.materials.size() > std::numeric_limits<uint16_t>::max()) reader.fail("too many volumes");

            // A volume name can carry a voltage, such as "silicon 5V": the material is
            // "silicon" and the volume is held at 5 V.
            MeshProperties::Material material;
            material.name = name.starts_with("detector") ? "detector" : std::regex_replace(name, voltage, "");
            material.name = std::string(trim(material.name));
            if (material.name.empty()) reader.fail("volume name '" + name + "' has no material name");

            std::smatch match;
            if (std::regex_search(name, match, voltage)) {
                double value = 0;
                if (!parse_number(match[2].str(), value)) reader.fail("invalid voltage in volume name '" + name + "'");
                material.voltage = value;
                // As in Nebula: a detector with a voltage is a metal.
                if (material.name == "detector") material.is_metal = true;
            }

            state.volume_index[tag] = static_cast<uint16_t>(mesh.materials.size());
            mesh.materials.push_back(material);
        } else if (dimension == 2) {
            if (state.surface_index.contains(tag)) reader.fail("surface tag " + std::to_string(tag) + " is named twice");
            if (mesh.boundary_conditions.size() > std::numeric_limits<uint16_t>::max()) reader.fail("too many surfaces");

            const auto condition = parse_boundary_condition(name);
            if (!condition)
                reader.fail("surface name '" + name + "' is not a boundary condition; expected "
                            "'<number>V' (Dirichlet) or 'D<number>' (Neumann)");

            state.surface_index[tag] = static_cast<uint16_t>(mesh.boundary_conditions.size());
            mesh.boundary_conditions.push_back(*condition);
            state.surface_names.push_back(name);
        }
        // Names of points and lines are not used.
    }
    reader.expect("$EndPhysicalNames");
}

void read_nodes(TextReader& reader, MeshProperties& mesh, MeshReadState& state) {
    std::string_view line = reader.require("the number of nodes");
    uint64_t count = 0;
    if (!take(line, count) || !trim(line).empty()) reader.fail("invalid number of nodes");
    if (count == 0) reader.fail("the mesh has no nodes");
    if (count >= NodeIndex::none) reader.fail("too many nodes");

    mesh.nodes.reserve(count);
    state.node_ids.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        line = reader.require("a node");
        uint64_t id = 0;
        std::array<double, 3> x{};
        if (!take(line, id) || !take(line, x[0]) || !take(line, x[1]) || !take(line, x[2]) || !trim(line).empty())
            reader.fail("invalid node line; expected 'id x y z'");
        if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2]))
            reader.fail("node coordinates must be finite");
        state.node_ids.push_back(id);
        mesh.nodes.push_back(x);
    }
    reader.expect("$EndNodes");

    uint64_t duplicate = 0;
    if (!state.node_index.build(state.node_ids, duplicate))
        reader.fail("node id " + std::to_string(duplicate) + " occurs more than once");
}

// Six times the signed volume of a tet.
double signed_volume6(const MeshProperties& mesh, const std::array<uint32_t, 4>& node) {
    const auto& a = mesh.nodes[node[0]];
    double e[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) e[i][k] = mesh.nodes[node[i + 1]][k] - a[k];
    return e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
           e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
           e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
}

double longest_edge(const MeshProperties& mesh, const std::array<uint32_t, 4>& node) {
    double longest2 = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            double length2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = mesh.nodes[node[i]][k] - mesh.nodes[node[j]][k];
                length2 += d * d;
            }
            longest2 = std::max(longest2, length2);
        }
    return std::sqrt(longest2);
}

void read_elements(TextReader& reader, MeshProperties& mesh, MeshReadState& state) {
    std::string_view line = reader.require("the number of elements");
    uint64_t count = 0;
    if (!take(line, count) || !trim(line).empty()) reader.fail("invalid number of elements");

    state.material_use.assign(mesh.materials.size(), 0);
    state.boundary_use.assign(mesh.boundary_conditions.size(), 0);
    uint64_t skipped = 0, reoriented = 0;

    for (uint64_t i = 0; i < count; ++i) {
        line = reader.require("an element");
        uint64_t id = 0;
        int type = 0, tag_count = 0;
        if (!take(line, id) || !take(line, type) || !take(line, tag_count))
            reader.fail("invalid element line; expected 'id type number_of_tags tags... nodes...'");

        if (type == 15 || type == 1) {   // points and lines carry nothing we use
            ++skipped;
            continue;
        }
        if (type != 4 && type != 2)
            reader.fail("unsupported element type " + std::to_string(type) +
                        "; only 4-node tetrahedra (4) and 3-node triangles (2) are supported");

        if (tag_count < 1) reader.fail("element has no physical tag");
        long long physical_tag = 0;
        for (int t = 0; t < tag_count; ++t) {
            long long tag = 0;
            if (!take(line, tag)) reader.fail("invalid element tag");
            if (t == 0) physical_tag = tag;
        }

        const int node_count = type == 4 ? 4 : 3;
        std::array<uint32_t, 4> node{};
        for (int n = 0; n < node_count; ++n) {
            uint64_t node_id = 0;
            if (!take(line, node_id)) reader.fail("element has too few nodes or an invalid node id");
            node[n] = state.node_index.find(node_id);
            if (node[n] == NodeIndex::none) reader.fail("node id " + std::to_string(node_id) + " does not exist");
            for (int m = 0; m < n; ++m)
                if (node[m] == node[n]) reader.fail("element uses node id " + std::to_string(node_id) + " twice");
        }
        if (!trim(line).empty()) reader.fail("element has too many nodes");

        if (type == 4) {
            const auto material = state.volume_index.find(physical_tag);
            if (material == state.volume_index.end())
                reader.fail("tetrahedron has physical tag " + std::to_string(physical_tag) +
                            ", which is not a named volume in $PhysicalNames");

            // Fix the node order so that every tet has a positive volume.
            const double volume6 = signed_volume6(mesh, node);
            const double edge = longest_edge(mesh, node);
            if (std::abs(volume6) <= 1e-12 * edge * edge * edge) reader.fail("tetrahedron has zero volume");
            if (volume6 < 0) {
                std::swap(node[0], node[1]);
                ++reoriented;
            }

            mesh.elements.push_back({node, material->second});
            ++state.material_use[material->second];
        } else {
            const auto condition = state.surface_index.find(physical_tag);
            if (condition == state.surface_index.end())
                reader.fail("triangle has physical tag " + std::to_string(physical_tag) +
                            ", which is not a named surface in $PhysicalNames");

            mesh.boundary_elements.push_back({{node[0], node[1], node[2]}, condition->second});
            state.boundary_lines.push_back(reader.line());
            ++state.boundary_use[condition->second];
        }
    }
    reader.expect("$EndElements");

    if (skipped > 0)
        mesh.info.warnings.push_back(std::to_string(skipped) + " point and line elements were skipped");
    if (reoriented > 0)
        mesh.info.warnings.push_back(std::to_string(reoriented) + " tetrahedra had a negative volume; their node order was fixed");
}

void read_mesh_file(const fs::path& file, MeshProperties& mesh, MeshReadState& state) {
    const std::string text = read_file(file);
    TextReader reader(file, text);

    bool has_format = false, has_names = false, has_nodes = false, has_elements = false;
    std::string_view line;
    while (reader.next(line)) {
        if (line.empty()) continue;
        if (!has_format && line != "$MeshFormat") reader.fail("the file must start with $MeshFormat; is this a Gmsh file?");
        if (line.front() != '$') reader.fail("expected the start of a section, found '" + std::string(line) + "'");

        if (line == "$MeshFormat") {
            if (has_format) reader.fail("second $MeshFormat section");
            read_mesh_format(reader);
            has_format = true;
        } else if (line == "$PhysicalNames") {
            if (has_names) reader.fail("second $PhysicalNames section");
            if (has_elements) reader.fail("$PhysicalNames must come before $Elements");
            read_physical_names(reader, mesh, state);
            has_names = true;
        } else if (line == "$Nodes") {
            if (has_nodes) reader.fail("second $Nodes section");
            if (has_elements) reader.fail("$Nodes must come before $Elements");
            read_nodes(reader, mesh, state);
            has_nodes = true;
        } else if (line == "$Elements") {
            if (has_elements) reader.fail("second $Elements section");
            if (!has_names) reader.fail("$Elements found before $PhysicalNames; the mesh needs named physical groups");
            if (!has_nodes) reader.fail("$Elements found before $Nodes");
            read_elements(reader, mesh, state);
            has_elements = true;
        } else {
            // A section we do not use: skip to its end.
            const std::string end = "$End" + std::string(line.substr(1));
            std::string_view skipped;
            do {
                skipped = reader.require(end);
            } while (skipped != end);
        }
    }

    if (!has_format) fail(file, "no $MeshFormat section; is this a Gmsh file?");
    if (!has_names) fail(file, "no $PhysicalNames section; the mesh needs named physical groups");
    if (!has_nodes) fail(file, "no $Nodes section");
    if (!has_elements) fail(file, "no $Elements section");
    if (mesh.elements.empty()) fail(file, "the mesh has no tetrahedra");

    for (std::size_t m = 0; m < mesh.materials.size(); ++m)
        if (state.material_use[m] == 0)
            mesh.info.warnings.push_back("volume '" + mesh.materials[m].name + "' has no tetrahedra");
    for (std::size_t b = 0; b < mesh.boundary_conditions.size(); ++b)
        if (state.boundary_use[b] == 0)
            mesh.info.warnings.push_back("surface '" + state.surface_names[b] + "' has no triangles");
    if (mesh.boundary_elements.empty())
        mesh.info.warnings.push_back("the mesh has no boundary conditions");
}

// Checks that the mesh fits together: every face belongs to at most two tets, every
// boundary triangle is a face of a tet, and every node is used. Fills mesh.neighbours.
void check_geometry(const fs::path& file, MeshProperties& mesh, const MeshReadState& state) {
    using Face = std::array<uint32_t, 3>;
    const auto sorted = [](Face face) {
        std::sort(face.begin(), face.end());
        return face;
    };
    const auto describe = [&](const Face& face) {
        return "nodes " + std::to_string(state.node_ids[face[0]]) + ", " +
               std::to_string(state.node_ids[face[1]]) + ", " + std::to_string(state.node_ids[face[2]]);
    };

    // Every face of every tet, with the tet it belongs to. Face i is opposite local node i.
    struct TetFace {
        Face face;
        uint32_t element;
        uint8_t local;
    };
    std::vector<TetFace> faces;
    faces.reserve(4 * mesh.elements.size());
    std::vector<bool> used(mesh.nodes.size(), false);
    for (uint32_t e = 0; e < mesh.elements.size(); ++e) {
        const auto& n = mesh.elements[e].node;
        faces.push_back({sorted({n[1], n[2], n[3]}), e, 0});
        faces.push_back({sorted({n[0], n[2], n[3]}), e, 1});
        faces.push_back({sorted({n[0], n[1], n[3]}), e, 2});
        faces.push_back({sorted({n[0], n[1], n[2]}), e, 3});
        for (const uint32_t node : n) used[node] = true;
    }
    std::sort(faces.begin(), faces.end(), [](const TetFace& a, const TetFace& b) { return a.face < b.face; });

    // After sorting, the two tets that share a face are next to each other.
    mesh.neighbours.assign(mesh.elements.size(), {});
    for (std::size_t i = 1; i < faces.size(); ++i) {
        if (faces[i].face != faces[i - 1].face) continue;
        if (i >= 2 && faces[i].face == faces[i - 2].face)
            fail(file, "the face with " + describe(faces[i].face) + " belongs to more than two tetrahedra");
        mesh.neighbours[faces[i].element][faces[i].local] = faces[i - 1].element;
        mesh.neighbours[faces[i - 1].element][faces[i - 1].local] = faces[i].element;
    }

    std::vector<Face> boundary_faces;
    boundary_faces.reserve(mesh.boundary_elements.size());
    for (std::size_t i = 0; i < mesh.boundary_elements.size(); ++i) {
        const Face face = sorted(mesh.boundary_elements[i].node);
        const auto found = std::lower_bound(faces.begin(), faces.end(), face,
                                            [](const TetFace& a, const Face& b) { return a.face < b; });
        if (found == faces.end() || found->face != face)
            fail(file, state.boundary_lines[i], "triangle with " + describe(face) + " is not a face of any tetrahedron");
        boundary_faces.push_back(face);
    }
    std::sort(boundary_faces.begin(), boundary_faces.end());
    uint64_t repeated = 0;
    for (std::size_t i = 1; i < boundary_faces.size(); ++i)
        if (boundary_faces[i] == boundary_faces[i - 1]) ++repeated;
    if (repeated > 0)
        mesh.info.warnings.push_back(std::to_string(repeated) + " boundary triangles are listed more than once");

    const auto unused = std::count(used.begin(), used.end(), false);
    if (unused > 0)
        mesh.info.warnings.push_back(std::to_string(unused) + " nodes do not belong to any tetrahedron");
}

// ---------------------------------------------------------------- materials

struct MaterialFile {
    std::string name;
    std::optional<double> eps_r;
    bool is_metal = false;
};

// Reads only what is needed from a material .yaml file: the top-level "name", "epsilon_r"
// inside "dielectric", and "model" inside "band_structure" (a metal if it says "metal").
MaterialFile read_material_file(const fs::path& file) {
    const std::string text = read_file(file);
    TextReader reader(file, text);
    MaterialFile result;
    bool has_name = false;
    std::string section;

    const auto read_eps_r = [&](std::string_view s) {
        const std::size_t key = s.find("epsilon_r");
        if (key == std::string_view::npos) return;
        s.remove_prefix(key + std::strlen("epsilon_r"));
        s = trim(s);
        if (s.empty() || s.front() != ':') reader.fail("expected 'epsilon_r: <number>'");
        s.remove_prefix(1);
        s = trim(s.substr(0, s.find_first_of(",}")));
        double value = 0;
        if (!parse_number(s, value)) reader.fail("epsilon_r is not a number");
        if (value <= 0) reader.fail("epsilon_r must be positive");
        if (result.eps_r) reader.fail("epsilon_r is given twice");
        result.eps_r = value;
    };

    const auto read_model = [&](std::string_view s) {
        const std::size_t key = s.find("model");
        if (key == std::string_view::npos) return;
        s.remove_prefix(key + std::strlen("model"));
        s = trim(s);
        if (s.empty() || s.front() != ':') return;   // another key that contains "model"
        s.remove_prefix(1);
        result.is_metal = trim(s.substr(0, s.find_first_of(",}"))) == "metal";
    };

    std::size_t pos = 0;
    while (pos < text.size()) {
        // Indentation matters in YAML, so lines are split here instead of using next().
        const std::size_t end = std::min(text.find('\n', pos), text.size());
        std::string_view raw(text.data() + pos, end - pos);
        pos = end + 1;
        std::string_view ignored;
        reader.next(ignored);

        raw = raw.substr(0, raw.find('#'));
        const std::string_view line = trim(raw);
        if (line.empty()) continue;
        const bool top_level = !is_blank(raw.front());

        if (top_level) {
            const std::size_t colon = line.find(':');
            if (colon == std::string_view::npos) reader.fail("expected 'key: value'");
            section = std::string(trim(line.substr(0, colon)));
            std::string_view value = trim(line.substr(colon + 1));
            if (section == "name") {
                if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front())
                    value = value.substr(1, value.size() - 2);
                if (value.empty()) reader.fail("the material name is empty");
                if (has_name) reader.fail("the material name is given twice");
                result.name = std::string(value);
                has_name = true;
            } else if (section == "dielectric") {
                read_eps_r(value);
            } else if (section == "band_structure") {
                read_model(value);
            }
        } else if (section == "dielectric") {
            read_eps_r(line);
        } else if (section == "band_structure") {
            read_model(line);
        }
    }

    if (!has_name) fail(file, "the material file has no 'name'");
    return result;
}

void read_materials(const std::vector<fs::path>& files, MeshProperties& mesh) {
    std::vector<bool> has_file(mesh.materials.size(), false);
    std::vector<std::string> seen;

    for (const fs::path& file : files) {
        const MaterialFile material = read_material_file(file);
        if (std::find(seen.begin(), seen.end(), material.name) != seen.end())
            fail(file, "material '" + material.name + "' is given by more than one file");
        seen.push_back(material.name);

        bool found = false;
        for (std::size_t m = 0; m < mesh.materials.size(); ++m) {
            if (mesh.materials[m].name != material.name) continue;
            found = true;
            has_file[m] = true;
            if (material.eps_r) mesh.materials[m].eps_r = *material.eps_r;
            if (material.is_metal) mesh.materials[m].is_metal = true;
        }
        if (!found)
            mesh.info.warnings.push_back("material '" + material.name + "' (" + file.string() + ") is not used by the mesh");
        else if (!material.eps_r && !material.is_metal)   // eps_r means nothing for a metal
            mesh.info.warnings.push_back("material '" + material.name + "' (" + file.string() +
                                         ") has no dielectric epsilon_r; eps_r is left at 1");
    }

    for (std::size_t m = 0; m < mesh.materials.size(); ++m)
        if (!has_file[m] && mesh.materials[m].name != "vacuum")
            mesh.info.warnings.push_back("no material file for volume '" + mesh.materials[m].name + "'; eps_r is left at 1");

    for (const auto& material : mesh.materials)
        if (material.voltage && !material.is_metal)
            mesh.info.warnings.push_back("volume '" + material.name + "' has a voltage but its material is not a metal");
}

// ---------------------------------------------------------------- binary files

// The binary files are little-endian whatever machine this runs on.
uint32_t read_u32(const char* p) {
    unsigned char b[4];
    std::memcpy(b, p, 4);
    return uint32_t{b[0]} | uint32_t{b[1]} << 8 | uint32_t{b[2]} << 16 | uint32_t{b[3]} << 24;
}
double read_float(const char* p) { return std::bit_cast<float>(read_u32(p)); }
int32_t read_int(const char* p) { return static_cast<int32_t>(read_u32(p)); }

[[noreturn]] void fail_record(const fs::path& file, std::size_t record, const std::string& message) {
    fail(file, "record " + std::to_string(record) + ": " + message);
}

// .cha: 16 bytes per charge: x, y, z as float32, q as int32.
void read_charges(const fs::path& file, MeshProperties& mesh) {
    constexpr std::size_t record_size = 16;
    const std::string data = read_file(file);
    if (data.size() % record_size != 0)
        fail(file, "file size " + std::to_string(data.size()) + " is not a multiple of 16 bytes; is this a .cha file?");
    const std::size_t count = data.size() / record_size;
    if (count == 0) mesh.info.warnings.push_back("charge file " + file.string() + " contains no charges");

    std::array<double, 3> low = mesh.nodes.front(), high = mesh.nodes.front();
    for (const auto& node : mesh.nodes)
        for (int k = 0; k < 3; ++k) {
            low[k] = std::min(low[k], node[k]);
            high[k] = std::max(high[k], node[k]);
        }

    mesh.charges.reserve(mesh.charges.size() + count);
    std::size_t outside = 0, zero = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const char* p = data.data() + i * record_size;
        const std::array<double, 3> position{read_float(p), read_float(p + 4), read_float(p + 8)};
        const int32_t value = read_int(p + 12);

        if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
            fail_record(file, i, "position is not finite");
        if (value == 0) ++zero;

        for (int k = 0; k < 3; ++k)
            if (position[k] < low[k] || position[k] > high[k]) {
                ++outside;
                break;
            }
        mesh.charges.push_back({position, value});
    }
    if (zero > 0) mesh.info.warnings.push_back(std::to_string(zero) + " trapped charges have a charge of zero");
    if (outside > 0)
        mesh.info.warnings.push_back(std::to_string(outside) + " trapped charges lie outside the bounding box of the mesh");
}

}   // namespace

MeshProperties Importer::import_mesh(const std::filesystem::path& mesh_file,
                                     const std::vector<std::filesystem::path>& material_files,
                                     const std::optional<std::filesystem::path>& charges_file) const {
    const auto start = Clock::now();
    MeshProperties mesh;
    MeshReadState state;

    read_mesh_file(mesh_file, mesh, state);
    mesh.info.time_read = seconds_since(start);

    const auto geometry_start = Clock::now();
    check_geometry(mesh_file, mesh, state);
    mesh.info.time_geometry = seconds_since(geometry_start);

    read_materials(material_files, mesh);
    if (charges_file) read_charges(*charges_file, mesh);

    mesh.info.time = seconds_since(start);
    return mesh;
}

// .pri: 36 bytes per electron: x, y, z, dx, dy, dz, energy as float32, pixel_x, pixel_y as int32.
FreeCharges Importer::import_free_charges(const std::filesystem::path& primaries_file) const {
    const auto start = Clock::now();
    constexpr std::size_t record_size = 36;
    FreeCharges result;

    const std::string data = read_file(primaries_file);
    if (data.size() % record_size != 0)
        fail(primaries_file, "file size " + std::to_string(data.size()) + " is not a multiple of 36 bytes; is this a .pri file?");
    const std::size_t count = data.size() / record_size;
    if (count == 0) result.info.warnings.push_back("primaries file " + primaries_file.string() + " contains no electrons");

    result.electrons.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const char* p = data.data() + i * record_size;
        FreeCharges::Electron electron;
        electron.position = {read_float(p), read_float(p + 4), read_float(p + 8)};
        electron.direction = {read_float(p + 12), read_float(p + 16), read_float(p + 20)};
        electron.energy = read_float(p + 24);
        electron.pixel = {read_int(p + 28), read_int(p + 32)};

        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(electron.position[k]) || !std::isfinite(electron.direction[k]))
                fail_record(primaries_file, i, "position or direction is not finite");
        if (electron.direction[0] == 0 && electron.direction[1] == 0 && electron.direction[2] == 0)
            fail_record(primaries_file, i, "direction is zero");
        if (!std::isfinite(electron.energy) || electron.energy <= 0)
            fail_record(primaries_file, i, "energy must be positive");

        result.electrons.push_back(electron);
    }

    result.info.time = seconds_since(start);
    return result;
}
