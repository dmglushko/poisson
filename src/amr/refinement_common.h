#pragma once

// Private to the refiners: a mesh under refinement and the one operation both refiners are
// built from, replacing a tet by children described in its own local terms.

#include "amr/refiner.h"
#include "mesh_properties.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amr_detail {

// A point of a tet in local terms: node a (a == b), or the midpoint of the edge a-b (a < b).
struct LocalPoint {
    uint8_t a, b;
};
constexpr LocalPoint vertex(int i) { return {static_cast<uint8_t>(i), static_cast<uint8_t>(i)}; }
constexpr LocalPoint edge(int i, int j) {
    return {static_cast<uint8_t>(i < j ? i : j), static_cast<uint8_t>(i < j ? j : i)};
}
using ChildTemplate = std::array<LocalPoint, 4>;

// The 6 edges of a tet by local nodes, in the order of RefinementHistory::GreenFamily.
constexpr std::array<std::pair<int, int>, 6> tet_edges{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

constexpr int32_t no_tag = -1;

struct WorkTet {
    std::array<uint32_t, 4> node;
    // Per face (opposite node i): the input boundary element it lies in, or no_tag.
    std::array<int32_t, 4> face_tag;
    uint16_t material;
    uint32_t ancestor;                            // the input tet that contains it
};

inline uint64_t edge_key(uint32_t a, uint32_t b) {
    return a < b ? (uint64_t{a} << 32) | b : (uint64_t{b} << 32) | a;
}

class WorkMesh {
public:
    WorkMesh(const MeshProperties& input, uint64_t max_new_nodes, std::string who);

    const MeshProperties& input;
    std::vector<std::array<double, 3>> nodes;
    std::vector<std::array<uint32_t, 2>> node_parents;

    std::optional<uint32_t> find_midpoint(uint32_t a, uint32_t b) const;
    // The midpoint node of a-b, made if it does not exist yet. Throws std::runtime_error
    // beyond max_new_nodes.
    uint32_t midpoint(uint32_t a, uint32_t b);
    // Records an existing node as the midpoint of a-b (e.g. from prior refinement history).
    void set_midpoint(uint32_t a, uint32_t b, uint32_t m) { midpoints_[edge_key(a, b)] = m; }

    // Squared length of a-b, computed the same way whatever the order of a and b, so equal
    // edges compare equal in every tet.
    double length2(uint32_t a, uint32_t b) const;

    // Appends the children of t, given in t's local terms, to `out`. A child keeps t's
    // material; its faces inherit t's face tags where they lie in a face of t; it is turned
    // to a positive volume by swapping two nodes if needed.
    void subdivide(const WorkTet& t, std::span<const ChildTemplate> children, std::vector<WorkTet>& out);

    uint64_t new_nodes() const { return nodes.size() - input.nodes.size(); }

private:
    std::unordered_map<uint64_t, uint32_t> midpoints_;
    uint64_t max_new_nodes_;
    std::string who_;
};

// The input tets as work tets, with their face tags from the boundary elements.
std::vector<WorkTet> initial_tets(const MeshProperties& input);

// 6 times the signed volume.
double volume6(const std::vector<std::array<double, 3>>& nodes, const std::array<uint32_t, 4>& n);

// The refined mesh, its parents and the boundary elements from the face tags. The history
// is left to the caller.
RefinementResult output(const WorkMesh& work, const std::vector<WorkTet>& tets);

// Fills mesh.neighbours from the faces (as the Importer does).
void build_neighbours(MeshProperties& mesh);

} // namespace amr_detail
