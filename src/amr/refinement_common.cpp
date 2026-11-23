#include "refinement_common.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace amr_detail {

namespace {

using Vec3 = std::array<double, 3>;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

using Key = std::array<uint32_t, 3>;
Key sorted(Key key) {
    std::sort(key.begin(), key.end());
    return key;
}
Key face_nodes(const std::array<uint32_t, 4>& n, int opposite) {
    Key face{};
    for (int j = 0, f = 0; j < 4; ++j)
        if (j != opposite) face[f++] = n[j];
    return face;
}

} // namespace

WorkMesh::WorkMesh(const MeshProperties& input_mesh, uint64_t max_new_nodes, std::string who)
    : input(input_mesh), nodes(input_mesh.nodes), max_new_nodes_(max_new_nodes), who_(std::move(who)) {
    node_parents.reserve(nodes.size());
    for (uint32_t i = 0; i < nodes.size(); ++i) node_parents.push_back({i, i});
}

std::optional<uint32_t> WorkMesh::find_midpoint(uint32_t a, uint32_t b) const {
    const auto found = midpoints_.find(edge_key(a, b));
    if (found == midpoints_.end()) return std::nullopt;
    return found->second;
}

uint32_t WorkMesh::midpoint(uint32_t a, uint32_t b) {
    const auto [it, inserted] = midpoints_.try_emplace(edge_key(a, b), 0);
    if (!inserted) return it->second;
    if (new_nodes() >= max_new_nodes_) {
        midpoints_.erase(it);
        throw std::runtime_error(who_ + ": more than " + std::to_string(max_new_nodes_) +
                                 " new nodes (Settings::max_new_nodes)");
    }
    if (a > b) std::swap(a, b);
    const Vec3 &p = nodes[a], &q = nodes[b];
    nodes.push_back({0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])});
    node_parents.push_back({a, b});
    it->second = static_cast<uint32_t>(nodes.size() - 1);
    return it->second;
}

double WorkMesh::length2(uint32_t a, uint32_t b) const {
    if (a > b) std::swap(a, b);
    const Vec3 d = sub(nodes[b], nodes[a]);
    return dot(d, d);
}

double volume6(const std::vector<std::array<double, 3>>& nodes, const std::array<uint32_t, 4>& n) {
    const Vec3& p0 = nodes[n[0]];
    return dot(sub(nodes[n[1]], p0), cross(sub(nodes[n[2]], p0), sub(nodes[n[3]], p0)));
}

void WorkMesh::subdivide(const WorkTet& t, std::span<const ChildTemplate> children, std::vector<WorkTet>& out) {
    for (const ChildTemplate& tmpl : children) {
        WorkTet child;
        child.material = t.material;
        std::array<unsigned, 4> mask{};
        for (int q = 0; q < 4; ++q) {
            const LocalPoint p = tmpl[q];
            child.node[q] = p.a == p.b ? t.node[p.a] : midpoint(t.node[p.a], t.node[p.b]);
            mask[q] = (1u << p.a) | (1u << p.b);
        }
        // A child face lies in a face of t when its points use only three of t's nodes.
        for (int q = 0; q < 4; ++q) {
            unsigned used = 0;
            for (int r = 0; r < 4; ++r)
                if (r != q) used |= mask[r];
            child.face_tag[q] = std::popcount(used) == 3 ? t.face_tag[std::countr_zero(~used & 0xFu)] : no_tag;
        }
        const double v = volume6(nodes, child.node);
        if (v == 0) throw std::logic_error(who_ + ": made a tet of zero volume");
        if (v < 0) {
            std::swap(child.node[2], child.node[3]);
            std::swap(child.face_tag[2], child.face_tag[3]);
        }
        child.ancestor = t.ancestor;
        out.push_back(child);
    }
}

std::vector<WorkTet> initial_tets(const MeshProperties& input) {
    std::vector<std::pair<Key, int32_t>> boundary;
    boundary.reserve(input.boundary_elements.size());
    for (std::size_t b = 0; b < input.boundary_elements.size(); ++b)
        boundary.emplace_back(sorted(input.boundary_elements[b].node), static_cast<int32_t>(b));
    std::sort(boundary.begin(), boundary.end());

    std::vector<WorkTet> tets(input.elements.size());
    for (uint32_t k = 0; k < input.elements.size(); ++k) {
        WorkTet& t = tets[k];
        t.node = input.elements[k].node;
        t.material = input.elements[k].material;
        t.ancestor = k;
        for (int i = 0; i < 4; ++i) {
            const Key key = sorted(face_nodes(t.node, i));
            const auto found = std::lower_bound(boundary.begin(), boundary.end(), std::make_pair(key, int32_t{-1}));
            t.face_tag[i] = found != boundary.end() && found->first == key ? found->second : no_tag;
        }
    }
    return tets;
}

RefinementResult output(const WorkMesh& work, const std::vector<WorkTet>& tets) {
    const MeshProperties& input = work.input;
    RefinementResult result;
    MeshProperties& mesh = result.mesh;
    mesh.nodes = work.nodes;
    mesh.elements.reserve(tets.size());
    result.parents.elements.reserve(tets.size());

    struct Tagged {
        Key key;
        Key nodes;
        int32_t tag;
    };
    std::vector<Tagged> tagged;
    for (const WorkTet& t : tets) {
        mesh.elements.push_back({t.node, t.material});
        result.parents.elements.push_back(t.ancestor);
        for (int i = 0; i < 4; ++i)
            if (t.face_tag[i] != no_tag) {
                const Key nodes = face_nodes(t.node, i);
                tagged.push_back({sorted(nodes), nodes, t.face_tag[i]});
            }
    }
    // A tagged interior face is seen from both sides; keep it once.
    std::sort(tagged.begin(), tagged.end(), [](const Tagged& a, const Tagged& b) { return a.key < b.key; });
    tagged.erase(std::unique(tagged.begin(), tagged.end(), [](const Tagged& a, const Tagged& b) { return a.key == b.key; }),
                 tagged.end());
    mesh.boundary_elements.reserve(tagged.size());
    for (const Tagged& f : tagged) {
        // Same orientation as the input triangle it lies in.
        const auto& original = input.boundary_elements[f.tag];
        const Vec3 n_in = cross(sub(input.nodes[original.node[1]], input.nodes[original.node[0]]),
                                sub(input.nodes[original.node[2]], input.nodes[original.node[0]]));
        Key nodes = f.nodes;
        const Vec3 n_out = cross(sub(mesh.nodes[nodes[1]], mesh.nodes[nodes[0]]), sub(mesh.nodes[nodes[2]], mesh.nodes[nodes[0]]));
        if (dot(n_in, n_out) < 0) std::swap(nodes[1], nodes[2]);
        mesh.boundary_elements.push_back({nodes, original.boundary_condition});
    }

    mesh.boundary_conditions = input.boundary_conditions;
    mesh.materials = input.materials;
    mesh.charges = input.charges;
    mesh.screened = input.screened;
    build_neighbours(mesh);
    result.parents.nodes = work.node_parents;
    return result;
}

void build_neighbours(MeshProperties& mesh) {
    struct TetFace {
        Key      key;
        uint32_t element;
        uint8_t  local;
    };
    std::vector<TetFace> all;
    all.reserve(4 * mesh.elements.size());
    for (uint32_t k = 0; k < mesh.elements.size(); ++k)
        for (uint8_t i = 0; i < 4; ++i) all.push_back({sorted(face_nodes(mesh.elements[k].node, i)), k, i});
    std::sort(all.begin(), all.end(), [](const TetFace& a, const TetFace& b) { return a.key < b.key; });
    mesh.neighbours.assign(mesh.elements.size(), {});
    for (std::size_t i = 1; i < all.size(); ++i)
        if (all[i].key == all[i - 1].key) {
            mesh.neighbours[all[i].element][all[i].local] = all[i - 1].element;
            mesh.neighbours[all[i - 1].element][all[i - 1].local] = all[i].element;
        }
}

} // namespace amr_detail
