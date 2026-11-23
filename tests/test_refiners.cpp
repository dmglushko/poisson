// Tests for the refiners: after every refinement the mesh must be a valid, conforming tet
// mesh of the same domain, with the boundary conditions, materials and parents right.

#include "amr/longest_edge_bisection.h"
#include "amr/red_green_refinement.h"
#include "test_support.h"

#include <limits>
#include <map>
#include <set>
#include <stdexcept>

using Key = std::array<uint32_t, 3>;

Key sorted_key(Key k) {
    std::sort(k.begin(), k.end());
    return k;
}

double triangle_area(const MeshProperties& mesh, const Key& t) {
    const auto& a = mesh.nodes[t[0]];
    const auto& b = mesh.nodes[t[1]];
    const auto& c = mesh.nodes[t[2]];
    const Vec3 u{b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const Vec3 n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

// Volume / longest edge^3, divided by the same for a regular tet: 1 is perfect.
double quality(const MeshProperties& mesh, const std::array<uint32_t, 4>& n) {
    double longest = 0;
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) {
            double l = 0;
            for (int d = 0; d < 3; ++d) l += std::pow(mesh.nodes[n[a]][d] - mesh.nodes[n[b]][d], 2);
            longest = std::max(longest, std::sqrt(l));
        }
    return signed_volume(mesh, n) / std::pow(longest, 3) * 6 * std::sqrt(2.0);
}

double smallest_quality(const MeshProperties& mesh) {
    double q = std::numeric_limits<double>::infinity();
    for (const auto& e : mesh.elements) q = std::min(q, quality(mesh, e.node));
    return q;
}

// Everything a refined mesh must satisfy, relative to the mesh it came from.
void check_refined(const MeshProperties& before, const RefinementResult& r, std::span<const uint32_t> marked) {
    const MeshProperties& mesh = r.mesh;
    CHECK(mesh.screened == before.screened);
    // Input nodes first and unchanged.
    for (std::size_t i = 0; i < before.nodes.size(); ++i) CHECK(mesh.nodes[i] == before.nodes[i]);
    // Positive volumes, same total volume per material.
    std::map<uint16_t, double> volume_before, volume_after;
    for (const auto& e : before.elements) volume_before[e.material] += signed_volume(before, e.node);
    bool positive = true;
    for (const auto& e : mesh.elements) {
        const double v = signed_volume(mesh, e.node);
        positive = positive && v > 0;
        volume_after[e.material] += v;
    }
    CHECK(positive);
    CHECK(volume_before.size() == volume_after.size());
    for (const auto& [m, v] : volume_before) CHECK_NEAR(volume_after[m], v, 1e-12);

    // Conforming: no face in more than two tets, and the faces in one tet cover exactly the
    // outer boundary (a hanging node would leave unmatched faces inside).
    std::map<Key, int> faces;
    std::map<Key, std::pair<uint32_t, int>> where;
    for (uint32_t e = 0; e < mesh.elements.size(); ++e)
        for (int f = 0; f < 4; ++f) {
            Key k{};
            for (int a = 0, b = 0; a < 4; ++a)
                if (a != f) k[b++] = mesh.elements[e].node[a];
            k = sorted_key(k);
            ++faces[k];
            where[k] = {e, f};
        }
    double outer_after = 0;
    bool at_most_two = true;
    for (const auto& [k, count] : faces) {
        at_most_two = at_most_two && count <= 2;
        if (count == 1) outer_after += triangle_area(mesh, k);
    }
    CHECK(at_most_two);
    std::map<Key, int> faces_before;
    for (const auto& e : before.elements)
        for (int f = 0; f < 4; ++f) {
            Key k{};
            for (int a = 0, b = 0; a < 4; ++a)
                if (a != f) k[b++] = e.node[a];
            ++faces_before[sorted_key(k)];
        }
    double outer_before = 0;
    for (const auto& [k, count] : faces_before)
        if (count == 1) outer_before += triangle_area(before, k);
    CHECK_NEAR(outer_after, outer_before, 1e-12);

    // Boundary elements: faces of the mesh, the same area per condition, outward like the input.
    std::map<uint16_t, double> bc_before, bc_after;
    for (const auto& b : before.boundary_elements) bc_before[b.boundary_condition] += triangle_area(before, b.node);
    bool on_faces = true, outward = true;
    for (const auto& b : mesh.boundary_elements) {
        bc_after[b.boundary_condition] += triangle_area(mesh, b.node);
        const auto found = where.find(sorted_key(b.node));
        on_faces = on_faces && found != where.end();
        if (found == where.end()) continue;
        // Outward: the tet's fourth node lies behind the triangle.
        const auto& e = mesh.elements[found->second.first];
        outward = outward && signed_volume(mesh, {b.node[0], b.node[1], b.node[2], e.node[found->second.second]}) < 0;
    }
    CHECK(on_faces);
    CHECK(outward);
    for (const auto& [bc, area] : bc_before) CHECK_NEAR(bc_after[bc], area, 1e-12);

    // Neighbours: exact reciprocal IDs across every shared face.
    CHECK(mesh.neighbours.size() == mesh.elements.size());
    bool neighbours_right = true;
    for (uint32_t e = 0; e < mesh.elements.size(); ++e)
        for (int f = 0; f < 4; ++f) {
            Key k{};
            for (int a = 0, b = 0; a < 4; ++a)
                if (a != f) k[b++] = mesh.elements[e].node[a];
            const Key sk = sorted_key(k);
            const bool shared = faces[sk] == 2;
            const auto nb = mesh.neighbours[e][f];
            neighbours_right = neighbours_right && nb.has_value() == shared;
            if (!nb) continue;
            neighbours_right = neighbours_right && *nb < mesh.elements.size() && *nb != e;
            if (*nb >= mesh.elements.size()) continue;
            bool back_match = false;
            for (int nf = 0; nf < 4; ++nf) {
                Key nk{};
                for (int a = 0, b = 0; a < 4; ++a)
                    if (a != nf) nk[b++] = mesh.elements[*nb].node[a];
                if (sorted_key(nk) == sk) {
                    back_match = mesh.neighbours[*nb][nf].has_value() && *mesh.neighbours[*nb][nf] == e;
                    break;
                }
            }
            neighbours_right = neighbours_right && back_match;
        }
    CHECK(neighbours_right);

    // Parents: new nodes are midpoints of earlier nodes; each tet's centroid lies in its parent.
    CHECK(r.parents.nodes.size() == mesh.nodes.size());
    CHECK(r.parents.elements.size() == mesh.elements.size());
    bool nodes_right = true, elements_right = true;
    for (uint32_t i = 0; i < mesh.nodes.size(); ++i) {
        const auto [a, b] = r.parents.nodes[i];
        if (i < before.nodes.size()) {
            nodes_right = nodes_right && a == i && b == i;
            continue;
        }
        nodes_right = nodes_right && a < b && b < i;
        for (int d = 0; d < 3; ++d)
            nodes_right = nodes_right && mesh.nodes[i][d] == 0.5 * (mesh.nodes[a][d] + mesh.nodes[b][d]);
    }
    for (uint32_t e = 0; e < mesh.elements.size(); ++e) {
        Vec3 c{0, 0, 0};
        for (uint32_t node : mesh.elements[e].node)
            for (int d = 0; d < 3; ++d) c[d] += 0.25 * mesh.nodes[node][d];
        const auto& parent = before.elements[r.parents.elements[e]];
        const double whole = signed_volume(before, parent.node);
        for (int i = 0; i < 4; ++i) {
            MeshProperties probe;
            probe.nodes = {before.nodes[parent.node[0]], before.nodes[parent.node[1]], before.nodes[parent.node[2]],
                           before.nodes[parent.node[3]]};
            probe.nodes[i] = c;
            elements_right = elements_right && signed_volume(probe, {0, 1, 2, 3}) / whole > -1e-9;
        }
        elements_right = elements_right && parent.material == mesh.elements[e].material;
    }
    CHECK(nodes_right);
    CHECK(elements_right);

    // No marked tet survives unchanged.
    std::set<std::array<uint32_t, 4>> after;
    for (const auto& e : mesh.elements) {
        auto n = e.node;
        std::sort(n.begin(), n.end());
        after.insert(n);
    }
    bool marked_refined = true;
    for (uint32_t k : marked) {
        auto n = before.elements[k].node;
        std::sort(n.begin(), n.end());
        marked_refined = marked_refined && !after.contains(n);
    }
    CHECK(marked_refined);
    CHECK(mesh.charges.size() == before.charges.size());
    CHECK(mesh.materials.size() == before.materials.size());
}

void test_single_tet() {
    MeshProperties mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.elements = {{{0, 1, 2, 3}, 0}};
    mesh.materials = {{"one", 1, 0, false, {}}};
    mesh.boundary_conditions = {{MeshProperties::BcType::Dirichlet, 0, {}}};
    mesh.boundary_elements = {{{0, 2, 1}, 0}, {{0, 1, 3}, 0}, {{0, 3, 2}, 0}, {{1, 2, 3}, 0}};
    const std::vector<uint32_t> marked{0};
    const RefinementResult leb = LongestEdgeBisection().refine(mesh, marked);
    CHECK(leb.mesh.elements.size() == 2);
    CHECK(leb.mesh.boundary_elements.size() == 6);
    check_refined(mesh, leb, marked);
    const RefinementResult rg = RedGreenRefinement().refine(mesh, marked);
    CHECK(rg.mesh.elements.size() == 8);
    CHECK(rg.mesh.nodes.size() == 10);
    CHECK(rg.mesh.boundary_elements.size() == 16);
    CHECK(rg.history.families.empty());
    check_refined(mesh, rg, marked);
    CHECK(rg.info.refiner == "red_green");
    CHECK(rg.info.details.at("red") == 1);
}

// Refine around a point for several cycles; check every mesh and how the shapes hold up.
void cycles(const Refiner& refiner, int count, bool jitter) {
    using Bc = MeshProperties::BoundaryCondition;
    const std::vector<MeshProperties::Material> materials{{"a", 1, 0, false, {}}, {"b", 4, 0, false, {}}};
    MeshProperties mesh = cube_mesh(3, {0, 1, 2, 2, 2, 2},
                                    {Bc{MeshProperties::BcType::Dirichlet, 0, {}}, Bc{MeshProperties::BcType::Dirichlet, 1, {}},
                                     Bc{MeshProperties::BcType::Neumann, 0, {}}},
                                    materials, 0.5);
    mesh.screened = true;
    mesh.charges = {{{0.3, 0.3, 0.3}, -1}};
    if (jitter) {
        // Move the interior nodes by up to 0.15 h, the same way every run: an irregular mesh.
        uint32_t state = 12345;
        const auto next = [&] {
            state = state * 1664525u + 1013904223u;
            return (state >> 8) / double(1 << 24) - 0.5;
        };
        for (auto& p : mesh.nodes) {
            const bool interior = p[0] > 0 && p[0] < 1 && p[1] > 0 && p[1] < 1 && p[2] > 0 && p[2] < 1 && p[0] != 0.5;
            const std::array<double, 3> d{next(), next(), next()};
            if (interior)
                for (int i = 0; i < 3; ++i) p[i] += 0.3 / 3 * d[i];
        }
        for (auto& e : mesh.elements) {
            double cx = 0;
            for (uint32_t node : e.node) cx += mesh.nodes[node][0] / 4;
            e.material = cx > 0.5 ? 1 : 0;
        }
    }
    const double initial_quality = smallest_quality(mesh);
    RefinementHistory history;
    const Vec3 target{0.31, 0.27, 0.62};
    for (int cycle = 0; cycle < count; ++cycle) {
        // Mark the tets with a node near the target, closer each cycle.
        const double radius = 0.4 / std::pow(1.6, cycle);
        std::vector<uint32_t> marked;
        for (uint32_t e = 0; e < mesh.elements.size(); ++e) {
            // The tet that contains the target, always.
            const auto& n = mesh.elements[e].node;
            bool inside = true;
            for (int i = 0; i < 4; ++i) {
                MeshProperties copy;
                copy.nodes = {mesh.nodes[n[0]], mesh.nodes[n[1]], mesh.nodes[n[2]], mesh.nodes[n[3]]};
                copy.nodes[i] = target;
                inside = inside && signed_volume(copy, {0, 1, 2, 3}) >= 0;
            }
            if (inside) {
                marked.push_back(e);
                continue;
            }
            for (uint32_t node : mesh.elements[e].node) {
                const auto& p = mesh.nodes[node];
                if (std::hypot(p[0] - target[0], p[1] - target[1], p[2] - target[2]) < radius) {
                    marked.push_back(e);
                    break;
                }
            }
        }
        const RefinementResult r = refiner.refine(mesh, marked, history);
        check_refined(mesh, r, marked);
        if (!r.history.family.empty()) CHECK(r.history.family.size() == r.mesh.elements.size());
        const double q = smallest_quality(r.mesh);
        std::cout << "INFO " << refiner.name() << (jitter ? " jittered" : "") << " cycle " << cycle << ": " << marked.size() << " marked, "
                  << r.mesh.elements.size() << " tets, smallest quality " << q / initial_quality
                  << " of the initial";
        for (const auto& [name, value] : r.info.details)
            if (name == "red" || name == "green_families" || name == "reopened_green_families" || name == "passes")
                std::cout << ", " << name << " " << value;
        std::cout << "\n";
        CHECK(q > 0.2 * initial_quality);
        mesh = r.mesh;
        history = r.history;
    }
}

void test_red_green_history() {
    MeshProperties mesh = cube_mesh(2);
    const RedGreenRefinement refiner;
    const std::vector<uint32_t> first{0};
    const RefinementResult r1 = refiner.refine(mesh, first);
    CHECK(!r1.history.families.empty());
    // Mark one green tet: its family is reopened, its parent goes red.
    uint32_t green = 0;
    while (!r1.history.family[green]) ++green;
    const std::vector<uint32_t> second{green};
    const RefinementResult r2 = refiner.refine(r1.mesh, second, r1.history);
    CHECK(r2.info.details.at("reopened_green_families") >= 1);
    check_refined(r1.mesh, r2, second);
    // Nothing marked: the same mesh and history.
    const RefinementResult r3 = refiner.refine(r2.mesh, {}, r2.history);
    CHECK(r3.mesh.elements.size() == r2.mesh.elements.size());
    CHECK(r3.history.family == r2.history.family);
    CHECK(r3.info.refined_elements == 0);
    // Without the history the green tets are refined like any other.
    const RefinementResult r4 = refiner.refine(r1.mesh, second);
    CHECK(r4.info.details.at("reopened_green_families") == 0);
    check_refined(r1.mesh, r4, second);

    // Regression: reopened green parent split multiple times in one cycle must still report
    // the exact input member containing each final descendant's centroid.
    const std::vector<uint32_t> m1{0, 18};
    const RefinementResult g1 = refiner.refine(mesh, m1);
    check_refined(mesh, g1, m1);
    const std::vector<uint32_t> m2{88};
    const RefinementResult g2 = refiner.refine(g1.mesh, m2, g1.history);
    check_refined(g1.mesh, g2, m2);
}

void test_bad_input() {
    const MeshProperties mesh = cube_mesh(1);
    const std::vector<uint32_t> out_of_range{6};
    CHECK_THROWS(std::invalid_argument, LongestEdgeBisection().refine(mesh, out_of_range), "marked tet 6 is out of range");
    RefinementHistory wrong;
    wrong.family.resize(2);
    const std::vector<uint32_t> one{0};
    CHECK_THROWS(std::invalid_argument, RedGreenRefinement().refine(mesh, one, wrong), "the history has 2 entries for 6 tets");
    // Non-empty families with empty per-tet family vector must be rejected (no OOB midpoints).
    RefinementHistory orphan;
    orphan.families.push_back({{0, 1, 2, 3}, {999, {}, {}, {}, {}, {}}});
    CHECK_THROWS(std::invalid_argument, RedGreenRefinement().refine(mesh, one, orphan), "no per-tet family assignments");
    MeshProperties bad_mat = mesh;
    bad_mat.elements[0].material = 9;
    CHECK_THROWS(std::invalid_argument, RedGreenRefinement().refine(bad_mat, one), "tet 0 has material 9");
    MeshProperties bad_bc = mesh;
    bad_bc.boundary_elements[0].boundary_condition = 9;
    CHECK_THROWS(std::invalid_argument, LongestEdgeBisection().refine(bad_bc, one), "boundary element 0 has boundary condition 9");
    MeshProperties bad_bnode = mesh;
    bad_bnode.boundary_elements[0].node[0] = 99;
    CHECK_THROWS(std::invalid_argument, LongestEdgeBisection().refine(bad_bnode, one), "boundary element 0 has node 99");
    CHECK_THROWS(std::runtime_error, RedGreenRefinement({.max_new_nodes = 3}).refine(mesh, one), "more than 3 new nodes");
    CHECK_THROWS(std::invalid_argument, LongestEdgeBisection({.max_new_nodes = 0}), "max_new_nodes must be positive");
    // A history given to longest-edge bisection is ignored, with a warning.
    const RefinementResult rg = RedGreenRefinement().refine(mesh, one);
    const RefinementResult leb = LongestEdgeBisection().refine(rg.mesh, one, rg.history);
    CHECK(leb.info.warnings.size() == 1);
    CHECK(leb.history.family.empty());
}

int main() {
    test_single_tet();
    cycles(LongestEdgeBisection(), 14, false);
    cycles(RedGreenRefinement(), 6, false);
    cycles(LongestEdgeBisection(), 14, true);
    cycles(RedGreenRefinement(), 5, true);
    test_red_green_history();
    test_bad_input();
    return finish("refiner");
}
