// Tests for MeshLocator: cold BVH lookup, warm neighbour-walk + cavity fallback, exact
// sphere queries, box queries, and input validation.

#include "common/mesh_locator.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

Vec3 bary_point(const MeshProperties& mesh, uint32_t k, const std::array<double, 4>& l) {
    Vec3 p{0, 0, 0};
    for (int i = 0; i < 4; ++i)
        for (int d = 0; d < 3; ++d) p[d] += l[i] * mesh.nodes[mesh.elements[k].node[i]][d];
    return p;
}

double point_in_tet_min_bary(const MeshProperties& mesh, uint32_t k, const Vec3& x) {
    const auto& n = mesh.elements[k].node;
    const double whole = signed_volume(mesh, n);
    double smallest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        MeshProperties probe;
        probe.nodes = {mesh.nodes[n[0]], mesh.nodes[n[1]], mesh.nodes[n[2]], mesh.nodes[n[3]]};
        probe.nodes[i] = x;
        smallest = std::min(smallest, signed_volume(probe, {0, 1, 2, 3}) / whole);
    }
    return smallest;
}

// Rebuilds mesh.neighbours so locate_near can walk across faces.
void populate_neighbours(MeshProperties& mesh) {
    using Key = std::array<uint32_t, 3>;
    struct TetFace {
        Key      key;
        uint32_t element;
        uint8_t  local;
    };
    std::vector<TetFace> all;
    all.reserve(4 * mesh.elements.size());
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto& n = mesh.elements[k].node;
        for (uint8_t i = 0; i < 4; ++i) {
            Key key{};
            for (int j = 0, f = 0; j < 4; ++j)
                if (j != i) key[f++] = n[j];
            std::sort(key.begin(), key.end());
            all.push_back({key, k, i});
        }
    }
    std::sort(all.begin(), all.end(), [](const TetFace& a, const TetFace& b) { return a.key < b.key; });
    mesh.neighbours.assign(mesh.elements.size(), MeshProperties::no_neighbours);
    for (std::size_t i = 1; i < all.size(); ++i)
        if (all[i].key == all[i - 1].key) {
            mesh.neighbours[all[i].element][all[i].local] = all[i - 1].element;
            mesh.neighbours[all[i - 1].element][all[i - 1].local] = all[i].element;
        }
}

void test_locate_interior_boundary_and_outside() {
    MeshProperties mesh = cube_mesh(4);
    populate_neighbours(mesh);
    const MeshLocator locator(mesh);

    // Every tet's centroid must map back to that exact tet with barycentric = 1/4.
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const Vec3 c = bary_point(mesh, k, {0.25, 0.25, 0.25, 0.25});
        const auto loc = locator.locate(c);
        CHECK(loc.has_value() && loc->element == k);
        for (double b : loc->barycentric) CHECK_NEAR(b, 0.25, 1e-12);
    }

    // Vertices, edge midpoints, face centers: must find a valid containing tet.
    for (const Vec3& q : std::vector<Vec3>{{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}, {0.25, 0.0, 0.75}, {0.375, 0.625, 0.125}}) {
        const auto loc = locator.locate(q);
        CHECK(loc.has_value());
        CHECK(point_in_tet_min_bary(mesh, loc->element, q) >= -1e-9);
        const Vec3 reconstructed = bary_point(mesh, loc->element, loc->barycentric);
        for (int d = 0; d < 3; ++d) CHECK_NEAR(reconstructed[d], q[d], 1e-12);
    }

    // Outside points must return empty.
    for (const Vec3& out : std::vector<Vec3>{{-0.01, 0.5, 0.5}, {1.01, 0.5, 0.5}, {0.5, 0.5, -1.0}, {5, 5, 5}}) {
        CHECK(!locator.locate(out).has_value());
        CHECK(!locator.locate_near(out, 0).has_value());
    }

    // Warm locate_near from a distant hint must find the exact containing tet.
    const Vec3 target = bary_point(mesh, static_cast<uint32_t>(mesh.elements.size() - 1), {0.1, 0.2, 0.3, 0.4});
    const auto warm = locator.locate_near(target, 0);
    CHECK(warm.has_value() && warm->element == mesh.elements.size() - 1);
    CHECK_NEAR(warm->barycentric[0], 0.1, 1e-12);
    CHECK_NEAR(warm->barycentric[3], 0.4, 1e-12);

    // With every face marked no_neighbour, locate_near still succeeds via BVH fallback.
    MeshProperties no_nb = mesh;
    no_nb.neighbours.assign(mesh.elements.size(), MeshProperties::no_neighbours);
    const MeshLocator no_nb_locator(no_nb);
    const auto fallback = no_nb_locator.locate_near(target, 0);
    CHECK(fallback.has_value() && fallback->element == mesh.elements.size() - 1);
}

// Non-convex mesh with a hole carved out of the center: walking across the hole hits a
// boundary and must seamlessly fall back to the BVH.
void test_non_convex_cavity_fallback() {
    MeshProperties mesh = cube_mesh(5);
    std::erase_if(mesh.elements, [&](const MeshProperties::Element& e) {
        Vec3 c{0, 0, 0};
        for (uint32_t node : e.node)
            for (int d = 0; d < 3; ++d) c[d] += 0.25 * mesh.nodes[node][d];
        return c[0] > 0.2 && c[0] < 0.8 && c[1] > 0.2 && c[1] < 0.8 && c[2] > 0.2 && c[2] < 0.8;
    });
    populate_neighbours(mesh);
    const MeshLocator locator(mesh);

    // Inside the cavity: must return empty.
    CHECK(!locator.locate({0.5, 0.5, 0.5}).has_value());
    CHECK(!locator.locate_near({0.5, 0.5, 0.5}, 0).has_value());

    // Start on the x < 0.2 side, walk to the x > 0.8 side directly across the hole.
    const auto left = locator.locate({0.05, 0.5, 0.5});
    const auto right = locator.locate({0.95, 0.5, 0.5});
    CHECK(left.has_value() && right.has_value() && left->element != right->element);
    const auto crossed = locator.locate_near({0.95, 0.5, 0.5}, left->element);
    CHECK(crossed.has_value() && crossed->element == right->element);
}

void test_sphere_and_box_queries() {
    // Single unit tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    // The face (1,0,0)-(0,1,0)-(0,0,1) lies on x+y+z = 1; distance from (1,1,1) to it is 2/sqrt(3).
    MeshProperties single;
    single.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    single.elements = {{{0, 1, 2, 3}, 0}};
    single.materials = {{"one", 1.0, 0.0, false, {}}};
    single.neighbours = {MeshProperties::no_neighbours};
    const MeshLocator single_loc(single);

    const double exact_dist = 2.0 / std::sqrt(3.0);
    // Point (1,1,1) is inside the tet's AABB [0,1]^3, so AABB-only would falsely include it,
    // while exact sphere-tet intersection must exclude it for radius < 2/sqrt(3).
    CHECK(single_loc.query_sphere({1, 1, 1}, exact_dist - 1e-3).empty());
    CHECK(single_loc.query_sphere({1, 1, 1}, exact_dist + 1e-3) == (std::vector<uint32_t>{0}));
    CHECK(single_loc.query_sphere({0.1, 0.1, 0.1}, 0.0) == (std::vector<uint32_t>{0}));
    CHECK(single_loc.query_sphere({1, 1, 1}, 0.0).empty());

    // Larger mesh: compare query_box against brute-force AABB check, and verify sphere results
    // are a subset of the enclosing cube box and contain all tets whose vertices lie in the ball.
    const MeshProperties mesh = cube_mesh(4);
    const MeshLocator locator(mesh);
    const Vec3 c{0.42, 0.55, 0.31};
    const double r = 0.22;
    const Vec3 lo{c[0] - r, c[1] - r, c[2] - r}, hi{c[0] + r, c[1] + r, c[2] + r};
    const auto in_box = locator.query_box(lo, hi);
    const auto in_sphere = locator.query_sphere(c, r);
    CHECK(std::is_sorted(in_box.begin(), in_box.end()));
    CHECK(std::is_sorted(in_sphere.begin(), in_sphere.end()));

    // Brute-force box overlap check.
    std::vector<uint32_t> expected_box;
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        Vec3 tlo = mesh.nodes[mesh.elements[k].node[0]], thi = tlo;
        for (int i = 1; i < 4; ++i) {
            const Vec3& p = mesh.nodes[mesh.elements[k].node[i]];
            for (int d = 0; d < 3; ++d) {
                tlo[d] = std::min(tlo[d], p[d]);
                thi[d] = std::max(thi[d], p[d]);
            }
        }
        bool overlap = true;
        for (int d = 0; d < 3; ++d) overlap = overlap && !(thi[d] < lo[d] || hi[d] < tlo[d]);
        if (overlap) expected_box.push_back(k);
    }
    CHECK(in_box == expected_box);

    // Every sphere hit must lie in the enclosing box, and every tet with a vertex inside the
    // ball must be returned by query_sphere.
    for (uint32_t k : in_sphere) CHECK(std::binary_search(in_box.begin(), in_box.end(), k));
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        bool vertex_inside = false;
        for (uint32_t node : mesh.elements[k].node) {
            const Vec3& p = mesh.nodes[node];
            if (std::hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) <= r - 1e-9) vertex_inside = true;
        }
        if (vertex_inside) CHECK(std::binary_search(in_sphere.begin(), in_sphere.end(), k));
    }

    // Large radius covers the entire mesh.
    CHECK(locator.query_sphere({0.5, 0.5, 0.5}, 10.0).size() == mesh.elements.size());
}

void test_empty_and_bad_inputs() {
    const MeshProperties empty;
    const MeshLocator empty_loc(empty);
    CHECK(!empty_loc.locate({0, 0, 0}).has_value());
    CHECK(empty_loc.query_sphere({0, 0, 0}, 1.0).empty());
    CHECK(empty_loc.query_box({0, 0, 0}, {1, 1, 1}).empty());
    CHECK_THROWS(std::invalid_argument, empty_loc.locate_near({0, 0, 0}, 0), "out of range");

    const MeshProperties mesh = cube_mesh(1);
    const MeshLocator loc(mesh);
    CHECK_THROWS(std::invalid_argument, loc.locate({std::nan(""), 0, 0}), "not finite");
    CHECK_THROWS(std::invalid_argument, loc.locate_near({0.5, 0.5, 0.5}, 99), "out of range");
    CHECK_THROWS(std::invalid_argument, loc.query_sphere({0, 0, 0}, -1.0), "non-negative");
    CHECK_THROWS(std::invalid_argument, loc.query_sphere({std::nan(""), 0, 0}, 1.0), "not finite");
    CHECK_THROWS(std::invalid_argument, loc.query_box({1, 0, 0}, {0, 1, 1}), "must not exceed hi");
    CHECK_THROWS(std::invalid_argument, loc.query_box({std::nan(""), 0, 0}, {1, 1, 1}), "must be finite");

    MeshProperties bad_node = mesh;
    bad_node.elements[0].node[0] = 99;
    CHECK_THROWS(std::invalid_argument, MeshLocator{bad_node}, "tet 0 has node 99");
    MeshProperties bad_vol = mesh;
    std::swap(bad_vol.elements[0].node[0], bad_vol.elements[0].node[1]);
    CHECK_THROWS(std::invalid_argument, MeshLocator{bad_vol}, "tet 0 has volume");
    MeshProperties bad_nb = mesh;
    bad_nb.neighbours.assign(mesh.elements.size(), MeshProperties::no_neighbours);
    bad_nb.neighbours[0][0] = 99;
    CHECK_THROWS(std::invalid_argument, MeshLocator{bad_nb}, "names neighbour 99");
    MeshProperties missing_nb = mesh;
    missing_nb.neighbours.clear();
    CHECK_THROWS(std::invalid_argument, MeshLocator{missing_nb}, "neighbours has 0 entries");
}

} // namespace

int main() {
    test_locate_interior_boundary_and_outside();
    test_non_convex_cavity_fallback();
    test_sphere_and_box_queries();
    test_empty_and_bad_inputs();
    return finish("locator");
}
