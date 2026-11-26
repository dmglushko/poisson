// Tests for MeshLocator and the standalone walks: cold BVH lookup, warm neighbour-walk +
// cavity fallback, walk_to_point / walk_along_segment statuses, exact sphere queries, box
// queries, and input validation.

#include "common/mesh_locator.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
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

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

using PointStatus = PointWalkResult::Status;
using SegmentStatus = SegmentWalkResult::Status;

Vec3 centroid(const MeshProperties& mesh, uint32_t k) { return bary_point(mesh, k, {0.25, 0.25, 0.25, 0.25}); }

// Removes every tet whose centroid satisfies `drop`, then rebuilds neighbours.
template <typename Pred>
MeshProperties carve(MeshProperties mesh, Pred drop) {
    std::erase_if(mesh.elements, [&](const MeshProperties::Element& e) {
        Vec3 c{0, 0, 0};
        for (uint32_t node : e.node)
            for (int d = 0; d < 3; ++d) c[d] += 0.25 * mesh.nodes[node][d];
        return drop(c);
    });
    populate_neighbours(mesh);
    return mesh;
}

double min4(const std::array<double, 4>& v) { return std::min({v[0], v[1], v[2], v[3]}); }

// Every barycentric coordinate is finite and they sum to 1.
bool sane(const std::array<double, 4>& l) {
    for (double b : l)
        if (!std::isfinite(b)) return false;
    return std::abs(l[0] + l[1] + l[2] + l[3] - 1.0) <= 1e-12;
}

// Whether all three nodes of the face opposite local node `face` of tet k have coordinate d == value.
bool face_on_plane(const MeshProperties& mesh, uint32_t k, int face, int d, double value) {
    for (int i = 0; i < 4; ++i)
        if (i != face && mesh.nodes[mesh.elements[k].node[i]][d] != value) return false;
    return true;
}

// Local node orderings vary per tet (all 12 even permutations, so orientation stays
// positive): barycentric[i] must still belong to node i, and centroids give 1/4.
void test_barycentric_with_permuted_local_order() {
    MeshProperties mesh = cube_mesh(3);
    std::vector<std::array<int, 4>> even;
    std::array<int, 4> perm{0, 1, 2, 3};
    do {
        int inversions = 0;
        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b) inversions += perm[a] > perm[b];
        if (inversions % 2 == 0) even.push_back(perm);
    } while (std::next_permutation(perm.begin(), perm.end()));
    CHECK(even.size() == 12);
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto old = mesh.elements[k].node;
        const auto& p = even[(k * 5) % even.size()];
        for (int i = 0; i < 4; ++i) mesh.elements[k].node[i] = old[p[i]];
        CHECK(signed_volume(mesh, mesh.elements[k].node) > 0);
    }
    populate_neighbours(mesh);
    const MeshLocator locator(mesh);

    const uint32_t far = static_cast<uint32_t>(mesh.elements.size() - 1);
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto loc = locator.locate(centroid(mesh, k));
        CHECK(loc.has_value() && loc->element == k);
        if (loc)
            for (double b : loc->barycentric) CHECK_NEAR(b, 0.25, 1e-12);

        const auto walk = walk_to_point(mesh, centroid(mesh, k), far);
        CHECK(walk.status == PointStatus::found && walk.element == k);
        for (double b : walk.barycentric) CHECK_NEAR(b, 0.25, 1e-12);

        const std::array<double, 4> l{0.1, 0.2, 0.3, 0.4};
        const auto asym = locator.walk_to_point(bary_point(mesh, k, l), k);
        CHECK(asym.status == PointStatus::found && asym.element == k && asym.steps == 0);
        for (int i = 0; i < 4; ++i) CHECK_NEAR(asym.barycentric[i], l[i], 1e-12);
    }

    // A point on a shared face (its vertex average) is found at once from either side.
    for (uint32_t k = 0; k < mesh.elements.size(); ++k)
        for (int f = 0; f < 4; ++f) {
            const uint32_t other = mesh.neighbours[k][f];
            if (other == MeshProperties::no_neighbour) continue;
            std::array<double, 4> l{1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3};
            l[f] = 0.0;
            const Vec3 q = bary_point(mesh, k, l);
            const auto a = walk_to_point(mesh, q, k), b = walk_to_point(mesh, q, other);
            CHECK(a.status == PointStatus::found && a.element == k && a.steps == 0);
            CHECK(b.status == PointStatus::found && b.element == other && b.steps == 0);
        }
}

void test_walk_to_point() {
    const MeshProperties mesh = cube_mesh(4);
    const MeshLocator locator(mesh);
    const uint32_t last = static_cast<uint32_t>(mesh.elements.size() - 1);
    const Vec3 target = bary_point(mesh, last, {0.1, 0.2, 0.3, 0.4});

    // found, via both the free function and the method.
    for (const PointWalkResult& r : {walk_to_point(mesh, target, 0), locator.walk_to_point(target, 0)}) {
        CHECK(r.status == PointStatus::found && r.element == last);
        CHECK(r.steps > 0 && r.steps <= MeshLocator::default_max_walk_steps);
        CHECK_NEAR(r.barycentric[0], 0.1, 1e-12);
        CHECK_NEAR(r.barycentric[3], 0.4, 1e-12);
    }

    // Already inside: no hops.
    const auto inside = walk_to_point(mesh, target, last);
    CHECK(inside.status == PointStatus::found && inside.element == last && inside.steps == 0);

    // Outside a convex mesh: exited through the most negative boundary face.
    for (const Vec3& out : std::vector<Vec3>{{1.5, 0.43, 0.61}, {-0.2, 0.5, 0.5}, {0.3, 0.7, 2.0}}) {
        const auto r = walk_to_point(mesh, out, 0);
        CHECK(r.status == PointStatus::exited);
        CHECK(r.exit_face < 4 && mesh.neighbours[r.element][r.exit_face] == MeshProperties::no_neighbour);
        CHECK(sane(r.barycentric));
        CHECK(r.barycentric[r.exit_face] < 0.0);
        for (int i = 0; i < 4; ++i)
            if (r.barycentric[i] < r.barycentric[r.exit_face])
                CHECK(mesh.neighbours[r.element][i] != MeshProperties::no_neighbour);
        CHECK(!locator.locate_near(out, 0).has_value());
    }
    // Starting in a boundary tet whose boundary face x lies beyond: exits with 0 hops.
    {
        const Vec3 on_face{1.0, 0.43, 0.61};
        const auto start = locator.locate(on_face);
        CHECK(start.has_value());
        const auto r = walk_to_point(mesh, {1.2, 0.43, 0.61}, start->element);
        CHECK(r.status == PointStatus::exited && r.steps == 0 && r.element == start->element);
        CHECK(face_on_plane(mesh, r.element, r.exit_face, 0, 1.0));
    }

    // step_limit: one hop allowed, target far away.
    const MeshLocator one_step(mesh, MeshLocator::Settings{1});
    for (const PointWalkResult& r : {walk_to_point(mesh, target, 0, 1), one_step.walk_to_point(target, 0)}) {
        CHECK(r.status == PointStatus::step_limit && r.steps == 1);
        CHECK(r.element == mesh.neighbours[0][0] || r.element == mesh.neighbours[0][1] ||
              r.element == mesh.neighbours[0][2] || r.element == mesh.neighbours[0][3]);
        CHECK(sane(r.barycentric) && min4(r.barycentric) < 0.0);
    }
    // One hop is enough for a target in a neighbour.
    for (int f = 0; f < 4; ++f) {
        const uint32_t nb = mesh.neighbours[0][f];
        if (nb == MeshProperties::no_neighbour) continue;
        const auto r = one_step.walk_to_point(centroid(mesh, nb), 0);
        CHECK(r.status == PointStatus::found && r.element == nb && r.steps == 1);
    }
    // locate_near falls back to the BVH when the step budget runs out.
    const auto fallback = one_step.locate_near(target, 0);
    CHECK(fallback.has_value() && fallback->element == last);

    // Long walks go past the 16 steepest-descent steps into the remembering walk.
    const MeshProperties big = cube_mesh(12);
    const uint32_t big_last = static_cast<uint32_t>(big.elements.size() - 1);
    const auto long_walk = walk_to_point(big, centroid(big, big_last), 0, 1000);
    CHECK(long_walk.status == PointStatus::found && long_walk.element == big_last && long_walk.steps > 16);

    CHECK_THROWS(std::invalid_argument, (MeshLocator{mesh, MeshLocator::Settings{0}}), "max_walk_steps");
    CHECK_THROWS(std::invalid_argument, walk_to_point(mesh, target, 0, 0), "max_steps");
}

// L-shaped mesh: the straight line from one arm to the other crosses the notch, so
// steepest descent alone would run into the notch wall; interior-first walks around it.
void test_walk_to_point_non_convex() {
    const MeshProperties mesh = carve(cube_mesh(4), [](const Vec3& c) { return c[0] > 0.5 && c[1] > 0.5; });
    const MeshLocator locator(mesh);
    const Vec3 from{0.41, 0.93, 0.47}, to{0.93, 0.07, 0.44};
    const auto start = locator.locate(from);
    const auto goal = locator.locate(to);
    CHECK(start.has_value() && goal.has_value());
    CHECK(!locator.locate({0.75, 0.75, 0.5}).has_value());   // the notch

    const auto r = walk_to_point(mesh, to, start->element);
    CHECK(r.status == PointStatus::found && r.element == goal->element && r.steps > 0);
    const auto m = locator.walk_to_point(to, start->element);
    CHECK(m.status == r.status && m.element == r.element && m.steps == r.steps);

    // Points in the notch are outside: the walk exits, locate_near and locate agree.
    const auto notch = walk_to_point(mesh, {0.8, 0.8, 0.5}, start->element);
    CHECK(notch.status == PointStatus::exited);
    CHECK(mesh.neighbours[notch.element][notch.exit_face] == MeshProperties::no_neighbour);
    CHECK(!locator.locate_near({0.8, 0.8, 0.5}, start->element).has_value());
}

void test_walk_along_segment() {
    const MeshProperties mesh = cube_mesh(4);
    const MeshLocator locator(mesh);
    const uint32_t last = static_cast<uint32_t>(mesh.elements.size() - 1);

    // Interior: found with t == 1 and barycentric at x1.
    const uint32_t a = 21;
    const uint32_t b = 200;
    const Vec3 xa = centroid(mesh, a), xb = centroid(mesh, b);
    for (const SegmentWalkResult& r : {walk_along_segment(mesh, xa, xb, a), locator.walk_along_segment(xa, xb, a)}) {
        CHECK(r.status == SegmentStatus::found && r.element == b && r.t == 1.0 && r.steps > 0);
        for (double l : r.barycentric) CHECK_NEAR(l, 0.25, 1e-12);
    }

    // Exit through the x = 1 face: exact face, t, and barycentric at p(t).
    {
        const Vec3 x0{0.62, 0.43, 0.61}, x1{1.5, 0.43, 0.61};
        const auto start = locator.locate(x0);
        CHECK(start.has_value());
        const auto r = walk_along_segment(mesh, x0, x1, start->element);
        CHECK(r.status == SegmentStatus::exited);
        CHECK(mesh.neighbours[r.element][r.exit_face] == MeshProperties::no_neighbour);
        CHECK(face_on_plane(mesh, r.element, r.exit_face, 0, 1.0));
        CHECK_NEAR(r.t, (1.0 - 0.62) / (1.5 - 0.62), 1e-12);
        CHECK(sane(r.barycentric));
        CHECK_NEAR(r.barycentric[r.exit_face], 0.0, 1e-12);
        CHECK(min4(r.barycentric) >= -1e-12);
        const Vec3 p = bary_point(mesh, r.element, r.barycentric);
        CHECK_NEAR(p[0], 1.0, 1e-12);
        CHECK_NEAR(p[1], 0.43, 1e-12);
        CHECK_NEAR(p[2], 0.61, 1e-12);
    }

    // x0 on a boundary face: inward is found, outward exits at t = 0.
    {
        const Vec3 x0{1.0, 0.43, 0.61};
        const auto start = locator.locate(x0);
        CHECK(start.has_value());
        const Vec3 in{0.83, 0.43, 0.61};
        const auto inward = walk_along_segment(mesh, x0, in, start->element);
        CHECK(inward.status == SegmentStatus::found && inward.element == locator.locate(in)->element);
        const auto outward = walk_along_segment(mesh, x0, {1.3, 0.43, 0.61}, start->element);
        CHECK(outward.status == SegmentStatus::exited && outward.steps == 0 && outward.element == start->element);
        CHECK(outward.t >= 0.0 && outward.t <= 1e-12);
        CHECK(face_on_plane(mesh, outward.element, outward.exit_face, 0, 1.0));

        // RK drift: x0 slightly outside the boundary face of start.
        const Vec3 drift{1.0 + 1e-10, 0.43, 0.61};
        const auto back = walk_along_segment(mesh, drift, in, start->element);
        CHECK(back.status == SegmentStatus::found && back.element == inward.element);
        const auto away = walk_along_segment(mesh, drift, {1.3, 0.43, 0.61}, start->element);
        CHECK(away.status == SegmentStatus::exited && away.t == 0.0 && away.steps == 0);
    }

    // x0 slightly outside an interior face of start (by 1e-10 along the face normal).
    for (int f = 0; f < 4; ++f) {
        const uint32_t nb = mesh.neighbours[a][f];
        if (nb == MeshProperties::no_neighbour) continue;
        std::array<double, 4> l{1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3};
        l[f] = 0.0;
        const Vec3 on = bary_point(mesh, a, l);
        const Vec3 inner = centroid(mesh, a);
        const Vec3 dir = sub(on, inner);
        const double len = std::hypot(dir[0], dir[1], dir[2]);
        const Vec3 x0{on[0] + 1e-10 * dir[0] / len, on[1] + 1e-10 * dir[1] / len, on[2] + 1e-10 * dir[2] / len};
        const auto stay = walk_along_segment(mesh, x0, inner, a);
        CHECK(stay.status == SegmentStatus::found && stay.element == a && stay.steps == 0);
        const auto cross_over = walk_along_segment(mesh, x0, centroid(mesh, nb), a);
        CHECK(cross_over.status == SegmentStatus::found && cross_over.element == nb && cross_over.steps == 1);
    }

    // x0 == x1: inside start, elsewhere in the mesh, and outside the mesh.
    {
        const auto same = walk_along_segment(mesh, xa, xa, a);
        CHECK(same.status == SegmentStatus::found && same.element == a && same.steps == 0 && same.t == 1.0);
        const auto moved = walk_along_segment(mesh, xb, xb, a);
        CHECK(moved.status == SegmentStatus::found && moved.element == b && moved.t == 1.0);
        CHECK(sane(moved.barycentric));
        const Vec3 out{1.4, 0.43, 0.61};
        const auto gone = walk_along_segment(mesh, out, out, a);
        CHECK(gone.status == SegmentStatus::exited && gone.t == 0.0 && sane(gone.barycentric));
        CHECK(mesh.neighbours[gone.element][gone.exit_face] == MeshProperties::no_neighbour);
    }

    // step_limit: stops after max_steps hops with t at the last crossing.
    {
        const Vec3 x0 = centroid(mesh, 0), x1 = centroid(mesh, last);
        const MeshLocator two_steps(mesh, MeshLocator::Settings{2});
        for (const SegmentWalkResult& r :
             {walk_along_segment(mesh, x0, x1, 0, 2), two_steps.walk_along_segment(x0, x1, 0)}) {
            CHECK(r.status == SegmentStatus::step_limit && r.steps == 2);
            CHECK(r.t > 0.0 && r.t < 1.0 && sane(r.barycentric));
            CHECK(min4(r.barycentric) >= -1e-9);
            const Vec3 p = bary_point(mesh, r.element, r.barycentric);
            for (int d = 0; d < 3; ++d) CHECK_NEAR(p[d], x0[d] + r.t * (x1[d] - x0[d]), 1e-12);
        }
        const auto full = walk_along_segment(mesh, x0, x1, 0);
        CHECK(full.status == SegmentStatus::found && full.element == last);
    }

    // Axis-aligned segments grazing mesh edges (y, z on grid lines) and lying in a mesh
    // face plane (z on a grid line).
    for (const auto& [x0, x1] : std::vector<std::pair<Vec3, Vec3>>{{{0.1, 0.25, 0.5}, {0.9, 0.25, 0.5}},
                                                                   {{0.1, 0.3, 0.5}, {0.9, 0.3, 0.5}},
                                                                   {{0.6, 0.1, 0.75}, {0.6, 0.95, 0.75}},
                                                                   {{0.1, 0.1, 0.1}, {0.9, 0.9, 0.9}}}) {
        const auto start = locator.locate(x0);
        CHECK(start.has_value());
        const auto r = walk_along_segment(mesh, x0, x1, start->element);
        CHECK(r.status == SegmentStatus::found && r.t == 1.0);
        CHECK(point_in_tet_min_bary(mesh, r.element, x1) >= -1e-9);
        CHECK(sane(r.barycentric));
    }
}

// U-shaped mesh: the segment between the two arms crosses the cavity, so the segment walk
// must report the cavity wall even though x1 is back inside the mesh; locate_near still
// finds x1 through the BVH.
void test_walk_along_segment_cavity() {
    const MeshProperties mesh =
        carve(cube_mesh(4), [](const Vec3& c) { return c[0] > 0.25 && c[0] < 0.75 && c[1] > 0.25; });
    const MeshLocator locator(mesh);
    const Vec3 x0{0.1, 0.61, 0.37}, x1{0.9, 0.61, 0.37};
    const auto start = locator.locate(x0);
    const auto goal = locator.locate(x1);
    CHECK(start.has_value() && goal.has_value());

    for (const SegmentWalkResult& r :
         {walk_along_segment(mesh, x0, x1, start->element), locator.walk_along_segment(x0, x1, start->element)}) {
        CHECK(r.status == SegmentStatus::exited);
        CHECK(mesh.neighbours[r.element][r.exit_face] == MeshProperties::no_neighbour);
        CHECK(face_on_plane(mesh, r.element, r.exit_face, 0, 0.25));
        CHECK_NEAR(r.t, (0.25 - 0.1) / 0.8, 1e-12);
        const Vec3 p = bary_point(mesh, r.element, r.barycentric);
        CHECK_NEAR(p[0], 0.25, 1e-12);
        CHECK_NEAR(p[1], 0.61, 1e-12);
        CHECK_NEAR(p[2], 0.37, 1e-12);
    }
    const auto near = locator.locate_near(x1, start->element);
    CHECK(near.has_value() && near->element == goal->element);
}

void test_walk_bad_inputs() {
    const MeshProperties mesh = cube_mesh(2);
    const MeshLocator locator(mesh);
    const Vec3 nan{std::nan(""), 0, 0}, x{0.5, 0.5, 0.5};

    CHECK_THROWS(std::invalid_argument, walk_to_point(mesh, nan, 0), "not finite");
    CHECK_THROWS(std::invalid_argument, walk_to_point(mesh, x, 999), "out of range");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(mesh, x, nan, 0), "not finite");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(mesh, nan, x, 0), "not finite");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(mesh, x, x, 999), "out of range");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(mesh, x, x, 0, 0), "max_steps");
    CHECK_THROWS(std::invalid_argument, locator.walk_to_point(nan, 0), "not finite");
    CHECK_THROWS(std::invalid_argument, locator.walk_to_point(x, 999), "out of range");
    CHECK_THROWS(std::invalid_argument, locator.walk_along_segment(x, nan, 0), "not finite");
    CHECK_THROWS(std::invalid_argument, locator.walk_along_segment(x, x, 999), "out of range");

    const MeshProperties empty;
    CHECK_THROWS(std::invalid_argument, walk_to_point(empty, x, 0), "out of range");

    MeshProperties missing_nb = mesh;
    missing_nb.neighbours.pop_back();
    CHECK_THROWS(std::invalid_argument, walk_to_point(missing_nb, x, 0), "neighbours has");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(missing_nb, x, x, 0), "neighbours has");

    // Per-visited-tet checks of the free functions.
    MeshProperties bad_node = mesh;
    bad_node.elements[0].node[2] = 9999;
    CHECK_THROWS(std::invalid_argument, walk_to_point(bad_node, x, 0), "tet 0 has node 9999");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(bad_node, x, x, 0), "tet 0 has node 9999");
    MeshProperties bad_coord = mesh;
    bad_coord.nodes[bad_coord.elements[0].node[1]][2] = std::numeric_limits<double>::infinity();
    CHECK_THROWS(std::invalid_argument, walk_to_point(bad_coord, x, 0), "non-finite coordinates");
    MeshProperties bad_vol = mesh;
    std::swap(bad_vol.elements[0].node[0], bad_vol.elements[0].node[1]);
    CHECK_THROWS(std::invalid_argument, walk_to_point(bad_vol, x, 0), "tet 0 has volume");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(bad_vol, x, x, 0), "tet 0 has volume");
    MeshProperties bad_nb = mesh;
    bad_nb.neighbours[0] = {9999, 9999, 9999, 9999};
    CHECK_THROWS(std::invalid_argument, walk_to_point(bad_nb, x, 0), "names neighbour 9999");
    CHECK_THROWS(std::invalid_argument, walk_along_segment(bad_nb, x, x, 0), "names neighbour 9999");
    // A malformed tet further along the walk is caught when visited.
    const uint32_t last = static_cast<uint32_t>(mesh.elements.size() - 1);
    MeshProperties bad_far = mesh;
    std::swap(bad_far.elements[last].node[0], bad_far.elements[last].node[1]);
    CHECK_THROWS(std::invalid_argument, walk_to_point(bad_far, centroid(mesh, last), 0),
                 "tet " + std::to_string(last) + " has volume");
}

// Triangle distance written independently of the locator (plane projection, else edges).
double brute_dist2_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const auto seg = [&](const Vec3& u, const Vec3& v) {
        const Vec3 uv = sub(v, u), up = sub(p, u);
        const double t = std::clamp(dot(up, uv) / dot(uv, uv), 0.0, 1.0);
        const Vec3 q{u[0] + t * uv[0], u[1] + t * uv[1], u[2] + t * uv[2]};
        return dot(sub(p, q), sub(p, q));
    };
    const Vec3 n = cross(sub(b, a), sub(c, a));
    const double h = dot(sub(p, a), n) / dot(n, n);
    const Vec3 q{p[0] - h * n[0], p[1] - h * n[1], p[2] - h * n[2]};
    const double wa = dot(cross(sub(b, q), sub(c, q)), n), wb = dot(cross(sub(c, q), sub(a, q)), n),
                 wc = dot(cross(sub(a, q), sub(b, q)), n);
    if (wa >= 0 && wb >= 0 && wc >= 0) return h * h * dot(n, n);
    return std::min({seg(a, b), seg(b, c), seg(c, a)});
}

double brute_dist_tet(const MeshProperties& mesh, uint32_t k, const Vec3& p) {
    if (point_in_tet_min_bary(mesh, k, p) >= 0) return 0.0;
    const auto& n = mesh.elements[k].node;
    const Vec3 &p0 = mesh.nodes[n[0]], &p1 = mesh.nodes[n[1]], &p2 = mesh.nodes[n[2]], &p3 = mesh.nodes[n[3]];
    return std::sqrt(std::min({brute_dist2_triangle(p, p0, p1, p2), brute_dist2_triangle(p, p0, p1, p3),
                               brute_dist2_triangle(p, p0, p2, p3), brute_dist2_triangle(p, p1, p2, p3)}));
}

// Range queries against brute force on a shuffled-order mesh, with radii large enough to
// hit the subtree fast-accept and small enough to need the exact per-tet tests.
void test_range_queries_brute_force() {
    MeshProperties mesh = cube_mesh(8);
    std::mt19937 rng(12345);
    std::shuffle(mesh.elements.begin(), mesh.elements.end(), rng);
    populate_neighbours(mesh);
    const MeshLocator locator(mesh);
    const uint32_t n = static_cast<uint32_t>(mesh.elements.size());

    std::uniform_real_distribution<double> coord(-0.2, 1.2), rad(0.0, 0.6);
    std::vector<uint32_t> buffer(100, 7);   // non-empty, reused across calls
    for (int q = 0; q < 40; ++q) {
        const Vec3 c{coord(rng), coord(rng), coord(rng)};
        const double r = q == 0 ? 0.0 : (q == 1 ? 5.0 : rad(rng));
        const auto sorted = locator.query_sphere(c, r);
        CHECK(std::is_sorted(sorted.begin(), sorted.end()));
        CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());

        const std::size_t capacity = buffer.capacity();
        locator.query_sphere(c, r, buffer);
        CHECK(buffer.capacity() >= capacity);
        std::vector<uint32_t> unsorted = buffer;
        std::sort(unsorted.begin(), unsorted.end());
        CHECK(unsorted == sorted);

        for (uint32_t k = 0; k < n; ++k) {
            const double d = brute_dist_tet(mesh, k, c);
            const bool in = std::binary_search(sorted.begin(), sorted.end(), k);
            if (d < r - 1e-9) CHECK(in);
            if (d > r + 1e-9) CHECK(!in);
        }
        if (q == 1) CHECK(sorted.size() == n);
    }

    std::uniform_real_distribution<double> ext(0.0, 0.7);
    for (int q = 0; q < 40; ++q) {
        Vec3 lo{coord(rng), coord(rng), coord(rng)}, hi = lo;
        if (q == 0) {
            lo = {-1, -1, -1};
            hi = {2, 2, 2};
        } else if (q > 1)
            for (int d = 0; d < 3; ++d) hi[d] += ext(rng);   // q == 1: degenerate point box
        const auto got = locator.query_box(lo, hi);
        std::vector<uint32_t> expected;
        for (uint32_t k = 0; k < n; ++k) {
            Vec3 tlo = mesh.nodes[mesh.elements[k].node[0]], thi = tlo;
            for (int i = 1; i < 4; ++i)
                for (int d = 0; d < 3; ++d) {
                    tlo[d] = std::min(tlo[d], mesh.nodes[mesh.elements[k].node[i]][d]);
                    thi[d] = std::max(thi[d], mesh.nodes[mesh.elements[k].node[i]][d]);
                }
            bool overlap = true;
            for (int d = 0; d < 3; ++d) overlap = overlap && !(thi[d] < lo[d] || hi[d] < tlo[d]);
            if (overlap) expected.push_back(k);
        }
        CHECK(got == expected);
        if (q == 0) CHECK(got.size() == n);
    }

    // Cold and warm lookups on the shuffled mesh.
    for (uint32_t k = 0; k < n; k += 37) {
        const auto loc = locator.locate(centroid(mesh, k));
        CHECK(loc.has_value() && loc->element == k);
        const auto near = locator.locate_near(centroid(mesh, k), (k * 7919) % n);
        CHECK(near.has_value() && near->element == k);
    }
}

// Rigid transform helper (arbitrary rotation + translation) for non-axis-aligned tests.
Vec3 rigid_transform(const Vec3& p, const Vec3& offset) {
    // Orthonormal rotation matrix (angle ~37 deg around unit axis (1, 2, 3)/sqrt(14)).
    const double ax = 1.0 / std::sqrt(14.0), ay = 2.0 / std::sqrt(14.0), az = 3.0 / std::sqrt(14.0);
    const double c = std::cos(0.6457718232379019), s = std::sin(0.6457718232379019), t = 1.0 - c;
    const double r00 = t * ax * ax + c,      r01 = t * ax * ay - s * az, r02 = t * ax * az + s * ay;
    const double r10 = t * ax * ay + s * az, r11 = t * ay * ay + c,      r12 = t * ay * az - s * ax;
    const double r20 = t * ax * az - s * ay, r21 = t * ay * az + s * ax, r22 = t * az * az + c;
    return {r00 * p[0] + r01 * p[1] + r02 * p[2] + offset[0],
            r10 * p[0] + r11 * p[1] + r12 * p[2] + offset[1],
            r20 * p[0] + r21 * p[1] + r22 * p[2] + offset[2]};
}

MeshProperties transformed_mesh(MeshProperties mesh, double scale, bool rotate, const Vec3& offset) {
    for (Vec3& p : mesh.nodes) {
        p = {p[0] * scale, p[1] * scale, p[2] * scale};
        if (rotate) p = rigid_transform(p, offset);
        else p = {p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]};
    }
    return mesh;
}

// Builds a minimal 3-tet L-shaped prism around the reflex edge (0,0,0)-(0,0,1):
//   tet 0: x >= 0, y <= 0 (boundary wall x = 0, interior face y = 0 to tet 1)
//   tet 1: x >= 0, y >= 0 (interior faces y = 0 to tet 0 and x = 0 to tet 2)
//   tet 2: x <= 0, y >= 0 (boundary wall y = 0, interior face x = 0 to tet 1)
MeshProperties reflex_edge_mesh() {
    MeshProperties m;
    m.nodes = {
        {0, 0, 0},   // 0: reflex edge bottom
        {0, 0, 1},   // 1: reflex edge top
        {0, -1, 0},  // 2
        {1, 0, 0},   // 3
        {0, 1, 0},   // 4
        {-1, 0, 0},  // 5
    };
    auto add_tet = [&](std::array<uint32_t, 4> n) {
        if (signed_volume(m, n) < 0) std::swap(n[2], n[3]);
        m.elements.push_back({n, 0});
    };
    add_tet({0, 1, 2, 3});   // tet 0: x >= 0, y <= 0
    add_tet({0, 1, 3, 4});   // tet 1: x >= 0, y >= 0
    add_tet({0, 1, 4, 5});   // tet 2: x <= 0, y >= 0
    m.materials = {{"one", 1.0, 0.0, false, {}}};
    populate_neighbours(m);
    return m;
}

void test_edge_grazes_and_wall_tolerance() {
    // 1. Reflex-edge graze: exact axis-aligned and rotated/offset near-tie versions.
    {
        const MeshProperties base = reflex_edge_mesh();
        // Segment goes from inside tet 0 through the reflex edge (0, 0, 0.25) into tet 2.
        const Vec3 g0{0.2, -0.2, 0.25}, g1{-0.2, 0.2, 0.25};
        const auto r_exact = walk_along_segment(base, g0, g1, 0);
        CHECK(r_exact.status == SegmentStatus::found && r_exact.element == 2 && r_exact.t == 1.0);
        CHECK(sane(r_exact.barycentric));

        // Same geometry rotated by an irrational angle and translated off-origin so the
        // boundary and interior face crossing times differ by ULPs.
        const Vec3 offset{1.7, -3.2, 0.9};
        const MeshProperties rot = transformed_mesh(base, 1.0, true, offset);
        const Vec3 rg0 = rigid_transform(g0, offset), rg1 = rigid_transform(g1, offset);
        const auto r_rot = walk_along_segment(rot, rg0, rg1, 0);
        CHECK(r_rot.status == SegmentStatus::found && r_rot.element == 2 && r_rot.t == 1.0);
        CHECK(sane(r_rot.barycentric));

        // Also test max_steps = 1: redirecting at the reflex edge on hop 1 hits step_limit,
        // while an actual boundary exit on hop 1 returns exited.
        const auto r_lim = walk_along_segment(base, g0, g1, 0, 1);
        CHECK(r_lim.status == SegmentStatus::step_limit && r_lim.steps == 1);
        const Vec3 exit_after_1{0.3, 0.6, 0.25};   // crosses y = 0 into tet 1, then exits tet 1's outer face
        const auto r_exit1 = walk_along_segment(base, g0, exit_after_1, 0, 1);
        CHECK(r_exit1.status == SegmentStatus::exited && r_exit1.steps == 1 && r_exit1.element == 1);
    }

    // 2. Convex boundary-edge graze (axis-aligned and rotated): exits cleanly at the right t.
    {
        const MeshProperties cube = cube_mesh(2);
        const MeshLocator loc(cube);
        const Vec3 x0{0.5, 0.5, 0.37}, x1{1.5, 1.5, 0.37};   // hits convex edge (1, 1, 0.37) at t = 0.5
        const auto start = loc.locate(x0);
        CHECK(start.has_value());
        const auto r = walk_along_segment(cube, x0, x1, start->element);
        CHECK(r.status == SegmentStatus::exited);
        CHECK(cube.neighbours[r.element][r.exit_face] == MeshProperties::no_neighbour);
        CHECK_NEAR(r.t, 0.5, 1e-12);
        const Vec3 p = bary_point(cube, r.element, r.barycentric);
        CHECK_NEAR(p[0], 1.0, 1e-12);
        CHECK_NEAR(p[1], 1.0, 1e-12);
        CHECK_NEAR(p[2], 0.37, 1e-12);

        const Vec3 offset{-2.3, 4.1, 1.5};
        const MeshProperties rcube = transformed_mesh(cube, 1.0, true, offset);
        const MeshLocator rloc(rcube);
        const Vec3 rx0 = rigid_transform(x0, offset), rx1 = rigid_transform(x1, offset);
        const auto rstart = rloc.locate(rx0);
        CHECK(rstart.has_value());
        const auto rr = walk_along_segment(rcube, rx0, rx1, rstart->element);
        CHECK(rr.status == SegmentStatus::exited);
        CHECK(rcube.neighbours[rr.element][rr.exit_face] == MeshProperties::no_neighbour);
        CHECK_NEAR(rr.t, 0.5, 1e-12);
    }

    // 3. Endpoint grazing tolerance (deliberately kept to avoid false t = 0 exits on wall-sliding rays).
    {
        MeshProperties single;
        single.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        single.elements = {{{0, 1, 2, 3}, 0}};
        single.materials = {{"one", 1.0, 0.0, false, {}}};
        single.neighbours = {MeshProperties::no_neighbours};
        // x1 is only 5e-10 (< bary_tol) beyond x = 0, so the segment exits y = 0 at t = 0.5.
        const auto r = walk_along_segment(single, {0.0, 0.25, 0.25}, {-5e-10, -0.25, 0.25}, 0);
        CHECK(r.status == SegmentStatus::exited && r.steps == 0);
        CHECK(face_on_plane(single, r.element, r.exit_face, 1, 0.0));
        CHECK_NEAR(r.t, 0.5, 1e-12);

        // Sliding along a rotated/translated boundary face across multiple tets must stay found
        // even under tiny (-1e-13) outward numerical drift.
        const Vec3 offset{2.5, -1.3, 3.7};
        const MeshProperties rcube = transformed_mesh(cube_mesh(3), 1.0, true, offset);
        const MeshLocator rloc(rcube);
        const Vec3 sx0 = rigid_transform({0.0, 0.15, 0.35}, offset);
        const Vec3 sx1 = rigid_transform({-1e-13, 0.85, 0.65}, offset);
        const auto sstart = rloc.locate(sx0);
        CHECK(sstart.has_value());
        const auto sr = walk_along_segment(rcube, sx0, sx1, sstart->element);
        CHECK(sr.status == SegmentStatus::found && sr.t == 1.0 && sr.steps > 0);
    }

    // 4. Stationary segment (x0 == x1) outside start walks interior-first like walk_to_point.
    {
        const MeshProperties mesh = carve(cube_mesh(4), [](const Vec3& c) { return c[0] > 0.5 && c[1] > 0.5; });
        const MeshLocator loc(mesh);
        const Vec3 from{0.41, 0.93, 0.47}, to{0.93, 0.07, 0.44};
        const auto start = loc.locate(from);
        const auto goal = loc.locate(to);
        CHECK(start.has_value() && goal.has_value());
        const auto pw = walk_to_point(mesh, to, start->element);
        const auto sw = walk_along_segment(mesh, to, to, start->element);
        CHECK(pw.status == PointStatus::found && sw.status == SegmentStatus::found);
        CHECK(sw.element == goal->element && sw.element == pw.element && sw.t == 1.0);
    }
}

void test_robustness_and_edge_cases() {
    // 1. Query-independent volume6: very distant query (2^54) does not cancel volume to 0.
    {
        const MeshProperties mesh = cube_mesh(2);
        const MeshLocator loc(mesh);
        const Vec3 far_pt{static_cast<double>(1ULL << 54), 0.25, 0.25};
        const auto w = walk_to_point(mesh, far_pt, 0);
        CHECK(w.status == PointStatus::exited);
        CHECK(!loc.locate_near(far_pt, 0).has_value());
        CHECK(!loc.locate(far_pt).has_value());
    }

    // 2. Crafted coplanar zero-volume tet is rejected by constructor and free walks.
    {
        MeshProperties flat;
        flat.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
        flat.elements = {{{0, 1, 2, 3}, 0}};
        flat.materials = {{"one", 1.0, 0.0, false, {}}};
        flat.neighbours = {MeshProperties::no_neighbours};
        CHECK_THROWS(std::invalid_argument, MeshLocator{flat}, "has volume");
        CHECK_THROWS(std::invalid_argument, walk_to_point(flat, {1e9, 2e9, 3e9}, 0), "has volume");
        CHECK_THROWS(std::invalid_argument, walk_along_segment(flat, {0, 0, 0}, {1e9, 2e9, 3e9}, 0), "has volume");
    }

    // 3. Overflowing dot products (NaN) never falsely return `found`.
    {
        const MeshProperties huge = transformed_mesh(cube_mesh(1), 1e100, false, {0, 0, 0});
        const MeshLocator loc(huge);
        const Vec3 extreme{1e308, -1e308, 1e308};
        const auto w = walk_to_point(huge, extreme, 0);
        CHECK(w.status != PointStatus::found);
        CHECK(!loc.locate(extreme).has_value());
        CHECK(!loc.locate_near(extreme, 0).has_value());
    }

    // 4. Malformed mesh with disjoint mutually linked neighbours never falsely reports `found`.
    {
        MeshProperties broken;
        broken.nodes = {
            {0, 0, 0},   {1, 0, 0},   {0, 1, 0},   {0, 0, 1},
            {10, 0, 0},  {11, 0, 0},  {10, 1, 0},  {10, 0, 1},
        };
        broken.elements = {{{0, 1, 2, 3}, 0}, {{4, 5, 6, 7}, 0}};
        broken.materials = {{"one", 1.0, 0.0, false, {}}};
        broken.neighbours = {MeshProperties::no_neighbours, MeshProperties::no_neighbours};
        // Link face 0 (opposite node 0 / node 4, i.e. x+y+z = 1 and (x-10)+y+z = 1) mutually.
        broken.neighbours[0][0] = 1;
        broken.neighbours[1][0] = 0;
        const auto r = walk_along_segment(broken, {0.1, 0.1, 0.1}, {2.0, 0.1, 0.1}, 0, 8);
        CHECK(r.status != SegmentStatus::found);
    }

    // 5. Cold locate finds a point with three barycentric coords at -0.95e-9 (4th = 1 + 2.85e-9).
    {
        MeshProperties single;
        single.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        single.elements = {{{0, 1, 2, 3}, 0}};
        single.materials = {{"one", 1.0, 0.0, false, {}}};
        single.neighbours = {MeshProperties::no_neighbours};
        const MeshLocator loc(single);
        const Vec3 corner_near = bary_point(single, 0, {1.0 + 2.85e-9, -0.95e-9, -0.95e-9, -0.95e-9});
        const auto found = loc.locate(corner_near);
        CHECK(found.has_value() && found->element == 0);
        const auto near = loc.locate_near(corner_near, 0);
        CHECK(near.has_value() && near->element == 0);
    }

    // 6. Scale invariance: crossing order on 2^(+-200)-scaled meshes matches unit scale, and
    //    SI nanometer-scale meshes work cleanly without absolute padding bloat.
    {
        const MeshProperties unit = cube_mesh(3);
        const Vec3 x0{0.05, 0.12, 0.18}, x1{0.92, 0.87, 0.79};
        const auto ref = walk_along_segment(unit, x0, x1, 0);
        CHECK(ref.status == SegmentStatus::found);
        for (double scale : {std::ldexp(1.0, 200), std::ldexp(1.0, -200), 1e-9}) {
            const MeshProperties scaled = transformed_mesh(unit, scale, false, {0, 0, 0});
            const Vec3 sx0{x0[0] * scale, x0[1] * scale, x0[2] * scale};
            const Vec3 sx1{x1[0] * scale, x1[1] * scale, x1[2] * scale};
            const auto r = walk_along_segment(scaled, sx0, sx1, 0);
            CHECK(r.status == ref.status && r.element == ref.element && r.steps == ref.steps);
            CHECK_NEAR(r.t, ref.t, 1e-12);
        }
    }

    // 7. query_sphere: r = 0 just outside a face (5e-10) must NOT hit; huge radius throws.
    {
        MeshProperties single;
        single.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        single.elements = {{{0, 1, 2, 3}, 0}};
        single.materials = {{"one", 1.0, 0.0, false, {}}};
        single.neighbours = {MeshProperties::no_neighbours};
        const MeshLocator loc(single);
        CHECK(loc.query_sphere({-5e-10, 0.25, 0.25}, 0.0).empty());
        CHECK(loc.query_sphere({0.0, 0.25, 0.25}, 0.0) == (std::vector<uint32_t>{0}));
        CHECK(loc.query_sphere({0.25, 0.25, 0.25}, 0.0) == (std::vector<uint32_t>{0}));
        CHECK_THROWS(std::invalid_argument, loc.query_sphere({0.0, 0.0, 0.0}, 1e200), "too large");

        const MeshProperties empty;
        const MeshLocator empty_loc(empty);
        CHECK_THROWS(std::invalid_argument, empty_loc.query_sphere({0.0, 0.0, 0.0}, 1e200), "too large");
    }

    // 8. Exact tie across BVH leaves in locate(): largest min barycentric, then smallest index.
    {
        const MeshProperties mesh = cube_mesh(3);
        const MeshLocator loc(mesh);
        // Interior grid vertex shared by many tets spanning multiple BVH leaves.
        const Vec3 v{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        const auto got = loc.locate(v);
        CHECK(got.has_value());
        uint32_t expected = std::numeric_limits<uint32_t>::max();
        double best_min = -std::numeric_limits<double>::infinity();
        for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
            const double m = point_in_tet_min_bary(mesh, k, v);
            if (m >= -1e-9 && (m > best_min + 1e-15 || (std::abs(m - best_min) <= 1e-15 && k < expected))) {
                best_min = m;
                expected = k;
            }
        }
        CHECK(got->element == expected);
    }

    // 9. walk_to_point boundary exit after hop 16 (remembering-walk phase).
    {
        const MeshProperties big = cube_mesh(12);
        const auto r = walk_to_point(big, {1.2, 0.91, 0.91}, 0, 1000);
        CHECK(r.status == PointStatus::exited && r.steps > 16);
        CHECK(big.neighbours[r.element][r.exit_face] == MeshProperties::no_neighbour);
        CHECK(face_on_plane(big, r.element, r.exit_face, 0, 1.0));
    }
}

} // namespace

int main() {
    test_locate_interior_boundary_and_outside();
    test_non_convex_cavity_fallback();
    test_sphere_and_box_queries();
    test_empty_and_bad_inputs();
    test_barycentric_with_permuted_local_order();
    test_walk_to_point();
    test_walk_to_point_non_convex();
    test_walk_along_segment();
    test_walk_along_segment_cavity();
    test_walk_bad_inputs();
    test_range_queries_brute_force();
    test_edge_grazes_and_wall_tolerance();
    test_robustness_and_edge_cases();
    return finish("locator");
}
