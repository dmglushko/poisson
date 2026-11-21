#pragma once

// Shared by the tests: checks, and a structured tet mesh of a cube.

#include "mesh_properties.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

inline int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::cerr << "FAILED line " << __LINE__ << ": " #condition "\n";          \
            ++failures;                                                               \
        }                                                                             \
    } while (false)

#define CHECK_NEAR(a, b, tolerance)                                                   \
    do {                                                                              \
        const double a_ = (a), b_ = (b);                                              \
        if (!(std::abs(a_ - b_) <= (tolerance))) {                                    \
            std::cerr << "FAILED line " << __LINE__ << ": " #a " = " << a_            \
                      << ", expected " << b_ << " within " << (tolerance) << "\n";    \
            ++failures;                                                               \
        }                                                                             \
    } while (false)

// The call must throw `Exception` with `expected` somewhere in its message.
template <typename Exception>
void check_throws(int line, const std::function<void()>& call, const std::string& expected) {
    try {
        call();
    } catch (const Exception& error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            std::cerr << "FAILED line " << line << ": expected '" << expected << "' in: " << error.what() << "\n";
            ++failures;
        }
        return;
    } catch (const std::exception& error) {
        std::cerr << "FAILED line " << line << ": wrong exception type: " << error.what() << "\n";
        ++failures;
        return;
    }
    std::cerr << "FAILED line " << line << ": no exception, expected '" << expected << "'\n";
    ++failures;
}
#define CHECK_THROWS(Exception, call, expected) check_throws<Exception>(__LINE__, [&] { call; }, expected)

inline int finish(const char* name) {
    if (failures == 0) std::cout << "all " << name << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

using Vec3 = std::array<double, 3>;

inline double signed_volume(const MeshProperties& mesh, const std::array<uint32_t, 4>& n) {
    const auto& p = mesh.nodes;
    const Vec3 a{p[n[1]][0] - p[n[0]][0], p[n[1]][1] - p[n[0]][1], p[n[1]][2] - p[n[0]][2]};
    const Vec3 b{p[n[2]][0] - p[n[0]][0], p[n[2]][1] - p[n[0]][1], p[n[2]][2] - p[n[0]][2]};
    const Vec3 c{p[n[3]][0] - p[n[0]][0], p[n[3]][1] - p[n[0]][1], p[n[3]][2] - p[n[0]][2]};
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
}

// Which face of the unit cube a boundary triangle lies on: 0..5 for x=0, x=1, y=0, y=1,
// z=0, z=1.
inline int cube_side(const MeshProperties& mesh, const std::array<uint32_t, 3>& t) {
    for (int d = 0; d < 3; ++d)
        for (int s = 0; s < 2; ++s) {
            bool all = true;
            for (uint32_t node : t) all = all && mesh.nodes[node][d] == s;
            if (all) return 2 * d + s;
        }
    return -1;
}

// The unit cube cut into n^3 small cubes, each into 6 tets (Kuhn), positively oriented.
// Boundary triangles, oriented outward, get boundary condition side_bc[side] (cube_side).
// Material 1 (if given) for tets with centroid x > split_x, else 0. Neighbours are filled.
inline MeshProperties cube_mesh(int n, std::array<uint16_t, 6> side_bc = {0, 0, 0, 0, 0, 0},
                                std::vector<MeshProperties::BoundaryCondition> bcs = {{MeshProperties::BcType::Dirichlet, 0, {}}},
                                std::vector<MeshProperties::Material> materials = {{"one", 1, 0, false, {}}},
                                double split_x = 2) {
    MeshProperties mesh;
    const auto id = [n](int i, int j, int k) { return static_cast<uint32_t>((i * (n + 1) + j) * (n + 1) + k); };
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            for (int k = 0; k <= n; ++k) mesh.nodes.push_back({double(i) / n, double(j) / n, double(k) / n});
    const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (const auto& perm : perms) {
                    std::array<int, 3> c{i, j, k};
                    std::array<uint32_t, 4> t{};
                    t[0] = id(c[0], c[1], c[2]);
                    for (int s = 0; s < 3; ++s) {
                        ++c[perm[s]];
                        t[s + 1] = id(c[0], c[1], c[2]);
                    }
                    if (signed_volume(mesh, t) < 0) std::swap(t[2], t[3]);
                    double cx = 0;
                    for (uint32_t node : t) cx += mesh.nodes[node][0] / 4;
                    mesh.elements.push_back({t, static_cast<uint16_t>(cx > split_x ? 1 : 0)});
                }
    // Faces that belong to one tet are on the boundary.
    std::map<std::array<uint32_t, 3>, std::pair<int, std::array<uint32_t, 4>>> faces;
    std::map<std::array<uint32_t, 3>, int> local;
    for (const auto& e : mesh.elements)
        for (int f = 0; f < 4; ++f) {
            std::array<uint32_t, 3> face{};
            for (int a = 0, b = 0; a < 4; ++a)
                if (a != f) face[b++] = e.node[a];
            auto key = face;
            std::sort(key.begin(), key.end());
            auto& entry = faces[key];
            ++entry.first;
            entry.second = e.node;
            local[key] = f;
        }
    for (const auto& [key, entry] : faces) {
        if (entry.first != 1) continue;
        // Outward: the face of a positive tet, opposite node f, in the order that makes the
        // remaining node lie behind it.
        const int f = local[key];
        std::array<uint32_t, 3> t{};
        for (int a = 0, b = 0; a < 4; ++a)
            if (a != f) t[b++] = entry.second[a];
        if (signed_volume(mesh, {t[0], t[1], t[2], entry.second[f]}) > 0) std::swap(t[1], t[2]);
        mesh.boundary_elements.push_back({t, side_bc[cube_side(mesh, t)]});
    }
    mesh.boundary_conditions = std::move(bcs);
    mesh.materials = std::move(materials);
    // Neighbours.
    std::map<std::array<uint32_t, 3>, std::pair<uint32_t, int>> seen;
    mesh.neighbours.assign(mesh.elements.size(), {});
    for (uint32_t e = 0; e < mesh.elements.size(); ++e)
        for (int f = 0; f < 4; ++f) {
            std::array<uint32_t, 3> key{};
            for (int a = 0, b = 0; a < 4; ++a)
                if (a != f) key[b++] = mesh.elements[e].node[a];
            std::sort(key.begin(), key.end());
            auto [it, inserted] = seen.try_emplace(key, e, f);
            if (!inserted) {
                mesh.neighbours[e][f] = it->second.first;
                mesh.neighbours[it->second.first][it->second.second] = e;
            }
        }
    return mesh;
}
