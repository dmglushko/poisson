#pragma once

// Private helpers shared by the verification cases: validation, the importer's naming of
// boundary conditions and volumes, and pieces of the gmsh .geo scripts.

#include "mesh_properties.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <utility>
#include <string>
#include <string_view>
#include <vector>

namespace verification_detail {

using Vec3 = std::array<double, 3>;
using BoundaryCondition = MeshProperties::BoundaryCondition;

[[noreturn]] void invalid(std::string_view where, const std::string& message);

// Shortest decimal text that reads back as exactly the same double.
std::string number(double value);

void require_finite(std::string_view where, double value, std::string_view what);
void require_positive(std::string_view where, double value, std::string_view what);
void require_non_negative(std::string_view where, double value, std::string_view what);

// A letter, then letters, digits or '_'; not "vacuum" and not starting with "detector"
// (the importer gives those a special meaning).
void require_material_name(std::string_view where, const std::string& name);

// Dirichlet/Neumann without alpha, Robin with a finite alpha >= 0, finite value.
void require_boundary_condition(std::string_view where, const BoundaryCondition& condition);

// The importer's surface name of a condition: "<u>V", "D<g>", "R<alpha>,<g>".
std::string surface_name(const BoundaryCondition& condition);

// The importer's volume name of a material: its name, plus " <voltage>V" when prescribed.
std::string volume_name(const MeshProperties::Material& material);

bool same_condition(const BoundaryCondition& a, const BoundaryCondition& b);

// Adds `condition` to `list` unless an equal one is already there.
void add_condition(std::vector<BoundaryCondition>& list, const BoundaryCondition& condition);

// A charge position as stored in a .cha file (float32), read back as double.
Vec3 as_stored(const Vec3& position);

// The start of every .geo script: the OpenCASCADE kernel and the mesh size constants.
std::string geo_header();

// The end of every .geo script: a finer mesh around `points` when h_charge > 0.
std::string geo_refinement(const std::vector<Vec3>& points);

// Physical groups: one statement per distinct name, joining the given list variables.
struct PhysicalGroup {
    std::string              name;
    std::vector<std::string> lists;   // .geo list variables such as "s_left[]"
};
void add_to_group(std::vector<PhysicalGroup>& groups, const std::string& name, const std::string& list);
std::string geo_physicals(const char* kind, const std::vector<PhysicalGroup>& groups);   // kind: "Volume" or "Surface"

// G(r) = exp(-k r) / r and dG/dr.
inline double yukawa(double r, double k) { return std::exp(-k * r) / r; }
inline double yukawa_derivative(double r, double k) { return -(k + 1 / r) * std::exp(-k * r) / r; }

inline double radius(const Vec3& x) { return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); }

// Every point within tol of the plane x[axis] = value.
template <std::size_t N>
bool on_plane(const std::array<Vec3, N>& points, int axis, double value, double tol) {
    for (const auto& p : points)
        if (std::abs(p[axis] - value) > tol) return false;
    return true;
}

// Every point within the box [lo, hi] grown by tol.
template <std::size_t N>
bool inside_box(const std::array<Vec3, N>& points, const Vec3& lo, const Vec3& hi, double tol) {
    for (const auto& p : points)
        for (int k = 0; k < 3; ++k)
            if (p[k] < lo[k] - tol || p[k] > hi[k] + tol) return false;
    return true;
}

// The box face (axis, low or high side) all points lie on, or -1 (as axis) if none.
template <std::size_t N>
std::pair<int, int> box_face(const std::array<Vec3, N>& points, const Vec3& lo, const Vec3& hi, double tol) {
    for (int a = 0; a < 3; ++a) {
        if (on_plane(points, a, lo[a], tol)) return {a, 0};
        if (on_plane(points, a, hi[a], tol)) return {a, 1};
    }
    return {-1, 0};
}

template <std::size_t N>
std::pair<double, double> radius_range(const std::array<Vec3, N>& points) {
    double lo = radius(points[0]), hi = lo;
    for (const auto& p : points) {
        lo = std::min(lo, radius(p));
        hi = std::max(hi, radius(p));
    }
    return {lo, hi};
}

// Geometry checks shared by the cases (see mesh_mismatches). Flat cases know exactly where
// each face and region is: `condition_on` gives the condition the case puts on a triangle
// (empty if the triangle is not on the case's boundary), `fits` whether a tet lies in a
// material's region.
std::vector<std::string> flat_geometry_mismatches(
    const MeshProperties& mesh, const std::vector<std::optional<std::size_t>>& material,
    const std::function<std::optional<BoundaryCondition>(const std::array<Vec3, 3>&)>& condition_on,
    const std::function<bool(const std::array<Vec3, 4>&, std::size_t)>& fits);

// Concentric cases. A meshed sphere is a polyhedron inside the sphere: some of its vertices
// lie on the sphere, the others (refined facets) inside it. So every vertex must be within
// its layer's outer radius, and some vertex of every layer, and of every sphere's boundary
// triangles, must lie on that radius. Below, a tet may reach half the layer under it.
struct SphereBoundary {
    double            radius;
    BoundaryCondition condition;
};
std::vector<std::string> spherical_geometry_mismatches(const MeshProperties& mesh,
                                                       const std::vector<std::optional<std::size_t>>& material,
                                                       const std::vector<std::string>& names,
                                                       const std::vector<std::pair<double, double>>& layers,
                                                       const std::vector<SphereBoundary>& spheres);

inline double distance(const Vec3& a, const Vec3& b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

}   // namespace verification_detail
