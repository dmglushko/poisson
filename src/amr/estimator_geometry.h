#pragma once

// Private to the estimators: what they need of a P1 solution on a tet mesh, computed once
// per call.

#include "mesh_properties.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace amr_detail {

using Vec3 = std::array<double, 3>;

inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

struct TetGeometry {
    double volume = 0;
    double diameter = 0;                          // the longest edge
    std::array<Vec3, 4> grad_lambda{};            // gradients of the barycentric coordinates
};

// One TetGeometry per tet. Throws std::invalid_argument, naming `who`, when a volume is not
// a finite positive number.
std::vector<TetGeometry> tet_geometry(const MeshProperties& mesh, const std::string& who);

// grad u_h on tet `element`, constant there.
Vec3 gradient(const MeshProperties& mesh, const TetGeometry& g, uint32_t element, std::span<const double> u);

// The point with barycentric coordinates `lambda` in tet `element`.
Vec3 point_at(const MeshProperties& mesh, uint32_t element, const std::array<double, 4>& lambda);

// eps and c of a material: -div(eps grad u) + c u = f.
struct Coefficients {
    double eps = 1;
    double c = 0;
    bool   metal = false;
};
Coefficients coefficients(const MeshProperties::Material& material);

// Every face of the mesh once. Local face i of a tet is opposite its node i.
struct Face {
    std::array<uint32_t, 3>  node;                // as in the tet `element`
    uint32_t                 element;
    uint8_t                  local;
    std::optional<uint32_t>  other;               // the tet on the other side, if any
    std::optional<uint16_t>  boundary_condition;  // from a boundary element on this face
};
std::vector<Face> faces(const MeshProperties& mesh);

// The longest edge of a triangle.
double triangle_diameter(const MeshProperties& mesh, const std::array<uint32_t, 3>& node);

// Throws std::invalid_argument unless 0 <= degree <= max_quadrature_degree, naming the setting.
void check_degree(int degree, const char* setting);

} // namespace amr_detail
