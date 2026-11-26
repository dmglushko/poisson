#pragma once

#include <array>
#include <vector>

// Quadrature rules on a tetrahedron and on a triangle, in barycentric coordinates. Every
// point lies strictly inside the element and every weight is strictly positive, with
// weights summing to 1, so the integral over an element is its volume (area) times the
// weighted sum.
//
// Degrees 0..8 use fully symmetric Witherden-Vincent rules (compact point counts,
// invariant under vertex permutations); degrees 9..max_quadrature_degree fall back to
// collapsed Gauss-Legendre products (Stroud's conical product with the Jacobian folded
// into the weights). Every rule is exact for all polynomials of total degree <= degree.

constexpr int max_quadrature_degree = 20;

struct TetQuadraturePoint {
    std::array<double, 4> barycentric;
    double                weight;
};

struct TriangleQuadraturePoint {
    std::array<double, 3> barycentric;
    double                weight;
};

// Throws std::invalid_argument for a degree outside [0, max_quadrature_degree].
std::vector<TetQuadraturePoint>      tet_rule(int degree);
std::vector<TriangleQuadraturePoint> triangle_rule(int degree);
