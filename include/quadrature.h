#pragma once

#include <array>
#include <vector>

// Quadrature rules on a tetrahedron and on a triangle, in barycentric coordinates. The
// weights are positive and sum to 1, so the integral over an element is its volume (area)
// times the weighted sum.
//
// The rules are collapsed Gauss-Legendre products (Stroud's conical product with the
// Jacobian folded into the weights): exact for every polynomial of total degree <= degree,
// with more points than the best known rules of the same degree, but simple, positive and
// available for any degree.

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
