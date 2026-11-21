#include "common/quadrature.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// n-point Gauss-Legendre rule on [0, 1]: exact up to degree 2n - 1.
std::vector<std::pair<double, double>> gauss_legendre(int n) {
    std::vector<std::pair<double, double>> rule;
    for (int i = 1; i <= n; ++i) {
        // Newton on P_n, starting from the usual estimate of the i-th root on [-1, 1].
        double x = std::cos(std::numbers::pi * (i - 0.25) / (n + 0.5));
        double derivative = 1;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p0 = 1, p1 = x;   // P_0, P_1, then up to P_{n-1}, P_n
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            const double pn = n == 0 ? 1 : p1, pn_1 = n == 1 ? 1 : p0;
            derivative = n * (x * pn - pn_1) / (x * x - 1);
            const double step = pn / derivative;
            x -= step;
            if (std::abs(step) < 1e-16) break;
        }
        const double weight = 2 / ((1 - x * x) * derivative * derivative);
        rule.emplace_back(0.5 * (1 - x), 0.5 * weight);   // mapped to [0, 1]
    }
    return rule;
}

// The number of Gauss points that integrate a polynomial of degree d exactly.
int points_for(int d) { return d / 2 + 1; }

void check_degree(int degree) {
    if (degree < 0 || degree > max_quadrature_degree)
        throw std::invalid_argument("quadrature degree must be in [0, " + std::to_string(max_quadrature_degree) +
                                    "], got " + std::to_string(degree));
}

} // namespace

// Unit tet x, y, z >= 0, x + y + z <= 1, as the image of the cube [0, 1]^3:
//   x = a, y = (1 - a) b, z = (1 - a)(1 - b) c,  Jacobian (1 - a)^2 (1 - b).
// A polynomial of degree p in x, y, z becomes one of degree p + 2 in a, p + 1 in b, p in c.
std::vector<TetQuadraturePoint> tet_rule(int degree) {
    check_degree(degree);
    const auto ra = gauss_legendre(points_for(degree + 2));
    const auto rb = gauss_legendre(points_for(degree + 1));
    const auto rc = gauss_legendre(points_for(degree));
    std::vector<TetQuadraturePoint> rule;
    rule.reserve(ra.size() * rb.size() * rc.size());
    for (const auto& [a, wa] : ra)
        for (const auto& [b, wb] : rb)
            for (const auto& [c, wc] : rc) {
                const double x = a, y = (1 - a) * b, z = (1 - a) * (1 - b) * c;
                const double weight = 6 * wa * wb * wc * (1 - a) * (1 - a) * (1 - b);   // volume 1/6
                rule.push_back({{1 - x - y - z, x, y, z}, weight});
            }
    return rule;
}

// Unit triangle as the image of the square: x = a, y = (1 - a) b, Jacobian 1 - a.
std::vector<TriangleQuadraturePoint> triangle_rule(int degree) {
    check_degree(degree);
    const auto ra = gauss_legendre(points_for(degree + 1));
    const auto rb = gauss_legendre(points_for(degree));
    std::vector<TriangleQuadraturePoint> rule;
    rule.reserve(ra.size() * rb.size());
    for (const auto& [a, wa] : ra)
        for (const auto& [b, wb] : rb) {
            const double x = a, y = (1 - a) * b;
            rule.push_back({{1 - x - y, x, y}, 2 * wa * wb * (1 - a)});   // area 1/2
        }
    return rule;
}
