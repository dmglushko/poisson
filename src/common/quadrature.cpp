#include "common/quadrature.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// ---------------------------------------------------------------------------
// Symmetric, strictly positive, strictly interior simplex rules for degrees 0..8.
//
// Parameters are derived from Witherden & Vincent (2015), "On the identification of
// symmetric quadrature rules for finite element methods", Comput. Math. Appl. 69(10),
// 1232-1241, via PyFR's BSD-3 quadrature tables (pyfr/quadrules/{tet,tri}/
// witherden-vincent-*-sp.txt), mapped to barycentric coordinates with weights summing to 1.
// ---------------------------------------------------------------------------

void tet_orbit_1(std::vector<TetQuadraturePoint>& rule, double w) {
    rule.push_back({{0.25, 0.25, 0.25, 0.25}, w});
}

void tet_orbit_4(std::vector<TetQuadraturePoint>& rule, double a, double w) {
    const double b = 1.0 - 3.0 * a;
    rule.push_back({{b, a, a, a}, w});
    rule.push_back({{a, b, a, a}, w});
    rule.push_back({{a, a, b, a}, w});
    rule.push_back({{a, a, a, b}, w});
}

void tet_orbit_6(std::vector<TetQuadraturePoint>& rule, double a, double w) {
    const double b = 0.5 - a;
    rule.push_back({{a, a, b, b}, w});
    rule.push_back({{a, b, a, b}, w});
    rule.push_back({{a, b, b, a}, w});
    rule.push_back({{b, a, a, b}, w});
    rule.push_back({{b, a, b, a}, w});
    rule.push_back({{b, b, a, a}, w});
}

void tet_orbit_12(std::vector<TetQuadraturePoint>& rule, double a, double b, double w) {
    const double c = 1.0 - 2.0 * a - b;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) continue;
            std::array<double, 4> l{a, a, a, a};
            l[i] = b;
            l[j] = c;
            rule.push_back({l, w});
        }
    }
}

std::vector<TetQuadraturePoint> symmetric_tet_rule(int degree) {
    std::vector<TetQuadraturePoint> rule;
    if (degree <= 1) {
        rule.reserve(1);
        tet_orbit_1(rule, 1.0);
    } else if (degree == 2) {
        rule.reserve(4);
        tet_orbit_4(rule, 0.1381966011250105, 0.25);
    } else if (degree == 3) {
        rule.reserve(8);
        tet_orbit_4(rule, 0.32816330251638171, 0.13621784253708735);
        tet_orbit_4(rule, 0.10804724989842862, 0.11378215746291265);
    } else if (degree <= 5) {
        rule.reserve(14);
        tet_orbit_4(rule, 0.31088591926330061, 0.11268792571801586);
        tet_orbit_4(rule, 0.092735250310891207, 0.073493043116361956);
        tet_orbit_6(rule, 0.045503704125649636, 0.042546020777081466);
    } else if (degree == 6) {
        rule.reserve(24);
        tet_orbit_4(rule, 0.32233789014227548, 0.055357181543654724);
        tet_orbit_4(rule, 0.040673958534611337, 0.010077211055320643);
        tet_orbit_4(rule, 0.21460287125915203, 0.039922750258167494);
        tet_orbit_12(rule, 0.063661001875017498, 0.26967233145831582, 0.04821428571428571);
    } else if (degree == 7) {
        rule.reserve(35);
        tet_orbit_1(rule, 0.095485289464130846);
        tet_orbit_4(rule, 0.31570114977820279, 0.042329581209967028);
        tet_orbit_6(rule, 0.050489822598396336, 0.03189692783285758);
        tet_orbit_12(rule, 0.021265472541483241, 0.1466388138184849, 0.008110770829903342);
        tet_orbit_12(rule, 0.18883383102600104, 0.047160700360997919, 0.03720713072833462);
    } else {
        // degree == 8
        rule.reserve(46);
        tet_orbit_4(rule, 0.04231654368476731, 0.0075252561535401989);
        tet_orbit_4(rule, 0.31418170912403903, 0.041763782856934897);
        tet_orbit_4(rule, 0.10795272496221082, 0.02642665090840883);
        tet_orbit_4(rule, 0.1851094877825866, 0.052031747563738531);
        tet_orbit_6(rule, 0.064408671416169794, 0.036280930261308825);
        tet_orbit_12(rule, 0.20413933387602912, 0.0079239539457973684, 0.015453486150960335);
        tet_orbit_12(rule, 0.021433930127130574, 0.23966807631943055, 0.0071569028908444327);
    }
    return rule;
}

void tri_orbit_1(std::vector<TriangleQuadraturePoint>& rule, double w) {
    rule.push_back({{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, w});
}

void tri_orbit_3(std::vector<TriangleQuadraturePoint>& rule, double a, double w) {
    const double b = 1.0 - 2.0 * a;
    rule.push_back({{b, a, a}, w});
    rule.push_back({{a, b, a}, w});
    rule.push_back({{a, a, b}, w});
}

void tri_orbit_6(std::vector<TriangleQuadraturePoint>& rule, double a, double b, double w) {
    const double c = 1.0 - a - b;
    rule.push_back({{a, b, c}, w});
    rule.push_back({{a, c, b}, w});
    rule.push_back({{b, a, c}, w});
    rule.push_back({{b, c, a}, w});
    rule.push_back({{c, a, b}, w});
    rule.push_back({{c, b, a}, w});
}

std::vector<TriangleQuadraturePoint> symmetric_triangle_rule(int degree) {
    std::vector<TriangleQuadraturePoint> rule;
    if (degree <= 1) {
        rule.reserve(1);
        tri_orbit_1(rule, 1.0);
    } else if (degree == 2) {
        rule.reserve(3);
        tri_orbit_3(rule, 1.0 / 6.0, 1.0 / 3.0);
    } else if (degree <= 4) {
        rule.reserve(6);
        tri_orbit_3(rule, 0.091576213509770743, 0.10995174365532186);
        tri_orbit_3(rule, 0.44594849091596495, 0.22338158967801147);
    } else if (degree == 5) {
        rule.reserve(7);
        tri_orbit_1(rule, 0.225);
        tri_orbit_3(rule, 0.47014206410511505, 0.13239415278850619);
        tri_orbit_3(rule, 0.10128650732345634, 0.12593918054482714);
    } else if (degree == 6) {
        rule.reserve(12);
        tri_orbit_3(rule, 0.063089014491502227, 0.050844906370206812);
        tri_orbit_3(rule, 0.24928674517091043, 0.11678627572637938);
        tri_orbit_6(rule, 0.053145049844816938, 0.31035245103378439, 0.082851075618373571);
    } else if (degree == 7) {
        rule.reserve(15);
        tri_orbit_3(rule, 0.03373064855458785, 0.016545050110792131);
        tri_orbit_3(rule, 0.47430969250471822, 0.077086646185986069);
        tri_orbit_3(rule, 0.24157738259540359, 0.12794417123015558);
        tri_orbit_6(rule, 0.047036644652595237, 0.1986833147973516, 0.055878732903199779);
    } else {
        // degree == 8
        rule.reserve(16);
        tri_orbit_1(rule, 0.14431560767778717);
        tri_orbit_3(rule, 0.050547228317030957, 0.032458497623198079);
        tri_orbit_3(rule, 0.45929258829272318, 0.095091634267284619);
        tri_orbit_3(rule, 0.17056930775176019, 0.10321737053471824);
        tri_orbit_6(rule, 0.0083947774099576069, 0.26311282963463817, 0.027230314174434989);
    }
    return rule;
}

// ---------------------------------------------------------------------------
// Collapsed Gauss-Legendre product fallback for degrees 9..max_quadrature_degree.
// ---------------------------------------------------------------------------

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
    if (degree <= 8) return symmetric_tet_rule(degree);
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
    if (degree <= 8) return symmetric_triangle_rule(degree);
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
