// Tests for the quadrature rules: positivity, strict interiority, permutation symmetry
// (degrees 0..8), and monomial exactness up to max_quadrature_degree.

#include "common/quadrature.h"
#include "test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

double factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

template <typename Point, std::size_t N>
void check_symmetry_and_uniqueness(const std::vector<Point>& rule) {
    // Every pair of points must be distinct.
    for (std::size_t i = 0; i < rule.size(); ++i) {
        for (std::size_t j = i + 1; j < rule.size(); ++j) {
            double max_diff = 0;
            for (std::size_t k = 0; k < N; ++k)
                max_diff = std::max(max_diff, std::abs(rule[i].barycentric[k] - rule[j].barycentric[k]));
            CHECK(max_diff > 1e-12);
        }
    }

    // Closed under every permutation of the N barycentric coordinates (S_N).
    std::array<std::size_t, N> perm{};
    std::iota(perm.begin(), perm.end(), 0);
    do {
        for (const auto& q : rule) {
            std::array<double, N> permuted{};
            for (std::size_t k = 0; k < N; ++k) permuted[k] = q.barycentric[perm[k]];
            bool found = false;
            for (const auto& r : rule) {
                double coord_err = 0;
                for (std::size_t k = 0; k < N; ++k)
                    coord_err = std::max(coord_err, std::abs(r.barycentric[k] - permuted[k]));
                if (coord_err < 1e-13 && std::abs(r.weight - q.weight) < 1e-13) {
                    found = true;
                    break;
                }
            }
            CHECK(found);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
}

void test_tet() {
    constexpr std::array<std::size_t, 9> expected_symmetric_counts{1, 1, 4, 8, 14, 14, 24, 35, 46};
    for (int degree = 0; degree <= max_quadrature_degree; ++degree) {
        const auto rule = tet_rule(degree);
        if (degree <= 8) {
            CHECK(rule.size() == expected_symmetric_counts[degree]);
            check_symmetry_and_uniqueness<TetQuadraturePoint, 4>(rule);
        }
        double sum = 0;
        bool positive = true;
        bool interior = true;
        for (const auto& q : rule) {
            sum += q.weight;
            positive = positive && q.weight > 0;
            double bsum = 0;
            for (double l : q.barycentric) {
                interior = interior && l > 0;
                bsum += l;
            }
            CHECK_NEAR(bsum, 1, 1e-13);
        }
        CHECK_NEAR(sum, 1, 1e-13);
        CHECK(positive);
        CHECK(interior);
        // The mean of x^a y^b z^c over the unit tet: 6 a! b! c! / (a + b + c + 3)!.
        for (int a = 0; a <= degree; ++a)
            for (int b = 0; a + b <= degree; ++b)
                for (int c = 0; a + b + c <= degree; ++c) {
                    double value = 0;
                    for (const auto& q : rule)
                        value += q.weight * std::pow(q.barycentric[1], a) * std::pow(q.barycentric[2], b) *
                                 std::pow(q.barycentric[3], c);
                    const double exact = 6 * factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
                    CHECK_NEAR(value / exact, 1, 1e-11);
                }
    }
}

void test_triangle() {
    constexpr std::array<std::size_t, 9> expected_symmetric_counts{1, 1, 3, 6, 6, 7, 12, 15, 16};
    for (int degree = 0; degree <= max_quadrature_degree; ++degree) {
        const auto rule = triangle_rule(degree);
        if (degree <= 8) {
            CHECK(rule.size() == expected_symmetric_counts[degree]);
            check_symmetry_and_uniqueness<TriangleQuadraturePoint, 3>(rule);
        }
        double sum = 0;
        bool positive = true;
        bool interior = true;
        for (const auto& q : rule) {
            sum += q.weight;
            positive = positive && q.weight > 0;
            double bsum = 0;
            for (double l : q.barycentric) {
                interior = interior && l > 0;
                bsum += l;
            }
            CHECK_NEAR(bsum, 1, 1e-13);
        }
        CHECK_NEAR(sum, 1, 1e-13);
        CHECK(positive);
        CHECK(interior);
        for (int a = 0; a <= degree; ++a)
            for (int b = 0; a + b <= degree; ++b) {
                double value = 0;
                for (const auto& q : rule)
                    value += q.weight * std::pow(q.barycentric[1], a) * std::pow(q.barycentric[2], b);
                const double exact = 2 * factorial(a) * factorial(b) / factorial(a + b + 2);
                CHECK_NEAR(value / exact, 1, 1e-11);
            }
    }
}

int main() {
    test_tet();
    test_triangle();
    CHECK_THROWS(std::invalid_argument, tet_rule(-1), "quadrature degree must be in [0, 20]");
    CHECK_THROWS(std::invalid_argument, tet_rule(21), "quadrature degree must be in [0, 20]");
    CHECK_THROWS(std::invalid_argument, triangle_rule(-1), "quadrature degree must be in [0, 20]");
    CHECK_THROWS(std::invalid_argument, triangle_rule(21), "quadrature degree must be in [0, 20]");
    return finish("quadrature");
}
