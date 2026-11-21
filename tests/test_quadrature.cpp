// Tests for the quadrature rules: exact for every monomial up to the degree.

#include "quadrature.h"
#include "test_support.h"

#include <stdexcept>

double factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

void test_tet() {
    for (int degree = 0; degree <= max_quadrature_degree; ++degree) {
        const auto rule = tet_rule(degree);
        double sum = 0;
        bool positive = true;
        for (const auto& q : rule) {
            sum += q.weight;
            positive = positive && q.weight > 0;
        }
        CHECK_NEAR(sum, 1, 1e-13);
        CHECK(positive);
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
    for (int degree = 0; degree <= max_quadrature_degree; ++degree) {
        const auto rule = triangle_rule(degree);
        double sum = 0;
        for (const auto& q : rule) sum += q.weight;
        CHECK_NEAR(sum, 1, 1e-13);
        for (int a = 0; a <= degree; ++a)
            for (int b = 0; a + b <= degree; ++b) {
                double value = 0;
                for (const auto& q : rule) value += q.weight * std::pow(q.barycentric[1], a) * std::pow(q.barycentric[2], b);
                const double exact = 2 * factorial(a) * factorial(b) / factorial(a + b + 2);
                CHECK_NEAR(value / exact, 1, 1e-11);
            }
    }
}

int main() {
    test_tet();
    test_triangle();
    CHECK_THROWS(std::invalid_argument, tet_rule(-1), "quadrature degree must be in [0, 20]");
    CHECK_THROWS(std::invalid_argument, triangle_rule(21), "quadrature degree must be in [0, 20]");
    return finish("quadrature");
}
