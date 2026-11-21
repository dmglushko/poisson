// Tests for the marking strategies; the cases are those of pde2d (tests/adapt/marking.cpp).

#include "amr/marking.h"
#include "test_support.h"

#include <limits>
#include <stdexcept>

using Marked = std::vector<uint32_t>;

MeshProperties mesh_with(std::size_t tets) {
    MeshProperties mesh;
    mesh.elements.resize(tets);
    return mesh;
}

Marked mark(const MarkingStrategy& strategy, const std::vector<double>& eta) {
    return strategy.mark(mesh_with(eta.size()), eta);
}

void check_bad_indicators(const MarkingStrategy& strategy) {
    CHECK_THROWS(std::invalid_argument, strategy.mark(mesh_with(3), std::vector<double>{1, 2}), "2 indicators for 3 tets");
    CHECK_THROWS(std::invalid_argument, mark(strategy, {1, -1}), "tet 1 must be finite and non-negative");
    CHECK_THROWS(std::invalid_argument, mark(strategy, {std::nan(""), 1}), "tet 0 must be finite and non-negative");
    CHECK_THROWS(std::invalid_argument, mark(strategy, {INFINITY}), "tet 0 must be finite and non-negative");
    CHECK(mark(strategy, {0, 0, 0}).empty());
    CHECK(mark(strategy, {}).empty());
}

void test_dorfler() {
    // Squares 1, 9, 4, 9, 0: total 23. Order: 1, 3 (a tie, by index), 2, 0.
    const std::vector<double> eta{1, 3, 2, 3, 0};
    CHECK(mark(DorflerMarking({.theta = 0.3}), eta) == (Marked{1}));          // 6.9: the lower index of the tie
    CHECK(mark(DorflerMarking({.theta = 0.5}), eta) == (Marked{1, 3}));       // 11.5
    CHECK(mark(DorflerMarking({.theta = 0.78}), eta) == (Marked{1, 3}));      // 17.94 < 18
    CHECK(mark(DorflerMarking({.theta = 0.79}), eta) == (Marked{1, 2, 3}));   // 18.17
    CHECK(mark(DorflerMarking({.theta = 1}), eta) == (Marked{0, 1, 2, 3}));   // every positive one
    CHECK(mark(DorflerMarking({.theta = 0}), eta).empty());
    CHECK(mark(DorflerMarking({.theta = 0.5}), {2, 2, 2, 2}) == (Marked{0, 1}));
    CHECK(mark(DorflerMarking({.theta = 0.5}), {1e200, 1e300, 1e300}) == (Marked{1}));   // no overflow
    CHECK(mark(DorflerMarking({.theta = 1}), {1e-4, 1e-13}) == (Marked{0, 1}));
    check_bad_indicators(DorflerMarking());
    for (double theta : {-1e-9, 1.5, std::nan("")})
        CHECK_THROWS(std::invalid_argument, DorflerMarking({.theta = theta}), "DorflerMarking::Settings::theta must be in [0, 1]");
}

void test_maximum() {
    const std::vector<double> eta{1, 4, 2, 3};
    CHECK(mark(MaximumMarking({.theta = 1}), eta) == (Marked{1}));
    CHECK(mark(MaximumMarking({.theta = 0.75}), eta) == (Marked{1, 3}));         // 3 >= 3 counts
    CHECK(mark(MaximumMarking({.theta = 0.25}), eta) == (Marked{0, 1, 2, 3}));
    CHECK(mark(MaximumMarking(), eta) == (Marked{1, 2, 3}));
    CHECK(mark(MaximumMarking({.theta = 0}), {0, 0, 3}) == (Marked{2}));         // zeros never
    check_bad_indicators(MaximumMarking());
    CHECK_THROWS(std::invalid_argument, MaximumMarking({.theta = 1.1}), "MaximumMarking::Settings::theta");
}

void test_fixed_fraction() {
    const std::vector<double> eta{1, 4, 2, 3, 0, 0, 0, 0, 0, 0};
    CHECK(mark(FixedFractionMarking({.fraction = 0.2}), eta) == (Marked{1, 3}));
    CHECK(mark(FixedFractionMarking({.fraction = 0.04}), eta).empty());           // rounds to 0
    CHECK(mark(FixedFractionMarking({.fraction = 1}), eta) == (Marked{0, 1, 2, 3}));
    std::vector<double> hundred(100, 1);
    CHECK(mark(FixedFractionMarking({.fraction = 0.07}), hundred).size() == 7);   // not 8
    check_bad_indicators(FixedFractionMarking());
    CHECK_THROWS(std::invalid_argument, FixedFractionMarking({.fraction = -0.1}), "FixedFractionMarking::Settings::fraction");
}

void test_equidistribution() {
    // Bound 2 / sqrt(4) = 1.
    CHECK(mark(EquidistributionMarking({.tolerance = 2}), {0.5, 1, 2, 0.99}) == (Marked{1, 2}));
    check_bad_indicators(EquidistributionMarking({.tolerance = 1}));
    for (double tolerance : {0.0, -1.0, std::nan(""), std::numeric_limits<double>::infinity()})
        CHECK_THROWS(std::invalid_argument, EquidistributionMarking({.tolerance = tolerance}),
                     "EquidistributionMarking::Settings::tolerance must be set");
}

int main() {
    test_dorfler();
    test_maximum();
    test_fixed_fraction();
    test_equidistribution();
    CHECK(DorflerMarking().name() == "dorfler");
    return finish("marking");
}
