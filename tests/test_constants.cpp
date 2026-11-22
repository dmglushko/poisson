#include "common/constants.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

int main() {
    using namespace poisson::constants;

    int failures = 0;
    const auto check_close = [&](const char* label, double actual, double expected, double rel_tol) {
        const double err = std::abs(actual - expected);
        if (!(err <= rel_tol * std::abs(expected))) {
            std::cerr << "FAILED " << label << ": actual=" << actual << ", expected=" << expected << "\n";
            ++failures;
        }
    };

    check_close("nm_to_m * m_to_nm", nm_to_m * m_to_nm, 1.0, 1e-15);
    check_close("e_over_eps0_nm", e_over_eps0_nm, 18.09512817, 1e-7);
    check_close("coulomb_constant_nm", coulomb_constant_nm, 1.439964548, 1e-7);
    check_close("4pi * coulomb_constant_nm", 4.0 * std::numbers::pi * coulomb_constant_nm, e_over_eps0_nm, 1e-15);
    check_close("e_over_me", e_over_me, 1.75882000838e11, 1e-9);

    if (failures > 0) return EXIT_FAILURE;
    std::cout << "all constants tests passed\n";
    return EXIT_SUCCESS;
}
