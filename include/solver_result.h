#pragma once

#include <cstdint>
#include <string>
#include <vector>

// What the solver returns: the solution x of A x = f (output data) and how the run went (Info).
struct SolverResult {
    struct Info {
        double   time = 0;                      // total, in seconds
        uint32_t iterations = 0;
        double   residual = 0;                  // final relative residual
        std::vector<std::string> warnings;      // e.g. no convergence within the iteration limit
    };

    std::vector<double> x;                      // one value per row of A
    Info                info;
};
