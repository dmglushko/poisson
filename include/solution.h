#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Solution {
    struct Info {
        double   time = 0;                      // total, in seconds
        uint32_t iterations = 0;
        double   residual = 0;                  // final relative residual
        std::vector<std::string> warnings;      // e.g. no convergence within the iteration limit
    };

    std::vector<double> u;                      // potential per node [V], same order as nodes
    Info                info;
};
