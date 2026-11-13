#pragma once

#include <string>
#include <vector>

// The potential at every node.
struct Solution {
    struct Info {
        double time = 0;                        // total, in seconds
        std::vector<std::string> warnings;
    };

    std::vector<double> u;                      // potential per node [V], same order as nodes
    Info                info;
};
