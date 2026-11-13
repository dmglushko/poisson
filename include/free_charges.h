#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// The free charges (primary electrons), passed to the ray tracer.
// One entry holds exactly what one record of a .pri file holds.
struct FreeCharges {
    struct Electron {
        std::array<double, 3>  position;    // x, y, z in nm
        std::array<double, 3>  direction;   // direction of motion, not normalised
        double                 energy;      // kinetic energy in eV
        std::array<int32_t, 2> pixel;       // image pixel this electron belongs to
    };

    struct Info {
        double time = 0;                        // total, in seconds
        std::vector<std::string> warnings;
    };

    std::vector<Electron> electrons;
    Info                  info;
};
