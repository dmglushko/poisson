#pragma once

#include <array>
#include <cstdint>
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

    std::vector<Electron> electrons;
};
