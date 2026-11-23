#pragma once

#include "verification/verification_case.h"

#include <cstdint>
#include <string>
#include <vector>

// The input files of a case, in the formats the importer reads.
struct CaseInputsResult {
    struct Info {
        double                   time = 0;                       // seconds
        std::vector<std::string> warnings;
        std::string              case_name;
        uint32_t                 materials = 0;                  // number of materials
        uint32_t                 charges = 0;                    // number of trapped charges
        double                   default_exclusion_radius = 0;   // nm, from the problem
        double                   boundary_error_bound = 0;       // V, from the problem
    };

    std::string geometry;    // gmsh .geo script
    std::string materials;   // material .yaml, one "---" document per material
    std::string charges;     // .cha bytes: x, y, z float32 and q int32 per charge, little-endian
    Info        info;
};

CaseInputsResult make_case_inputs(const VerificationCase& c);
