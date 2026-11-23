#pragma once

#include "mesh_properties.h"
#include "verification/verification_case.h"

#include <cstdint>
#include <string>
#include <vector>

// The error of a P1 nodal field against the exact solution of a case, on the field's own
// mesh. Metal tets are skipped (no equation is solved there; their potential is checked
// separately), and so are quadrature points and nodes closer than the exclusion radius to a
// trapped charge, where P1 cannot follow 1/r.
struct ErrorEvaluationResult {
    struct Info {
        double                   time = 0;
        std::vector<std::string> warnings;
        double   exclusion_radius = 0;        // nm, as used
        double   l2_error = 0;                // ||u_h - u||_L2 over the measured region [V nm^1.5]
        double   h1_error = 0;                // |u_h - u|_H1 seminorm [V nm^0.5]
        double   max_error = 0;               // max |u_h - u| over the measured non-metal nodes [V]
        double   l2_exact = 0;                // ||u||_L2 over the same region, for relative errors
        double   h1_exact = 0;                // |u|_H1 over the same region
        double   metal_potential_error = 0;   // max |u_h - u| over nodes of metal tets [V]
        double   measured_volume = 0;         // nm^3
        uint64_t excluded_points = 0;         // quadrature points inside an exclusion ball
        uint64_t metal_elements = 0;          // skipped metal tets
    };

    std::vector<double> element_l2_squared;   // per tet: integral of (u_h - u)^2 over its measured part
    std::vector<double> element_h1_squared;   // per tet: integral of |grad(u_h - u)|^2 over its measured part
    std::vector<double> node_error;           // per node: u_h - u; NaN where not measured
    Info                info;
};

class ErrorEvaluator {
public:
    struct Settings {
        double exclusion_radius = -1;   // nm; negative = the case's default_exclusion_radius
        int    quadrature_degree = 4;   // exactness degree of the tet rule, 0..max_quadrature_degree
    };

    ErrorEvaluator() = default;
    explicit ErrorEvaluator(Settings settings);   // throws std::invalid_argument for an invalid degree

    const Settings& settings() const { return settings_; }

    // u: one finite value per node. Every mesh material must be a material of the case (by
    // name). Throws std::invalid_argument if the sizes differ, a value is not finite, or a
    // material is missing.
    ErrorEvaluationResult evaluate(const MeshProperties& mesh, const std::vector<double>& u, const VerificationCase& c) const;

private:
    Settings settings_{};
};
