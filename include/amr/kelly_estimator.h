#pragma once

#include "amr/error_estimator.h"

// The Kelly error estimator: the face terms of the residual estimator (residual_estimator.h)
// without the element residual,
//
//   eta_K^2 = sum over the interior faces F of K of        h_F / (2 eps_F) ||J_F||_F^2
//           + sum over the Neumann and Robin faces F of K of   h_F / eps_K ||R_F||_F^2,
//
// with the same J_F, R_F, weights and rules for Dirichlet and metal faces. It does not use
// the source, so input.source may be empty.
//
// Details (in Info::details): jump_part, boundary_part.
class KellyEstimator final : public ErrorEstimator {
public:
    struct Settings {
        // Exact degree of the triangle rule for ||R_F||^2: 2 is exact for constant g and alpha.
        int face_degree = 2;
        PointSourceRule point_sources = PointSourceRule::exclude_patch;
    };

    // Throws std::invalid_argument for a degree outside [0, max_quadrature_degree].
    KellyEstimator();   // default Settings
    explicit KellyEstimator(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "kelly"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                   std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const override;
    bool needs_source() const override { return false; }
    Settings settings_;
};
