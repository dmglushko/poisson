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
        PointSourceRule point_sources = PointSourceRule::exclude_patch;
    };

    KellyEstimator();   // default Settings
    explicit KellyEstimator(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "kelly"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                   std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const override;
    Settings settings_;
};
