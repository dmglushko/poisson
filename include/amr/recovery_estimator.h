#pragma once

#include "amr/error_estimator.h"

// The recovery (Zienkiewicz-Zhu) error estimator. grad u_h is constant per tet and jumps
// between tets; a smoother field is recovered at every node as the volume-weighted average
// of the tets around it, interpolated linearly in between, and the difference is measured:
//
//   quantity = gradient:  G at the nodes,  eta_K^2 = integral over K of eps_K |G - grad u_h|^2
//   quantity = flux:      Q at the nodes,  eta_K^2 = integral over K of |Q - eps_K grad u_h|^2 / eps_K
//
// Both are in the energy norm. The difference is linear on K, so the integral is computed
// exactly from its values at the four nodes. Across a material interface neither grad u nor
// eps grad u is continuous (the normal flux and the tangential gradient are), so averaging across it
// reports an error on every interface tet that refinement never removes. With
// per_material = true a node on an interface gets one average per material, from that
// material's tets only, and both quantities then give the same eta (eps is constant within
// a material). The quantity matters only with per_material = false.
//
// Metal tets are left out of the averages and get eta_K = 0. It does not use the source or
// the boundary conditions. No extra entries in Info::details.
class RecoveryEstimator final : public ErrorEstimator {
public:
    enum class Quantity { gradient, flux };

    struct Settings {
        Quantity        quantity = Quantity::flux;
        bool            per_material = true;
        PointSourceRule point_sources = PointSourceRule::exclude_patch;
        // Radius in nm when point_sources == PointSourceRule::exclude_radius (finite, > 0).
        double          point_source_exclusion_radius = 0.0;
    };

    RecoveryEstimator();   // default Settings
    explicit RecoveryEstimator(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "recovery"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                   std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const override;
    Settings settings_;
};
