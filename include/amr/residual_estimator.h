#pragma once

#include "amr/error_estimator.h"

// The residual error estimator. For -div(eps grad u) + c u = f with u_h piecewise linear:
//
//   eta_K^2 = h_K^2 / eps_K  ||f - c_K u_h||_K^2
//           + sum over the interior faces F of K of        h_F / (2 eps_F) ||J_F||_F^2
//           + sum over the Neumann and Robin faces F of K of   h_F / eps_K ||R_F||_F^2
//
// - h_K is the longest edge of K, h_F the longest edge of F.
// - div(eps grad u_h) = 0 inside a tet (eps constant, u_h linear), so the element residual
//   is f - c u_h. Point sources do not enter it (their L2 norm is infinite); see
//   PointSourceRule.
// - J_F = (eps_K grad u_h|K - eps_L grad u_h|L) . n, n pointing out of K: the jump of the
//   normal flux across the face to tet L. eps_F = max(eps_K, eps_L).
// - R_F = g - alpha u_h - eps_K grad u_h . n on a Neumann (alpha = 0) or Robin face. An
//   outer face without a boundary element is natural: g = 0, alpha = 0. J_F is constant and
//   R_F linear on a face, so their norms are computed in closed form.
// - Dirichlet faces contribute nothing, nor do faces that touch a metal tet (the metal is
//   an equipotential; its surface carries whatever charge it needs). An interior face with
//   a Neumann or Robin boundary element is treated as an ordinary interior face (warning).
//
// The weights 1/eps make the estimate robust when eps jumps between materials
// (Bernardi-Verfuerth); it estimates the energy norm ||eps^(1/2) grad (u - u_h)||.
//
// Details (in Info::details): element_part, jump_part, boundary_part, the square roots of
// the sums of each kind of term (their squares add up to total^2).
class ResidualEstimator final : public ErrorEstimator {
public:
    struct Settings {
        // Exact degree of the tet rule for ||f - c u_h||^2. f is arbitrary, so this is a
        // choice of accuracy; c u_h alone needs 2. The face terms are computed exactly.
        int             tet_degree = 4;
        PointSourceRule point_sources = PointSourceRule::exclude_patch;
        // Radius in nm when point_sources == PointSourceRule::exclude_radius (finite, > 0).
        double          point_source_exclusion_radius = 0.0;
    };

    // Throws std::invalid_argument for a degree outside [0, max_quadrature_degree].
    ResidualEstimator();   // default Settings
    explicit ResidualEstimator(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "residual"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                   std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const override;
    Settings settings_;
};
