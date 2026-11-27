#pragma once

#include "amr/error_estimator.h"

// The trajectory-displacement estimator of Mitchell and Villarrubia ("SEM simulation with
// adaptive mesh refinement", Secs. 3.3-3.4). It ranks tets by their contribution to the
// error in an electron's landing position rather than by the energy norm of the potential
// error: a field error moves an electron in proportion to the field error itself, not to its
// square, and a larger tet is crossed by more trajectories, for longer. "Statistical" refers
// to that averaging over typical trajectories; no electrons are traced and no extra global
// solve is done.
//
//   eta_K = scale * | integral over K of grad e_K |       (Eq. 12-14 with h_K^3 = |K|)
//   total = sum of eta_K                                   (IndicatorSum::values, Eq. 13)
//
// e_K estimates u - u_h on K by the Bank-Weiser local problem (Eqs. 6-7), solved in the six
// quadratic edge bubbles 4 lambda_i lambda_j of K (zero at the vertices, so it is well posed):
//
//   integral over K of eps_K grad e_K . grad v + c_K e_K v  +  sum over Robin faces of alpha e_K v
//     = integral over K of (f - c_K u_h) v  +  sum over the faces F of K of integral over F of g_F v
//
//   interior face to tet L:  g_F = (eps_L grad u_h|L - eps_K grad u_h|K) . n / 2  (half the flux mismatch)
//   Neumann / Robin face:    g_F = g - alpha u_h - eps_K grad u_h . n               (the whole residual)
//   outer face, no element:  natural, g = 0, alpha = 0
//   Dirichlet face:          e_K = 0 there (the bubbles of its three edges are dropped)
//   face to a metal:         e_K = 0 if the metal has a prescribed voltage; otherwise skipped
//                            (a floating metal's potential is part of the solution)
//
// with n pointing out of K. An interior face that carries a Neumann or Robin boundary element
// is treated as an ordinary interior face (warning), as in the residual estimator. Point
// charges are not in the load: PointSourceRule handles them as in every estimator.
//
// scale is the paper's problem-dependent factor eR/(4TV) (R a typical trajectory length,
// T the electrons' kinetic energy, V the volume the trajectories sample). It does not change
// what the relative rules (Dörfler, maximum, fixed fraction, relative_tolerance) do; with it
// set, total estimates the relative landing-position error, so absolute_tolerance and
// equidistribution's tolerance get a meaning. The paper validates the indicator only without
// screening; with screening the c terms are kept for consistency.
//
// Details (in Info::details): none.
class TrajectoryEstimator final : public ErrorEstimator {
public:
    struct Settings {
        // eR/(4TV): for potentials in V and lengths in nm, R in nm, T in eV, V in nm^3, it is
        // R / (4 T V) in 1/(V nm^2), and total is dimensionless. Finite and > 0.
        double scale = 1;
        // Exact degree of the tet rule for the integral of (f - c u_h) times a bubble.
        // f is arbitrary, so this is a choice of accuracy; c u_h alone needs 3. The matrix
        // and the face loads are computed exactly.
        int             tet_degree = 4;
        PointSourceRule point_sources = PointSourceRule::exclude_patch;
        // Radius in nm when point_sources == PointSourceRule::exclude_radius (finite, > 0).
        double          point_source_exclusion_radius = 0.0;
    };

    // Throws std::invalid_argument for a degree outside [0, max_quadrature_degree] or a
    // scale that is not finite and positive.
    TrajectoryEstimator();   // default Settings
    explicit TrajectoryEstimator(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "trajectory"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                   std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const override;
    Settings settings_;
};
