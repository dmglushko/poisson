#pragma once

#include "verification/verification_case.h"

#include <array>
#include <cstdint>

// Cases solved with image charges. The image charges are only part of the formula; the
// mesh holds the one real charge. With G(r) = exp(-r / lambda) / r (1 / r without
// screening) and K = constants::coulomb_constant_nm, a charge q in a uniform medium eps_r
// gives u = q K G(r) / eps_r.

// A ball of radius R, one dielectric without screening, u = 0 on the sphere, and a charge q
// at `position` (|position| = d < R). The image q' = -q R / d sits at position R^2 / d^2:
// u = K / eps_r (q / |x - p| + q' / |x - p'|). Exact.
class GroundedBallImageCase final : public VerificationCase {
public:
    struct Settings {
        double                radius = 10;    // nm
        double                eps_r = 1;
        int32_t               charge = -1;
        std::array<double, 3> position{3, 0, 0};
    };

    explicit GroundedBallImageCase(Settings settings);

    std::string_view name() const override { return "grounded_ball_image"; }
    const Settings& settings() const { return settings_; }

private:
    double                exact_u(const std::array<double, 3>& x, std::size_t material) const override;
    std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const override;
    std::optional<MeshProperties::BoundaryCondition> condition_on(const std::array<std::array<double, 3>, 3>& triangle) const override;
    bool fits_region(const std::array<std::array<double, 3>, 4>& tet, std::size_t material) const override;

    Settings              settings_;
    std::array<double, 3> charge_position_{};   // as stored in .cha
    std::array<double, 3> image_position_{};
    double                image_charge_ = 0;
};

// The box [-W, W]^2 x [0, H] in one medium, a charge q at (0, 0, d), and the plane z = 0
// either u = 0 ("dirichlet": image -q at (0, 0, -d)) or Neumann 0 ("neumann": image +q).
// The other five faces are u = 0, which the formula only approximates;
// problem().boundary_error_bound says by how much (tiny for a box many screening lengths wide).
class HalfSpaceImageCase final : public VerificationCase {
public:
    enum class Plane { Dirichlet, Neumann };

    struct Settings {
        double  eps_r = 1;
        double  screening_length = 1;     // nm; 0 = no screening
        int32_t charge = -1;
        double  distance = 2;             // d, nm
        double  half_width = 20;          // W, nm
        double  height = 20;              // H, nm
        Plane   plane = Plane::Dirichlet;
        bool    screened = true;
    };

    explicit HalfSpaceImageCase(Settings settings);

    std::string_view name() const override { return "half_space_image"; }
    const Settings& settings() const { return settings_; }

private:
    double                exact_u(const std::array<double, 3>& x, std::size_t material) const override;
    std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const override;
    std::optional<MeshProperties::BoundaryCondition> condition_on(const std::array<std::array<double, 3>, 3>& triangle) const override;
    bool fits_region(const std::array<std::array<double, 3>, 4>& tet, std::size_t material) const override;

    Settings settings_;
    double   d_ = 0;                      // charge height as stored in .cha
    double   k_ = 0;                      // 1 / lambda, or 0
};

// The box [-W, W]^2 x [-H, H]: "upper" (eps_upper) for z > 0, "lower" (eps_lower) for z < 0,
// both with the same screening length, and a charge q at (0, 0, d) in the upper medium.
// With rho = (eps_upper - eps_lower) / (eps_upper + eps_lower):
//   z > 0: u = K q / eps_upper (G(r+) + rho G(r-)),  z < 0: u = 2 K q / (eps_upper + eps_lower) G(r+),
// r+- the distances to (0, 0, +-d). All faces are u = 0, which the formula only approximates
// (problem().boundary_error_bound).
class DielectricInterfaceCase final : public VerificationCase {
public:
    struct Settings {
        double  eps_upper = 1;
        double  eps_lower = 4;
        double  screening_length = 1;     // nm; 0 = no screening
        int32_t charge = -1;
        double  distance = 2;             // d, nm
        double  half_width = 20;          // W, nm
        double  half_height = 20;         // H, nm
        bool    screened = true;
    };

    explicit DielectricInterfaceCase(Settings settings);

    std::string_view name() const override { return "dielectric_interface"; }
    const Settings& settings() const { return settings_; }

private:
    double                exact_u(const std::array<double, 3>& x, std::size_t material) const override;
    std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const override;
    std::optional<MeshProperties::BoundaryCondition> condition_on(const std::array<std::array<double, 3>, 3>& triangle) const override;
    bool fits_region(const std::array<std::array<double, 3>, 4>& tet, std::size_t material) const override;

    Settings settings_;
    double   d_ = 0;
    double   k_ = 0;
};
