#pragma once

#include "verification/verification_case.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Cases whose exact solution depends on one coordinate: layers of dielectrics and metals
// across a box (PlanarLayersCase) or around a centre (SphericalLayersCase). In a dielectric
// layer with screening length lambda (and screening on), the potential is a combination of
// exp(-+x / lambda) (planar) or exp(-+r / lambda) / r (spherical), and linear in x or a + b/r
// without screening. The coefficients follow from a small linear system: continuity of u and
// of the flux eps_r du/dn between dielectrics, u = V on a metal, zero net charge on a
// floating metal, and the two boundary conditions.

struct VerificationLayer {
    std::string name;                     // material name: a letter, then letters, digits or '_'
    double      thickness = 1;            // nm
    double      eps_r = 1;
    double      screening_length = 0;     // nm; 0 = no screening
    bool        metal = false;            // one potential throughout the layer
    std::optional<double> voltage;        // metal only: prescribed potential [V]; empty = floating (no net charge)
};

// A box [0, L] x [0, width] x [0, height] with the layers along x from x = 0. The faces
// x = 0 and x = L carry `left` and `right`; the four side faces are Neumann 0. A metal layer
// at either end needs Neumann 0 on its end face.
class PlanarLayersCase final : public VerificationCase {
public:
    struct Settings {
        std::vector<VerificationLayer>  layers;
        MeshProperties::BoundaryCondition left{MeshProperties::BcType::Dirichlet, 0, std::nullopt};
        MeshProperties::BoundaryCondition right{MeshProperties::BcType::Dirichlet, 1, std::nullopt};
        double width = 1;                 // nm, along y
        double height = 1;                // nm, along z
        bool   screened = true;
    };

    // Throws std::invalid_argument for invalid settings or a problem without a unique solution.
    explicit PlanarLayersCase(Settings settings);
    ~PlanarLayersCase() override;

    std::string_view name() const override { return "planar_layers"; }
    const Settings& settings() const { return settings_; }

    struct Solution;

private:
    double                exact_u(const std::array<double, 3>& x, std::size_t material) const override;
    std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const override;
    std::vector<std::string> geometry_mismatches(const MeshProperties& mesh,
                                                 const std::vector<std::optional<std::size_t>>& material) const override;

    Settings                  settings_;
    std::unique_ptr<Solution> solution_;
};

// Concentric layers outward from `inner_radius`: a ball first if inner_radius is 0, else a
// shell whose inner sphere carries `inner`. The outer sphere carries `outer`. An optional
// point charge sits at the centre (ball only, in a dielectric).
class SphericalLayersCase final : public VerificationCase {
public:
    struct Settings {
        std::vector<VerificationLayer> layers;
        double inner_radius = 0;          // nm
        std::optional<MeshProperties::BoundaryCondition> inner;   // required when inner_radius > 0
        MeshProperties::BoundaryCondition outer{MeshProperties::BcType::Dirichlet, 0, std::nullopt};
        int32_t charge = 0;               // elementary charges at the centre
        bool    screened = true;
    };

    explicit SphericalLayersCase(Settings settings);
    ~SphericalLayersCase() override;

    std::string_view name() const override { return "spherical_layers"; }
    const Settings& settings() const { return settings_; }

    struct Solution;

private:
    double                exact_u(const std::array<double, 3>& x, std::size_t material) const override;
    std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const override;
    std::vector<std::string> geometry_mismatches(const MeshProperties& mesh,
                                                 const std::vector<std::optional<std::size_t>>& material) const override;

    Settings                  settings_;
    std::unique_ptr<Solution> solution_;
};
