#pragma once

#include "mesh_properties.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Problems with a known exact solution, for checking the pipeline against them
// (src/verification/architecture.md). Lengths are in nm, potentials in V, charges in
// elementary charges, as everywhere else.

// Everything the mesh of a case must contain.
struct VerificationProblem {
    // gmsh .geo script (OpenCASCADE kernel). Its physical volumes are named after `materials`
    // (with " <voltage>V" appended for a metal with a prescribed voltage) and its physical
    // surfaces after the importer's boundary condition names ("<u>V", "D<g>", "R<alpha>,<g>").
    // Mesh sizes come from the constants h_max, h_min, h_charge, r_fine and r_coarse, which
    // the caller sets with `gmsh -setnumber` (see VerificationCase::mesh_size_constants).
    std::string geometry;
    std::vector<MeshProperties::Material>          materials;             // one per physical volume
    std::vector<MeshProperties::BoundaryCondition> boundary_conditions;   // the distinct conditions on the surfaces
    std::vector<MeshProperties::TrappedCharge>     charges;               // positions are exact float32 values (.cha)
    bool                                           screened = true;

    // Errors are not measured closer than this to a charge by default [nm]; 0 without charges.
    double default_exclusion_radius = 0;
    // Bound on |u_exact| where a boundary condition only approximates the exact solution,
    // such as u = 0 on the far faces of a box around a charge [V]; 0 when every condition is exact.
    double boundary_error_bound = 0;
};

class VerificationCase {
public:
    virtual ~VerificationCase() = default;

    // The name used in configuration files, e.g. "planar_layers".
    virtual std::string_view name() const = 0;

    const VerificationProblem& problem() const { return problem_; }

    // Exact potential [V] and its gradient [V/nm] at x inside the volume
    // problem().materials[material]. The formula of that volume is used even slightly outside
    // it, so a faceted mesh of a curved interface still gets the right side's formula. Throws
    // std::invalid_argument for a material index out of range or a non-finite x.
    double                u(const std::array<double, 3>& x, std::size_t material) const;
    std::array<double, 3> grad_u(const std::array<double, 3>& x, std::size_t material) const;

    // Index of the material named `name` in problem().materials; throws std::invalid_argument
    // if there is none.
    std::size_t material_index(std::string_view name) const;


    // The .geo constants that set the mesh sizes, with their defaults in the generated script.
    static constexpr std::array<std::string_view, 5> mesh_size_constants{"h_max", "h_min", "h_charge", "r_fine", "r_coarse"};

protected:
    // Set by the derived constructor after validating its Settings.
    VerificationProblem problem_;

private:
    virtual double                exact_u(const std::array<double, 3>& x, std::size_t material) const = 0;
    virtual std::array<double, 3> exact_grad(const std::array<double, 3>& x, std::size_t material) const = 0;
    // Differences between where the mesh puts its tets and boundary conditions and where the
    // case has them. `material[m]` is the case material of mesh material m, if any.
    virtual std::vector<std::string> geometry_mismatches(const MeshProperties& mesh,
                                                         const std::vector<std::optional<std::size_t>>& material) const = 0;

    friend std::vector<std::string> mesh_mismatches(const MeshProperties& mesh, const VerificationCase& c);
};

// Differences between an imported mesh and the problem of `c`, empty when they match:
// - the screened flag, every mesh material against the case material of its name (metal,
//   voltage, and for dielectrics eps_r and screening length), the set of boundary
//   conditions, and the trapped charges;
// - geometry: every tet in its material's region and every boundary triangle carrying the
//   condition the case has there (exactly on flat faces; on spheres see
//   src/verification/architecture.md), every boundary triangle an outer face, and every
//   outer face with a condition.
std::vector<std::string> mesh_mismatches(const MeshProperties& mesh, const VerificationCase& c);
