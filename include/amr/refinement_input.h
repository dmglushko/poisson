#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

// What an error estimator needs to know about one solve: the FEM solution and the
// right-hand side of the problem the FEM solved. It says nothing about how the analytical
// part is represented; whoever builds the right-hand side wraps it in `source`.
//
// The equation, per tet K with eps_K = eps_r and c_K = eps_r / screening_length^2 when
// mesh.screened && screening_length > 0 (0 otherwise):
//     -div(eps_K grad u) + c_K u = f + sum of point sources.
struct RefinementInput {
    struct PointSource {                          // a delta in the right-hand side
        std::array<double, 3>   position;         // x, y, z in nm
        double                  value;            // its weight, in the units of f times a volume
        std::optional<uint32_t> element;          // the tet that contains it, if known
    };

    // u_h at every node, indexed like MeshProperties::nodes, Dirichlet nodes included (not
    // SolverResult::x, which has one value per row of A).
    std::span<const double> u;

    // The smooth part f of the right-hand side at a point x (nm) inside `element`: the same
    // f that Assembly integrates, with the same units and sign. The estimator adds -c u_h
    // itself. The element is passed because f may depend on the material (with screening,
    // f = rho_b - c w). Only estimators that use f call it (the residual estimator);
    // the others accept an empty function.
    std::function<double(const std::array<double, 3>& x, uint32_t element)> source;

    // The charges still deposited as deltas. Empty once every charge is a smooth cloud.
    std::vector<PointSource> point_sources;
};
