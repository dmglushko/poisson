#include "verification/error_evaluator.h"

#include "case_support.h"
#include "common/quadrature.h"
#include "verification_mesh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace verification_detail;

ErrorEvaluator::ErrorEvaluator(Settings settings) : settings_(settings) {
    if (settings_.quadrature_degree < 0 || settings_.quadrature_degree > max_quadrature_degree)
        throw std::invalid_argument("ErrorEvaluator: quadrature_degree must be in [0, " +
                                    std::to_string(max_quadrature_degree) + "]");
    if (std::isnan(settings_.exclusion_radius) || std::isinf(settings_.exclusion_radius))
        throw std::invalid_argument("ErrorEvaluator: exclusion_radius must be finite");
}

ErrorEvaluationResult ErrorEvaluator::evaluate(const MeshProperties& mesh, const std::vector<double>& u,
                                               const VerificationCase& c) const {
    const auto start = std::chrono::steady_clock::now();
    if (u.size() != mesh.nodes.size())
        throw std::invalid_argument("ErrorEvaluator: the field has " + std::to_string(u.size()) + " values for " +
                                    std::to_string(mesh.nodes.size()) + " nodes");
    for (const double v : u)
        if (!std::isfinite(v)) throw std::invalid_argument("ErrorEvaluator: the field has non-finite values");
    const std::vector<std::size_t> material = case_materials(mesh, c);
    const VerificationProblem& problem = c.problem();
    const double r0 = settings_.exclusion_radius < 0 ? problem.default_exclusion_radius : settings_.exclusion_radius;
    const auto near_charge = [&](const std::array<double, 3>& x) {
        for (const auto& charge : problem.charges)
            if (distance(x, charge.position) < r0) return true;
        return false;
    };
    const auto rule = tet_rule(settings_.quadrature_degree);

    ErrorEvaluationResult result;
    auto& info = result.info;
    info.exclusion_radius = r0;
    result.element_l2_squared.assign(mesh.elements.size(), 0);
    result.element_h1_squared.assign(mesh.elements.size(), 0);

    double l2 = 0, h1 = 0, l2_exact = 0, h1_exact = 0;
    constexpr std::size_t none = static_cast<std::size_t>(-1);
    std::vector<std::size_t> node_material(mesh.nodes.size(), none);
    std::array<std::array<double, 3>, 4> grad{};
    for (std::size_t e = 0; e < mesh.elements.size(); ++e) {
        const auto& element = mesh.elements[e];
        const std::size_t m = material[element.material];
        const bool metal = problem.materials[m].is_metal;
        for (const uint32_t n : element.node)
            if (node_material[n] == none || metal) node_material[n] = m;
        if (metal) {
            ++info.metal_elements;
            continue;
        }

        double volume = 0;
        p1_gradients(mesh, element, grad, volume);
        std::array<double, 3> grad_h{0, 0, 0};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k) grad_h[k] += u[element.node[i]] * grad[i][k];

        double el_l2 = 0, el_h1 = 0;
        for (const auto& q : rule) {
            std::array<double, 3> x{0, 0, 0};
            double u_h = 0;
            for (int i = 0; i < 4; ++i) {
                const auto& p = mesh.nodes[element.node[i]];
                for (int k = 0; k < 3; ++k) x[k] += q.barycentric[i] * p[k];
                u_h += q.barycentric[i] * u[element.node[i]];
            }
            if (near_charge(x)) {
                ++info.excluded_points;
                continue;
            }
            const double w = q.weight * volume;
            const double ue = c.u(x, m);
            const auto ge = c.grad_u(x, m);
            if (!std::isfinite(ue) || !std::isfinite(ge[0]) || !std::isfinite(ge[1]) || !std::isfinite(ge[2])) {
                // Only a point exactly on a charge (exclusion radius 0) may do this.
                bool on_charge = false;
                for (const auto& charge : problem.charges) on_charge = on_charge || verification_detail::distance(x, charge.position) == 0;
                if (!on_charge)
                    throw std::runtime_error("ErrorEvaluator: the exact solution of case '" + std::string(c.name()) +
                                             "' is not finite at a quadrature point of tet " + std::to_string(e));
                ++info.excluded_points;
                continue;
            }
            double dg = 0, ge2 = 0;
            for (int k = 0; k < 3; ++k) {
                dg += (grad_h[k] - ge[k]) * (grad_h[k] - ge[k]);
                ge2 += ge[k] * ge[k];
            }
            el_l2 += w * (u_h - ue) * (u_h - ue);
            el_h1 += w * dg;
            l2_exact += w * ue * ue;
            h1_exact += w * ge2;
            info.measured_volume += w;
        }
        result.element_l2_squared[e] = el_l2;
        result.element_h1_squared[e] = el_h1;
        l2 += el_l2;
        h1 += el_h1;
    }

    result.node_error.assign(mesh.nodes.size(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t n = 0; n < mesh.nodes.size(); ++n) {
        if (node_material[n] == none || near_charge(mesh.nodes[n])) continue;
        const double ue = c.u(mesh.nodes[n], node_material[n]);
        if (!std::isfinite(ue)) continue;
        const double err = u[n] - ue;
        result.node_error[n] = err;
        if (problem.materials[node_material[n]].is_metal) info.metal_potential_error = std::max(info.metal_potential_error, std::abs(err));
        else info.max_error = std::max(info.max_error, std::abs(err));
    }

    info.l2_error = std::sqrt(l2);
    info.h1_error = std::sqrt(h1);
    info.l2_exact = std::sqrt(l2_exact);
    info.h1_exact = std::sqrt(h1_exact);
    if (!problem.charges.empty() && r0 == 0)
        info.warnings.push_back("no exclusion radius around the charges: the exact H1 seminorm is infinite there, so "
                                "h1_error, h1_exact and max_error depend on where the quadrature points and nodes fall");
    if (info.measured_volume == 0) info.warnings.push_back("nothing was measured: every point is in a metal or an exclusion ball");
    if (problem.boundary_error_bound > 0)
        info.warnings.push_back("the boundary conditions only approximate the exact solution, by up to " +
                                std::to_string(problem.boundary_error_bound) + " V; errors below that are not meaningful");
    info.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
