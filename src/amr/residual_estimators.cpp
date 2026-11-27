// The residual and the Kelly estimators: one computation of the residual terms (element
// residual, flux jumps, boundary residuals); Kelly takes the face terms only.

#include "amr/kelly_estimator.h"
#include "amr/residual_estimator.h"

#include "estimator_geometry.h"
#include "common/quadrature.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace amr_detail;

namespace {

// The squares of the three kinds of terms, per tet.
struct Terms {
    std::vector<double> element, jump, boundary;
};

double root_of_sum(const std::vector<double>& squares) {
    double largest = 0;
    for (double s : squares) largest = std::max(largest, s);
    if (largest == 0) return 0;
    double scaled = 0;
    for (double s : squares) scaled += s / largest;
    return std::sqrt(largest) * std::sqrt(scaled);
}

// with_element: the residual estimator (element residual); otherwise Kelly.
Terms residual_terms(const MeshProperties& mesh, const RefinementInput& input, std::span<const uint8_t> excluded,
                     bool with_element, int tet_degree, const std::string& who, ErrorEstimate::Info& info) {
    const std::vector<TetGeometry> geometry = tet_geometry(mesh, who);
    const std::size_t count = mesh.elements.size();
    std::vector<Coefficients> material(mesh.materials.size());
    for (std::size_t m = 0; m < mesh.materials.size(); ++m) material[m] = coefficients(mesh.materials[m], mesh.screened);
    std::vector<Vec3> grad(count);
    for (uint32_t k = 0; k < count; ++k) grad[k] = gradient(mesh, geometry[k], k, input.u);
    const auto coef = [&](uint32_t k) -> const Coefficients& { return material[mesh.elements[k].material]; };

    Terms terms{std::vector<double>(count, 0), std::vector<double>(count, 0), std::vector<double>(count, 0)};

    // Element residual f - c u_h: quadrature when f is given; when input.source is empty
    // (f = 0), skip unscreened tets and integrate (-c u_h)^2 in closed form otherwise.
    if (with_element) {
        if (input.source) {
            const auto rule = tet_rule(tet_degree);
            for (uint32_t k = 0; k < count; ++k) {
                const Coefficients& c = coef(k);
                if (c.metal || excluded[k]) continue;
                const auto& n = mesh.elements[k].node;
                double sum = 0;
                for (const auto& q : rule) {
                    const Vec3 x = point_at(mesh, k, q.barycentric);
                    const double f = input.source(x, k);
                    if (!std::isfinite(f))
                        throw std::invalid_argument(who + ": the source is " + std::to_string(f) + " at (" +
                                                    std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " +
                                                    std::to_string(x[2]) + ") in tet " + std::to_string(k));
                    double u_h = 0;
                    for (int i = 0; i < 4; ++i) u_h += q.barycentric[i] * input.u[n[i]];
                    const double r = f - c.c * u_h;
                    sum += q.weight * r * r;
                }
                const double h = geometry[k].diameter;
                terms.element[k] = h * h / c.eps * geometry[k].volume * sum;
            }
        } else {
            for (uint32_t k = 0; k < count; ++k) {
                const Coefficients& c = coef(k);
                if (c.metal || excluded[k] || c.c == 0) continue;
                const auto& n = mesh.elements[k].node;
                std::array<double, 4> cu{};
                for (int i = 0; i < 4; ++i) cu[i] = c.c * input.u[n[i]];
                const double h = geometry[k].diameter;
                terms.element[k] = h * h / c.eps * integral_of_square(cu, geometry[k].volume);
            }
        }
    }

    uint32_t natural_interior = 0;
    for (const Face& face : faces(mesh)) {
        const uint32_t k = face.element;
        const Coefficients& ck = coef(k);
        const std::optional<MeshProperties::BoundaryCondition> bc =
            face.boundary_condition ? std::optional(mesh.boundary_conditions[*face.boundary_condition]) : std::nullopt;
        if (ck.metal) continue;
        if (bc && bc->type == MeshProperties::BcType::Dirichlet) continue;

        // n out of k: -grad lambda of the node opposite the face, normalised.
        const Vec3& g = geometry[k].grad_lambda[face.local];
        const double g_norm = norm(g);
        const Vec3 normal{-g[0] / g_norm, -g[1] / g_norm, -g[2] / g_norm};
        const double area = 0.5 * norm(cross(sub(mesh.nodes[face.node[1]], mesh.nodes[face.node[0]]),
                                             sub(mesh.nodes[face.node[2]], mesh.nodes[face.node[0]])));
        const double h_f = triangle_diameter(mesh, face.node);

        if (face.other) {
            const uint32_t l = *face.other;
            const Coefficients& cl = coef(l);
            if (cl.metal) continue;
            if (bc) ++natural_interior;
            const double jump = ck.eps * dot(grad[k], normal) - cl.eps * dot(grad[l], normal);
            const double half = 0.5 * h_f / std::max(ck.eps, cl.eps) * jump * jump * area;
            terms.jump[k] += half;
            terms.jump[l] += half;
            continue;
        }

        // Outer face: Neumann, Robin, or natural (no boundary element: g = 0). R_F is linear
        // on the face (g and alpha constant), so ||R_F||^2 is exact from its vertex values.
        const double g_value = bc ? bc->value : 0;
        const double alpha = bc && bc->type == MeshProperties::BcType::Robin ? bc->alpha.value_or(0) : 0;
        const double flux = ck.eps * dot(grad[k], normal);
        std::array<double, 3> r{};
        for (int i = 0; i < 3; ++i) r[i] = g_value - alpha * input.u[face.node[i]] - flux;
        terms.boundary[k] += h_f / ck.eps * integral_of_square(r, area);
    }
    if (natural_interior > 0)
        info.warnings.push_back(std::to_string(natural_interior) +
                                " interior faces carry a Neumann or Robin boundary element; they were treated as "
                                "ordinary interior faces");
    return terms;
}

} // namespace

ResidualEstimator::ResidualEstimator() : ResidualEstimator(Settings{}) {}

ResidualEstimator::ResidualEstimator(Settings settings)
    : ErrorEstimator(settings.point_sources), settings_(settings) {
    check_degree(settings_.tet_degree, "ResidualEstimator::Settings::tet_degree");
}

std::vector<double> ResidualEstimator::indicators(const MeshProperties& mesh, const RefinementInput& input,
                                                  std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const {
    Terms terms = residual_terms(mesh, input, excluded, true, settings_.tet_degree, name() + " estimator", info);
    std::vector<double> eta(terms.element.size(), 0);
    for (std::size_t k = 0; k < eta.size(); ++k) {
        if (excluded[k]) {
            terms.element[k] = 0;
            terms.jump[k] = 0;
            terms.boundary[k] = 0;
            continue;
        }
        eta[k] = std::sqrt(terms.element[k] + terms.jump[k] + terms.boundary[k]);
    }
    info.details["element_part"] = root_of_sum(terms.element);
    info.details["jump_part"] = root_of_sum(terms.jump);
    info.details["boundary_part"] = root_of_sum(terms.boundary);
    return eta;
}

KellyEstimator::KellyEstimator() : KellyEstimator(Settings{}) {}

KellyEstimator::KellyEstimator(Settings settings) : ErrorEstimator(settings.point_sources), settings_(settings) {}

std::vector<double> KellyEstimator::indicators(const MeshProperties& mesh, const RefinementInput& input,
                                               std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const {
    Terms terms = residual_terms(mesh, input, excluded, false, 0, name() + " estimator", info);
    std::vector<double> eta(terms.jump.size(), 0);
    for (std::size_t k = 0; k < eta.size(); ++k) {
        if (excluded[k]) {
            terms.jump[k] = 0;
            terms.boundary[k] = 0;
            continue;
        }
        eta[k] = std::sqrt(terms.jump[k] + terms.boundary[k]);
    }
    info.details["jump_part"] = root_of_sum(terms.jump);
    info.details["boundary_part"] = root_of_sum(terms.boundary);
    return eta;
}
