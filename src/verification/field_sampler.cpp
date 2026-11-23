#include "verification/field_sampler.h"

#include "common/mesh_locator.h"
#include "verification_mesh.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace verification_detail;

std::vector<std::array<double, 3>> FieldSampler::line(const std::array<double, 3>& from, const std::array<double, 3>& to,
                                                      uint32_t n) {
    if (n < 2) throw std::invalid_argument("FieldSampler::line: at least 2 points are needed");
    std::vector<std::array<double, 3>> points(n);
    for (uint32_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / (n - 1);
        for (int k = 0; k < 3; ++k) points[i][k] = i + 1 == n ? to[k] : from[k] + t * (to[k] - from[k]);
    }
    return points;
}

SamplingResult FieldSampler::sample(const MeshProperties& mesh, const std::vector<double>& u,
                                    const std::vector<std::array<double, 3>>& points, const VerificationCase* exact) const {
    const auto start = std::chrono::steady_clock::now();
    if (u.size() != mesh.nodes.size())
        throw std::invalid_argument("FieldSampler: the field has " + std::to_string(u.size()) + " values for " +
                                    std::to_string(mesh.nodes.size()) + " nodes");
    for (const double v : u)
        if (!std::isfinite(v)) throw std::invalid_argument("FieldSampler: the field has non-finite values");
    for (const auto& p : points)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
            throw std::invalid_argument("FieldSampler: sample points must be finite");
    std::vector<std::size_t> material;
    if (exact) material = case_materials(mesh, *exact);

    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    SamplingResult result;
    PointSamples& s = result.samples;
    const std::size_t n = points.size();
    s.points = points;
    s.found.assign(n, 0);
    s.element.assign(n, MeshProperties::no_neighbour);
    s.material.assign(n, PointSamples::no_material);
    s.u.assign(n, nan);
    s.grad.assign(n, {nan, nan, nan});
    if (exact) {
        s.u_exact.assign(n, nan);
        s.grad_exact.assign(n, {nan, nan, nan});
    }

    const MeshLocator locator(mesh);
    std::optional<uint32_t> hint;
    std::array<std::array<double, 3>, 4> grad{};
    for (std::size_t i = 0; i < n; ++i) {
        const auto location = hint ? locator.locate_near(points[i], *hint) : locator.locate(points[i]);
        if (!location) {
            ++result.info.outside;
            continue;
        }
        hint = location->element;
        const auto& element = mesh.elements[location->element];
        s.found[i] = 1;
        s.element[i] = location->element;
        s.material[i] = element.material;
        double value = 0;
        for (int j = 0; j < 4; ++j) value += location->barycentric[j] * u[element.node[j]];
        s.u[i] = value;
        double volume = 0;
        p1_gradients(mesh, element, grad, volume);
        std::array<double, 3> g{0, 0, 0};
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) g[k] += u[element.node[j]] * grad[j][k];
        s.grad[i] = g;
        if (exact) {
            s.u_exact[i] = exact->u(points[i], material[element.material]);
            s.grad_exact[i] = exact->grad_u(points[i], material[element.material]);
        }
    }
    if (result.info.outside > 0)
        result.info.warnings.push_back(std::to_string(result.info.outside) + " of " + std::to_string(n) +
                                       " sample points lie outside the mesh");
    result.info.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
