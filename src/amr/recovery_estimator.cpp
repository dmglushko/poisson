#include "amr/recovery_estimator.h"

#include "estimator_geometry.h"
#include "quadrature.h"

#include <cmath>

using namespace amr_detail;

RecoveryEstimator::RecoveryEstimator() : RecoveryEstimator(Settings{}) {}

RecoveryEstimator::RecoveryEstimator(Settings settings)
    : ErrorEstimator(settings.point_sources), settings_(settings) {
    check_degree(settings_.tet_degree, "RecoveryEstimator::Settings::tet_degree");
}

std::vector<double> RecoveryEstimator::indicators(const MeshProperties& mesh, const RefinementInput& input,
                                                  std::span<const uint8_t>, ErrorEstimate::Info&) const {
    const std::string who = name() + " estimator";
    const std::vector<TetGeometry> geometry = tet_geometry(mesh, who);
    const std::size_t count = mesh.elements.size();
    std::vector<Coefficients> material(mesh.materials.size());
    for (std::size_t m = 0; m < mesh.materials.size(); ++m) material[m] = coefficients(mesh.materials[m]);
    const bool flux = settings_.quantity == Quantity::flux;

    // The recovered quantity per tet: grad u_h or eps grad u_h.
    std::vector<Vec3> own(count);
    for (uint32_t k = 0; k < count; ++k) {
        own[k] = gradient(mesh, geometry[k], k, input.u);
        if (flux)
            for (double& v : own[k]) v *= material[mesh.elements[k].material].eps;
    }

    std::vector<double> eta(count, 0);
    std::vector<Vec3> recovered(mesh.nodes.size());
    std::vector<double> weight(mesh.nodes.size());
    const auto rule = tet_rule(settings_.tet_degree);

    // Average over the tets that `take` accepts, then measure those tets.
    const auto recover_and_measure = [&](const auto& take) {
        std::fill(recovered.begin(), recovered.end(), Vec3{0, 0, 0});
        std::fill(weight.begin(), weight.end(), 0.0);
        for (uint32_t k = 0; k < count; ++k) {
            if (!take(k)) continue;
            for (uint32_t node : mesh.elements[k].node) {
                for (int d = 0; d < 3; ++d) recovered[node][d] += geometry[k].volume * own[k][d];
                weight[node] += geometry[k].volume;
            }
        }
        for (std::size_t i = 0; i < recovered.size(); ++i)
            if (weight[i] > 0)
                for (double& v : recovered[i]) v /= weight[i];
        for (uint32_t k = 0; k < count; ++k) {
            if (!take(k)) continue;
            const auto& n = mesh.elements[k].node;
            const double eps = material[mesh.elements[k].material].eps;
            double sum = 0;
            for (const auto& q : rule) {
                Vec3 difference = {-own[k][0], -own[k][1], -own[k][2]};
                for (int i = 0; i < 4; ++i)
                    for (int d = 0; d < 3; ++d) difference[d] += q.barycentric[i] * recovered[n[i]][d];
                sum += q.weight * dot(difference, difference);
            }
            eta[k] = std::sqrt(geometry[k].volume * sum * (flux ? 1 / eps : eps));
        }
    };

    if (settings_.per_material) {
        for (std::size_t m = 0; m < mesh.materials.size(); ++m)
            if (!material[m].metal) recover_and_measure([&](uint32_t k) { return mesh.elements[k].material == m; });
    } else {
        recover_and_measure([&](uint32_t k) { return !material[mesh.elements[k].material].metal; });
    }
    return eta;
}
