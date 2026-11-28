// The shared part of every estimator: ErrorEstimator::estimate.

#include "amr/error_estimator.h"

#include "common/mesh_locator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

ErrorEstimator::ErrorEstimator(PointSourceRule point_sources, double exclusion_radius, IndicatorSum sum)
    : point_sources_(point_sources), exclusion_radius_(exclusion_radius), sum_(sum) {
    if (point_sources_ == PointSourceRule::exclude_radius) {
        if (!(std::isfinite(exclusion_radius_) && exclusion_radius_ > 0.0))
            throw std::invalid_argument(
                "ErrorEstimator: point_source_exclusion_radius must be finite and positive for exclude_radius");
        if (!std::isfinite(exclusion_radius_ * exclusion_radius_))
            throw std::invalid_argument(
                "ErrorEstimator: point_source_exclusion_radius is too large (square overflows)");
    } else if (!(std::isfinite(exclusion_radius_) && exclusion_radius_ >= 0.0)) {
        throw std::invalid_argument("ErrorEstimator: point_source_exclusion_radius must be finite and non-negative");
    }
}

ErrorEstimate ErrorEstimator::estimate(const MeshProperties& mesh, const RefinementInput& input) const {
    const auto start = std::chrono::steady_clock::now();
    const std::string who = name() + " estimator";

    if (input.u.size() != mesh.nodes.size())
        throw std::invalid_argument(who + ": u has " + std::to_string(input.u.size()) + " values for " +
                                    std::to_string(mesh.nodes.size()) + " nodes (one per node is needed)");
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        const auto& p = mesh.nodes[i];
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
            throw std::invalid_argument(who + ": node " + std::to_string(i) + " has non-finite coordinates");
        if (!std::isfinite(input.u[i]))
            throw std::invalid_argument(who + ": u at node " + std::to_string(i) + " is not finite");
    }
    for (std::size_t m = 0; m < mesh.materials.size(); ++m) {
        const auto& mat = mesh.materials[m];
        if (!mat.is_metal && !(std::isfinite(mat.eps_r) && mat.eps_r > 0))
            throw std::invalid_argument(who + ": material " + std::to_string(m) + " has eps_r " +
                                        std::to_string(mat.eps_r) + "; a finite positive value is needed");
        if (!(std::isfinite(mat.screening_length) && mat.screening_length >= 0))
            throw std::invalid_argument(who + ": material " + std::to_string(m) + " has invalid screening_length");
    }
    for (std::size_t k = 0; k < mesh.elements.size(); ++k) {
        for (uint32_t node : mesh.elements[k].node)
            if (node >= mesh.nodes.size())
                throw std::invalid_argument(who + ": tet " + std::to_string(k) + " has node " + std::to_string(node) +
                                            ", but the mesh has " + std::to_string(mesh.nodes.size()) + " nodes");
        if (mesh.elements[k].material >= mesh.materials.size())
            throw std::invalid_argument(who + ": tet " + std::to_string(k) + " has material " +
                                        std::to_string(mesh.elements[k].material) + ", but the mesh has " +
                                        std::to_string(mesh.materials.size()) + " materials");
    }
    for (std::size_t b = 0; b < mesh.boundary_elements.size(); ++b) {
        for (uint32_t node : mesh.boundary_elements[b].node)
            if (node >= mesh.nodes.size())
                throw std::invalid_argument(who + ": boundary element " + std::to_string(b) + " has node " +
                                            std::to_string(node) + ", but the mesh has " +
                                            std::to_string(mesh.nodes.size()) + " nodes");
        if (mesh.boundary_elements[b].boundary_condition >= mesh.boundary_conditions.size())
            throw std::invalid_argument(who + ": boundary element " + std::to_string(b) + " has boundary condition " +
                                        std::to_string(mesh.boundary_elements[b].boundary_condition) +
                                        ", but the mesh has " + std::to_string(mesh.boundary_conditions.size()));
    }
    for (std::size_t c = 0; c < mesh.boundary_conditions.size(); ++c) {
        const auto& bc = mesh.boundary_conditions[c];
        if (!std::isfinite(bc.value) || (bc.alpha && !std::isfinite(*bc.alpha)))
            throw std::invalid_argument(who + ": boundary condition " + std::to_string(c) + " is not finite");
        if (bc.type == MeshProperties::BcType::Robin && bc.alpha && *bc.alpha < 0)
            throw std::invalid_argument(who + ": boundary condition " + std::to_string(c) +
                                        " has a negative Robin alpha");
    }
    for (std::size_t s = 0; s < input.point_sources.size(); ++s) {
        const auto& p = input.point_sources[s];
        if (!std::isfinite(p.value) || !std::isfinite(p.position[0]) || !std::isfinite(p.position[1]) ||
            !std::isfinite(p.position[2]))
            throw std::invalid_argument(who + ": point source " + std::to_string(s) + " is not finite");
        if (p.element && *p.element >= mesh.elements.size())
            throw std::invalid_argument(who + ": point source " + std::to_string(s) + " names tet " +
                                        std::to_string(*p.element) + ", but the mesh has " +
                                        std::to_string(mesh.elements.size()));
    }

    ErrorEstimate result;
    result.info.estimator = name();

    // The point source rule: build the exclusion mask before computing indicators so every
    // estimator can keep its diagnostic breakdown consistent with the masked total.
    std::vector<uint8_t> excluded_mask(mesh.elements.size(), 0);
    uint32_t excluded = 0;
    if (point_sources_ == PointSourceRule::exclude_patch && !input.point_sources.empty()) {
        std::optional<MeshLocator> locator;
        std::vector<bool> patch_node(mesh.nodes.size(), false);
        uint32_t outside = 0;
        for (const auto& p : input.point_sources) {
            if (p.value == 0.0) continue;
            std::optional<uint32_t> element = p.element;
            if (!element) {
                if (!locator) locator.emplace(mesh);
                if (const auto found = locator->locate(p.position)) element = found->element;
            }
            if (!element) {
                ++outside;
                continue;
            }
            for (uint32_t node : mesh.elements[*element].node) patch_node[node] = true;
        }
        for (uint32_t k = 0; k < mesh.elements.size(); ++k)
            for (uint32_t node : mesh.elements[k].node)
                if (patch_node[node]) {
                    excluded_mask[k] = 1;
                    ++excluded;
                    break;
                }
        if (outside > 0)
            result.info.warnings.push_back(std::to_string(outside) + " point sources lie outside the mesh; they were skipped");
    } else if (point_sources_ == PointSourceRule::exclude_radius && !input.point_sources.empty()) {
        std::optional<MeshLocator> locator;
        std::vector<uint32_t> hits;
        uint32_t outside = 0;
        for (const auto& p : input.point_sources) {
            if (p.value == 0.0) continue;
            if (!locator) locator.emplace(mesh);
            locator->query_sphere(p.position, exclusion_radius_, hits);
            if (hits.empty()) {
                ++outside;
                continue;
            }
            for (uint32_t k : hits) {
                if (!excluded_mask[k]) {
                    excluded_mask[k] = 1;
                    ++excluded;
                }
            }
        }
        if (outside > 0)
            result.info.warnings.push_back(std::to_string(outside) +
                                           " point source exclusion spheres intersect no tets; they were skipped");
    }

    result.eta = indicators(mesh, input, excluded_mask, result.info);
    if (result.eta.size() != mesh.elements.size())
        throw std::logic_error(who + ": returned " + std::to_string(result.eta.size()) + " indicators for " +
                               std::to_string(mesh.elements.size()) + " tets");
    uint32_t non_metal = 0;
    uint32_t excluded_non_metal = 0;
    for (std::size_t k = 0; k < result.eta.size(); ++k) {
        if (!mesh.materials[mesh.elements[k].material].is_metal) {
            ++non_metal;
            if (excluded_mask[k]) ++excluded_non_metal;
        }
        if (excluded_mask[k]) result.eta[k] = 0;
    }
    result.all_excluded = non_metal > 0 && excluded_non_metal == non_metal;
    if (result.all_excluded) {
        result.info.warnings.push_back(
            "all non-metal elements were excluded by the point-source rule; total error could not be estimated");
    }

    // Scaled sums so large finite indicators (e.g. 1e200) do not overflow when squared.
    double largest = 0;
    for (std::size_t k = 0; k < result.eta.size(); ++k) {
        if (!(std::isfinite(result.eta[k]) && result.eta[k] >= 0))
            throw std::logic_error(who + ": the indicator of tet " + std::to_string(k) + " is " +
                                   std::to_string(result.eta[k]));
        largest = std::max(largest, result.eta[k]);
    }
    double scaled_sum = 0;
    if (largest > 0) {
        for (double v : result.eta) {
            const double r = v / largest;
            scaled_sum += sum_ == IndicatorSum::squares ? r * r : r;
        }
    }
    result.sum = sum_;
    result.total = largest * (sum_ == IndicatorSum::squares ? std::sqrt(scaled_sum) : scaled_sum);

    result.info.elements = static_cast<uint32_t>(mesh.elements.size());
    result.info.max_indicator = largest;
    result.info.point_sources = static_cast<uint32_t>(input.point_sources.size());
    result.info.excluded_elements = excluded;
    result.info.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
