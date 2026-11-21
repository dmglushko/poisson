// The marking strategies: the shared checks in MarkingStrategy::mark and the four
// strategies' select(). The rules are those of pde2d (adapt/marking.cpp).

#include "amr/marking.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

// theta and fraction: a number in [0, 1]. Written so that NaN fails too.
void check_unit_interval(double value, const std::string& what) {
    if (!(value >= 0.0 && value <= 1.0))
        throw std::invalid_argument(what + " must be in [0, 1], got " + std::to_string(value));
}

// Tet indices by decreasing indicator, equal indicators by increasing index (a stable sort).
std::vector<uint32_t> largest_first(std::span<const double> eta) {
    std::vector<uint32_t> order(eta.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return eta[a] > eta[b]; });
    return order;
}

} // namespace

std::vector<uint32_t> MarkingStrategy::mark(const MeshProperties& mesh, std::span<const double> eta) const {
    if (eta.size() != mesh.elements.size())
        throw std::invalid_argument(name() + " marking: " + std::to_string(eta.size()) + " indicators for " +
                                    std::to_string(mesh.elements.size()) + " tets (one per tet is needed)");
    bool any_positive = false;
    for (std::size_t k = 0; k < eta.size(); ++k) {
        if (!std::isfinite(eta[k]) || eta[k] < 0.0)
            throw std::invalid_argument(name() + " marking: the indicator of tet " + std::to_string(k) +
                                        " must be finite and non-negative, got " + std::to_string(eta[k]));
        any_positive = any_positive || eta[k] > 0.0;
    }
    if (!any_positive) return {};
    std::vector<uint32_t> picked = select(eta);
    // The shared rules: a zero indicator is never marked; sorted, without repetitions.
    std::erase_if(picked, [&](uint32_t k) { return !(eta[k] > 0.0); });
    std::sort(picked.begin(), picked.end());
    picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
    return picked;
}

// --- Dörfler ---------------------------------------------------------------------------

DorflerMarking::DorflerMarking() : DorflerMarking(Settings{}) {}

DorflerMarking::DorflerMarking(Settings settings) : settings_(settings) {
    check_unit_interval(settings_.theta, "DorflerMarking::Settings::theta");
}

std::vector<uint32_t> DorflerMarking::select(std::span<const double> eta) const {
    const std::vector<uint32_t> order = largest_first(eta);
    std::vector<uint32_t> picked;
    // theta = 1 asks for every positive indicator. The sums below cannot promise that: a
    // tiny square (1e-13 next to 1e-4) vanishes when added to the total.
    if (settings_.theta == 1.0) {
        for (uint32_t k : order)
            if (eta[k] > 0.0) picked.push_back(k);
        return picked;
    }
    // Squares of eta / max eta, so that large indicators cannot overflow when squared; the
    // total is added in the same order as the partial sums, so the two agree bit for bit.
    const double largest = eta[order.front()];
    const auto square = [&](uint32_t k) {
        const double r = eta[k] / largest;
        return r * r;
    };
    double total = 0.0;
    for (uint32_t k : order) total += square(k);
    const double bound = settings_.theta * total;
    double sum = 0.0;
    for (uint32_t k : order) {
        if (sum >= bound) break;
        picked.push_back(k);
        sum += square(k);
    }
    return picked;
}

// --- maximum ---------------------------------------------------------------------------

MaximumMarking::MaximumMarking() : MaximumMarking(Settings{}) {}

MaximumMarking::MaximumMarking(Settings settings) : settings_(settings) {
    check_unit_interval(settings_.theta, "MaximumMarking::Settings::theta");
}

std::vector<uint32_t> MaximumMarking::select(std::span<const double> eta) const {
    const double bound = settings_.theta * *std::max_element(eta.begin(), eta.end());
    std::vector<uint32_t> picked;
    for (uint32_t k = 0; k < eta.size(); ++k)
        if (eta[k] >= bound) picked.push_back(k);
    return picked;
}

// --- fixed fraction --------------------------------------------------------------------

FixedFractionMarking::FixedFractionMarking() : FixedFractionMarking(Settings{}) {}

FixedFractionMarking::FixedFractionMarking(Settings settings) : settings_(settings) {
    check_unit_interval(settings_.fraction, "FixedFractionMarking::Settings::fraction");
}

std::vector<uint32_t> FixedFractionMarking::select(std::span<const double> eta) const {
    std::vector<uint32_t> order = largest_first(eta);
    // Rounded to the nearest: 0.07 * 100 is 7.000000000000001 in doubles.
    const auto count = static_cast<std::size_t>(std::round(settings_.fraction * static_cast<double>(eta.size())));
    order.resize(std::min(count, order.size()));
    return order;
}

// --- equidistribution ------------------------------------------------------------------

EquidistributionMarking::EquidistributionMarking(Settings settings) : settings_(settings) {
    if (!(std::isfinite(settings_.tolerance) && settings_.tolerance > 0.0))
        throw std::invalid_argument("EquidistributionMarking::Settings::tolerance must be set, finite and positive, got " +
                                    std::to_string(settings_.tolerance));
}

std::vector<uint32_t> EquidistributionMarking::select(std::span<const double> eta) const {
    const double bound = settings_.tolerance / std::sqrt(static_cast<double>(eta.size()));
    std::vector<uint32_t> picked;
    for (uint32_t k = 0; k < eta.size(); ++k)
        if (eta[k] >= bound) picked.push_back(k);
    return picked;
}
