#include "amr/error_based_adapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace {

// The stage reports keep only what the adapter's own Info does not already say: the input
// and output element counts are AdaptationResult::Info::elements_before/after, and the
// number of marked tets is the mark stage's "marked".

StageInfo estimate_stage(const ErrorEstimate& estimate) {
    const ErrorEstimate::Info& info = estimate.info;
    StageInfo stage;
    stage.stage = "estimate";
    stage.method = info.estimator;
    stage.time = info.time;
    stage.metrics = info.details;
    stage.metrics["total"] = estimate.total;
    stage.metrics["max_indicator"] = info.max_indicator;
    stage.metrics["point_sources"] = static_cast<double>(info.point_sources);
    stage.metrics["excluded_elements"] = static_cast<double>(info.excluded_elements);
    stage.warnings = info.warnings;
    return stage;
}

StageInfo refine_stage(const RefinementResult::Info& info) {
    StageInfo stage;
    stage.stage = "refine";
    stage.method = info.refiner;
    stage.time = info.time;
    stage.metrics = info.details;
    stage.metrics["refined_elements"] = static_cast<double>(info.refined_elements);
    stage.metrics["nodes_before"] = static_cast<double>(info.nodes_before);
    stage.metrics["nodes_after"] = static_cast<double>(info.nodes_after);
    stage.warnings = info.warnings;
    return stage;
}

} // namespace

ErrorBasedAdapter::ErrorBasedAdapter(std::shared_ptr<const ErrorEstimator> estimator,
                                     std::shared_ptr<const MarkingStrategy> marking,
                                     std::shared_ptr<const Refiner> refiner)
    : ErrorBasedAdapter(std::move(estimator), std::move(marking), std::move(refiner), Settings{}) {}

ErrorBasedAdapter::ErrorBasedAdapter(std::shared_ptr<const ErrorEstimator> estimator,
                                     std::shared_ptr<const MarkingStrategy> marking,
                                     std::shared_ptr<const Refiner> refiner, Settings settings)
    : estimator_(std::move(estimator)), marking_(std::move(marking)), refiner_(std::move(refiner)),
      settings_(settings) {
    if (!estimator_) throw std::invalid_argument("ErrorBasedAdapter: estimator must not be null");
    if (!marking_) throw std::invalid_argument("ErrorBasedAdapter: marking strategy must not be null");
    if (!refiner_) throw std::invalid_argument("ErrorBasedAdapter: refiner must not be null");
    if (settings_.absolute_tolerance && !(std::isfinite(*settings_.absolute_tolerance) && *settings_.absolute_tolerance > 0))
        throw std::invalid_argument("ErrorBasedAdapter::Settings::absolute_tolerance must be finite and positive, got " +
                                    std::to_string(*settings_.absolute_tolerance));
    if (settings_.relative_tolerance &&
        !(std::isfinite(*settings_.relative_tolerance) && *settings_.relative_tolerance > 0 && *settings_.relative_tolerance < 1))
        throw std::invalid_argument("ErrorBasedAdapter::Settings::relative_tolerance must be in (0, 1), got " +
                                    std::to_string(*settings_.relative_tolerance));
    if (settings_.max_elements && *settings_.max_elements == 0)
        throw std::invalid_argument("ErrorBasedAdapter::Settings::max_elements must be positive");
}

std::unique_ptr<MeshAdapter::Session> ErrorBasedAdapter::start() const { return start(State{}); }

std::unique_ptr<ErrorBasedAdapter::Session> ErrorBasedAdapter::start(State initial) const {
    if (initial.initial_error && !(std::isfinite(*initial.initial_error) && *initial.initial_error >= 0))
        throw std::invalid_argument("ErrorBasedAdapter::State::initial_error must be finite and >= 0, got " +
                                    std::to_string(*initial.initial_error));
    // Not make_unique: the Session constructor is private to this class.
    return std::unique_ptr<Session>(new Session(*this, std::move(initial)));
}

ErrorBasedAdapter::Session::Session(const ErrorBasedAdapter& adapter, State initial)
    : MeshAdapter::Session(adapter.name()), estimator_(adapter.estimator_), marking_(adapter.marking_),
      refiner_(adapter.refiner_), settings_(adapter.settings_), state_(std::move(initial)) {}

AdaptationResult ErrorBasedAdapter::Session::adapt_step(const MeshProperties& mesh, const RefinementInput& input) {
    AdaptationResult result;
    const bool over_limit = settings_.max_elements && mesh.elements.size() > *settings_.max_elements;
    if (over_limit && !settings_.estimate_at_limit) {
        result.status = AdaptationStatus::limit_reached;
        return result;
    }

    // 1. Estimate
    ErrorEstimate est = estimator_->estimate(mesh, input);
    if (!std::isfinite(est.total))
        throw std::runtime_error("error_based adapter: estimated total error is not finite (" +
                                 std::to_string(est.total) + ")");
    result.info.stages.push_back(estimate_stage(est));

    // Keep the baseline local until this step succeeds so a failing refinement does not
    // poison relative_tolerance on a subsequent retry. The estimate is returned on every
    // outcome after this point, including the ones that stop early.
    const double baseline = state_.initial_error.value_or(est.total);
    std::optional<double> bound = settings_.absolute_tolerance;
    if (settings_.relative_tolerance) {
        const double rel = *settings_.relative_tolerance * baseline;
        bound = bound ? std::min(*bound, rel) : rel;
    }
    if (bound && est.total <= *bound) {
        state_.initial_error = baseline;
        result.status = AdaptationStatus::converged;
        result.extras = ErrorBasedExtras{std::move(est.eta)};
        return result;
    }
    if (over_limit) {
        state_.initial_error = baseline;
        result.status = AdaptationStatus::limit_reached;
        result.extras = ErrorBasedExtras{std::move(est.eta)};
        return result;
    }

    // 2. Mark
    const auto mark_start = std::chrono::steady_clock::now();
    const std::vector<uint32_t> marked = marking_->mark(mesh, est.eta, est.sum);
    const double mark_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - mark_start).count();
    result.info.stages.push_back(StageInfo{
        .stage = "mark",
        .method = marking_->name(),
        .time = mark_time,
        .metrics = {{"marked", static_cast<double>(marked.size())}},
        .warnings = {},
    });
    if (marked.empty()) {
        state_.initial_error = baseline;
        result.status = AdaptationStatus::unchanged;
        result.extras = ErrorBasedExtras{std::move(est.eta)};
        return result;
    }

    // 3. Refine
    RefinementResult refined = refiner_->refine(mesh, marked, state_.history);
    result.info.stages.push_back(refine_stage(refined.info));
    state_.initial_error = baseline;
    state_.history = std::move(refined.history);
    result.status = AdaptationStatus::adapted;
    result.mesh = std::move(refined.mesh);
    result.parents = std::move(refined.parents);
    result.extras = ErrorBasedExtras{std::move(est.eta)};
    return result;
}
