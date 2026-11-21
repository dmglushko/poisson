#include "amr/error_based_adapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace {

StageInfo estimate_stage(const ErrorEstimate::Info& info) {
    StageInfo stage;
    stage.stage = "estimate";
    stage.method = info.estimator;
    stage.time = info.time;
    stage.metrics = info.details;
    stage.metrics["elements"] = static_cast<double>(info.elements);
    stage.metrics["total"] = info.total;
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
    stage.metrics["marked"] = static_cast<double>(info.marked);
    stage.metrics["refined_elements"] = static_cast<double>(info.refined_elements);
    stage.metrics["elements_before"] = static_cast<double>(info.elements_before);
    stage.metrics["elements_after"] = static_cast<double>(info.elements_after);
    stage.metrics["nodes_before"] = static_cast<double>(info.nodes_before);
    stage.metrics["nodes_after"] = static_cast<double>(info.nodes_after);
    stage.warnings = info.warnings;
    return stage;
}

class ErrorBasedSession final : public MeshAdapter::Session {
public:
    ErrorBasedSession(std::string adapter_name, std::shared_ptr<const ErrorEstimator> estimator,
                      std::shared_ptr<const MarkingStrategy> marking, std::shared_ptr<const Refiner> refiner,
                      ErrorBasedAdapter::Settings settings)
        : Session(std::move(adapter_name)), estimator_(std::move(estimator)), marking_(std::move(marking)),
          refiner_(std::move(refiner)), settings_(settings) {}

private:
    AdaptationResult adapt_step(const MeshProperties& mesh, const RefinementInput& input) override {
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
        result.info.error_estimate = est.total;
        result.info.stages.push_back(estimate_stage(est.info));

        // Keep the baseline local until this step succeeds so a failing refinement does not
        // poison relative_tolerance on a subsequent retry.
        const double baseline = initial_error_.value_or(est.total);
        std::optional<double> bound = settings_.absolute_tolerance;
        if (settings_.relative_tolerance) {
            const double rel = *settings_.relative_tolerance * baseline;
            bound = bound ? std::min(*bound, rel) : rel;
        }
        if (bound && est.total <= *bound) {
            initial_error_ = baseline;
            result.status = AdaptationStatus::converged;
            return result;
        }
        if (over_limit) {
            initial_error_ = baseline;
            result.status = AdaptationStatus::limit_reached;
            return result;
        }

        // 2. Mark
        const auto mark_start = std::chrono::steady_clock::now();
        const std::vector<uint32_t> marked = marking_->mark(mesh, est.eta);
        const double mark_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - mark_start).count();
        result.info.stages.push_back(StageInfo{
            .stage = "mark",
            .method = marking_->name(),
            .time = mark_time,
            .metrics = {{"elements", static_cast<double>(mesh.elements.size())},
                        {"marked", static_cast<double>(marked.size())}},
            .warnings = {},
        });
        if (marked.empty()) {
            initial_error_ = baseline;
            result.status = AdaptationStatus::unchanged;
            return result;
        }

        // 3. Refine
        RefinementResult refined = refiner_->refine(mesh, marked, history_);
        result.info.stages.push_back(refine_stage(refined.info));
        initial_error_ = baseline;
        history_ = std::move(refined.history);
        result.status = AdaptationStatus::adapted;
        result.mesh = std::move(refined.mesh);
        result.parents = std::move(refined.parents);
        return result;
    }

    std::shared_ptr<const ErrorEstimator>  estimator_;
    std::shared_ptr<const MarkingStrategy> marking_;
    std::shared_ptr<const Refiner>         refiner_;
    ErrorBasedAdapter::Settings            settings_;
    RefinementHistory                      history_;
    std::optional<double>                  initial_error_;
};

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

std::unique_ptr<MeshAdapter::Session> ErrorBasedAdapter::start() const {
    return std::make_unique<ErrorBasedSession>(name(), estimator_, marking_, refiner_, settings_);
}
