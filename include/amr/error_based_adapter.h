#pragma once

#include "amr/error_estimator.h"
#include "amr/marking.h"
#include "amr/mesh_adapter.h"
#include "amr/refiner.h"

#include <cstdint>
#include <memory>
#include <optional>

// Composite mesh adapter: estimate -> mark -> refine.
//
// Owns its three interchangeable components and its stopping rules (error tolerances and
// element limit). Each Session started from this adapter keeps its own RefinementHistory
// (for red-green refinement) and first-cycle error (for relative_tolerance) privately, so
// callers never manage refinement state manually.
class ErrorBasedAdapter final : public MeshAdapter {
public:
    struct Settings {
        // Stop with AdaptationStatus::converged when ErrorEstimate::total <= tolerance.
        // relative_tolerance is multiplied by the first estimated cycle's total error.
        // When both are set, the smaller (stricter) bound applies.
        std::optional<double>   absolute_tolerance = std::nullopt;   // finite and > 0
        std::optional<double>   relative_tolerance = std::nullopt;   // in (0, 1)

        // Input-mesh stopping threshold: stop with AdaptationStatus::limit_reached when
        // mesh.elements.size() > *max_elements. Equality still permits one refinement step,
        // and the returned mesh can exceed *max_elements by that step's refinement and
        // conformity-closure cost.
        std::optional<uint64_t> max_elements = std::nullopt;         // > 0

        // Controls what happens when mesh.elements.size() > *max_elements:
        //   true  (default): run the estimator first so info.error_estimate is populated
        //                    (and check convergence), then stop without marking or refining.
        //   false:           stop immediately before estimation (useful for expensive
        //                    estimators such as goal-oriented / adjoint DWR).
        bool                    estimate_at_limit = true;
    };

    struct StepResult {
        AdaptationResult  adaptation;
        RefinementHistory history;
    };

    // Throws std::invalid_argument for any null component or invalid setting.
    ErrorBasedAdapter(std::shared_ptr<const ErrorEstimator> estimator,
                      std::shared_ptr<const MarkingStrategy> marking,
                      std::shared_ptr<const Refiner> refiner);
    ErrorBasedAdapter(std::shared_ptr<const ErrorEstimator> estimator,
                      std::shared_ptr<const MarkingStrategy> marking,
                      std::shared_ptr<const Refiner> refiner,
                      Settings settings);

    const Settings&        settings() const { return settings_; }
    const ErrorEstimator&  estimator() const { return *estimator_; }
    const MarkingStrategy& marking() const { return *marking_; }
    const Refiner&         refiner() const { return *refiner_; }

    std::string              name() const override { return "error_based"; }
    std::unique_ptr<Session> start() const override;

    // Runs one adaptation step with explicit RefinementHistory and optional first-cycle
    // baseline error (for relative_tolerance across stateless invocations such as the CLI).
    // Uses the exact same session logic and checks as Session::adapt().
    StepResult adapt_once(const MeshProperties& mesh, const RefinementInput& input,
                          RefinementHistory history = {},
                          std::optional<double> initial_error = std::nullopt) const;

private:
    std::shared_ptr<const ErrorEstimator>  estimator_;
    std::shared_ptr<const MarkingStrategy> marking_;
    std::shared_ptr<const Refiner>         refiner_;
    Settings                               settings_;
};
