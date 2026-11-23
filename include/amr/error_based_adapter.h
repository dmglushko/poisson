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
// element limit). Each Session started from this adapter carries a State (the red-green
// RefinementHistory and the first cycle's error for relative_tolerance) from one cycle to
// the next by itself, so an in-process loop never touches it. When cycles run in separate
// processes (one poisson_adapt call each), read it with Session::state() after a cycle and
// pass it to start(State) for the next one: the result is the same as one long session.
class ErrorBasedAdapter final : public MeshAdapter {
public:
    // Everything a session carries from one cycle to the next. After each adapt() call it
    // belongs to the mesh the caller continues with: the refined mesh when the call
    // returned `adapted`, otherwise the unchanged input mesh.
    struct State {
        // Red-green green families of that mesh; empty for other refiners and for a mesh no
        // refiner made (imported).
        RefinementHistory     history;
        // Total error of the first estimated cycle: the base relative_tolerance multiplies.
        // Empty until a cycle has estimated. Must be finite and >= 0.
        std::optional<double> initial_error;
    };

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
        //   true  (default): run the estimator first so the estimate stage is reported in info.stages
        //                    (and check convergence), then stop without marking or refining.
        //   false:           stop immediately before estimation (useful for expensive
        //                    estimators such as goal-oriented / adjoint DWR).
        bool                    estimate_at_limit = true;
    };

    class Session final : public MeshAdapter::Session {
    public:
        // The state after the latest successful adapt() call (or the initial state).
        const State& state() const { return state_; }

    private:
        friend class ErrorBasedAdapter;
        Session(const ErrorBasedAdapter& adapter, State initial);
        AdaptationResult adapt_step(const MeshProperties& mesh, const RefinementInput& input) override;

        std::shared_ptr<const ErrorEstimator>  estimator_;
        std::shared_ptr<const MarkingStrategy> marking_;
        std::shared_ptr<const Refiner>         refiner_;
        ErrorBasedAdapter::Settings            settings_;
        State                                  state_;
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

    std::string name() const override { return "error_based"; }
    // A fresh session: same as start(State{}).
    std::unique_ptr<MeshAdapter::Session> start() const override;
    // A session resuming from a saved state. Throws std::invalid_argument for a non-finite
    // or negative initial_error; a history that does not fit the mesh is rejected by the
    // refiner on the first refinement.
    std::unique_ptr<Session> start(State initial) const;

private:
    std::shared_ptr<const ErrorEstimator>  estimator_;
    std::shared_ptr<const MarkingStrategy> marking_;
    std::shared_ptr<const Refiner>         refiner_;
    Settings                               settings_;
};
