// Tests for MeshAdapter, ErrorBasedAdapter, and the PoissonDriver placeholder header.

#include "amr/error_based_adapter.h"
#include "amr/kelly_estimator.h"
#include "amr/longest_edge_bisection.h"
#include "amr/red_green_refinement.h"
#include "amr/residual_estimator.h"
#include "amr/trajectory_estimator.h"
#include "poisson_driver.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>

constexpr double pi = std::numbers::pi;

std::vector<double> smooth_u(const MeshProperties& mesh) {
    std::vector<double> u;
    u.reserve(mesh.nodes.size());
    for (const auto& p : mesh.nodes)
        u.push_back(std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]));
    return u;
}

// The total error the adapter estimated this cycle: the "total" metric of its estimate stage,
// or empty when it did not estimate.
std::optional<double> estimated_total(const AdaptationResult& r) {
    for (const StageInfo& stage : r.info.stages)
        if (stage.stage == "estimate") return stage.metrics.at("total");
    return std::nullopt;
}

RefinementInput smooth_input(const std::vector<double>& u) {
    return {u, [](const std::array<double, 3>& x, uint32_t) {
                return 3 * pi * pi * std::sin(pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[2]);
            },
            {}};
}

// Multi-cycle adaptation with RedGreenRefinement: the Session keeps RefinementHistory
// privately so green families are reopened on later cycles without the caller touching it.
void test_multi_cycle_and_session_history() {
    const ErrorBasedAdapter adapter(std::make_shared<ResidualEstimator>(),
                                    std::make_shared<DorflerMarking>(DorflerMarking::Settings{.theta = 0.5}),
                                    std::make_shared<RedGreenRefinement>(),
                                    ErrorBasedAdapter::Settings{.relative_tolerance = 0.6});
    CHECK(adapter.name() == "error_based");

    // Two independent sessions from the same const adapter must give identical results.
    const auto run_once = [&]() {
        MeshProperties mesh = cube_mesh(2);
        auto session = adapter.start();
        uint32_t cycles = 0;
        double reopened_total = 0;
        AdaptationResult last;
        for (; cycles < 10; ++cycles) {
            const auto u = smooth_u(mesh);
            last = session->adapt(mesh, smooth_input(u));
            CHECK(last.info.adapter == "error_based");
            CHECK(last.info.elements_before == mesh.elements.size());
            CHECK(estimated_total(last).has_value());
            if (last.status != AdaptationStatus::adapted) break;
            CHECK(last.mesh.has_value() && last.parents.has_value());
            CHECK(last.info.elements_after == last.mesh->elements.size());
            CHECK(last.info.stages.size() == 3);
            CHECK(last.info.stages[0].stage == "estimate" && last.info.stages[0].method == "residual");
            CHECK(last.info.stages[1].stage == "mark" && last.info.stages[1].method == "dorfler");
            CHECK(last.info.stages[2].stage == "refine" && last.info.stages[2].method == "red_green");
            reopened_total += last.info.stages[2].metrics.at("reopened_green_families");
            mesh = std::move(*last.mesh);
        }
        CHECK(last.status == AdaptationStatus::converged);
        CHECK(!last.mesh.has_value());
        CHECK(last.info.stages.size() == 1);   // stopped after estimate
        CHECK(last.info.elements_after == last.info.elements_before);
        CHECK(cycles >= 2);
        CHECK(reopened_total >= 1);            // proves session preserved RedGreen history
        return mesh.elements.size();
    };
    const std::size_t n1 = run_once();
    const std::size_t n2 = run_once();
    CHECK(n1 == n2);
}

void test_absolute_and_combined_tolerance() {
    const MeshProperties mesh = cube_mesh(2);
    const auto u = smooth_u(mesh);
    const RefinementInput input = smooth_input(u);
    const double initial = KellyEstimator().estimate(mesh, input).total;
    CHECK(initial > 0);

    // Absolute tolerance above initial error converges on cycle 0 without refining.
    const ErrorBasedAdapter generous(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                     std::make_shared<LongestEdgeBisection>(),
                                     ErrorBasedAdapter::Settings{.absolute_tolerance = initial * 2});
    const AdaptationResult r = generous.start()->adapt(mesh, input);
    CHECK(r.status == AdaptationStatus::converged);
    CHECK(!r.mesh.has_value());
    CHECK(r.info.stages.size() == 1);
    CHECK_NEAR(estimated_total(r).value(), initial, 1e-12);

    // Both absolute and relative set: the smaller (stricter) bound wins, so a generous
    // absolute tolerance does not stop early when relative_tolerance = 0.5 demands halving.
    const ErrorBasedAdapter strict(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                   std::make_shared<LongestEdgeBisection>(),
                                   ErrorBasedAdapter::Settings{.absolute_tolerance = initial * 2, .relative_tolerance = 0.5});
    const AdaptationResult r2 = strict.start()->adapt(mesh, input);
    CHECK(r2.status == AdaptationStatus::adapted);
    CHECK(r2.mesh.has_value());
}

void test_max_elements_limit() {
    const MeshProperties mesh = cube_mesh(2);   // 48 tets
    const auto u = smooth_u(mesh);
    const RefinementInput input = smooth_input(u);

    // Default estimate_at_limit = true: runs estimate, reports it in info.stages, then stops.
    const ErrorBasedAdapter with_estimate(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                          std::make_shared<LongestEdgeBisection>(),
                                          ErrorBasedAdapter::Settings{.max_elements = 40});
    const AdaptationResult r1 = with_estimate.start()->adapt(mesh, input);
    CHECK(r1.status == AdaptationStatus::limit_reached);
    CHECK(!r1.mesh.has_value());
    CHECK(estimated_total(r1).has_value() && estimated_total(r1).value() > 0);
    CHECK(r1.info.stages.size() == 1 && r1.info.stages[0].stage == "estimate");

    // estimate_at_limit = false: stops immediately before running the estimator.
    const ErrorBasedAdapter skip_estimate(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                          std::make_shared<LongestEdgeBisection>(),
                                          ErrorBasedAdapter::Settings{.max_elements = 40, .estimate_at_limit = false});
    const AdaptationResult r2 = skip_estimate.start()->adapt(mesh, input);
    CHECK(r2.status == AdaptationStatus::limit_reached);
    CHECK(!r2.mesh.has_value());
    CHECK(!estimated_total(r2).has_value());
    CHECK(r2.info.stages.empty());

    // Exactly at max_elements (not strictly greater) still adapts once.
    const ErrorBasedAdapter at_limit(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                     std::make_shared<LongestEdgeBisection>(),
                                     ErrorBasedAdapter::Settings{.max_elements = mesh.elements.size()});
    const AdaptationResult r3 = at_limit.start()->adapt(mesh, input);
    CHECK(r3.status == AdaptationStatus::adapted);
    CHECK(r3.mesh.has_value() && r3.mesh->elements.size() > mesh.elements.size());
}

void test_unchanged_and_driver_placeholder() {
    const MeshProperties mesh = cube_mesh(2);
    std::vector<double> linear_u;
    for (const auto& p : mesh.nodes) linear_u.push_back(2 * p[0] - p[1] + 3 * p[2]);
    const RefinementInput input{linear_u, {}, {}};

    const auto adapter = std::make_shared<ErrorBasedAdapter>(
        std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(), std::make_shared<LongestEdgeBisection>());
    const AdaptationResult r = adapter->start()->adapt(mesh, input);
    CHECK(r.status == AdaptationStatus::unchanged);
    CHECK(!r.mesh.has_value());
    CHECK(r.info.stages.size() == 2);   // estimate + mark; refine skipped
    CHECK_NEAR(estimated_total(r).value(), 0.0, 1e-12);
    CHECK(r.info.stages[1].metrics.at("marked") == 0);

    // Smoke-check placeholder PoissonDriver header compiles and stores the optional adapter.
    const PoissonDriver single;
    CHECK(!single.adapter() && single.settings().max_cycles == 1);
    const PoissonDriver adaptive(adapter, {.max_cycles = 5, .max_nodes = 100'000});
    CHECK(adaptive.adapter() == adapter && adaptive.settings().max_cycles == 5 && *adaptive.settings().max_nodes == 100'000);
}

class ConstantEstimator final : public ErrorEstimator {
public:
    explicit ConstantEstimator(double value) : ErrorEstimator(PointSourceRule::ignore), value_(value) {}
    std::string name() const override { return "constant"; }

private:
    std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput&, std::span<const uint8_t>,
                                   ErrorEstimate::Info&) const override {
        return std::vector<double>(mesh.elements.size(), value_);
    }
    double value_;
};

void test_overflow_and_failed_first_call() {
    const MeshProperties mesh = cube_mesh(1);   // 6 tets
    const std::vector<double> u(mesh.nodes.size(), 0.0);
    const RefinementInput input{u, {}, {}};

    // Six 1e200 indicators have a representable norm sqrt(6)*1e200; must adapt, not falsely converge.
    const ErrorBasedAdapter huge(std::make_shared<ConstantEstimator>(1e200), std::make_shared<DorflerMarking>(),
                                 std::make_shared<LongestEdgeBisection>(),
                                 ErrorBasedAdapter::Settings{.relative_tolerance = 0.5});
    const AdaptationResult r = huge.start()->adapt(mesh, input);
    CHECK(r.status == AdaptationStatus::adapted);
    CHECK(estimated_total(r).has_value() && std::isfinite(estimated_total(r).value()));
    CHECK_NEAR(estimated_total(r).value() / (std::sqrt(6.0) * 1e200), 1.0, 1e-12);

    // Unrepresentable total norm (6 * (1e308)^2 overflows double) must throw rather than converge.
    const ErrorBasedAdapter unrepresentable(std::make_shared<ConstantEstimator>(1e308), std::make_shared<DorflerMarking>(),
                                            std::make_shared<LongestEdgeBisection>(),
                                            ErrorBasedAdapter::Settings{.relative_tolerance = 0.5});
    CHECK_THROWS(std::runtime_error, unrepresentable.start()->adapt(mesh, input), "not finite");

    // A failed first refinement (e.g. max_new_nodes exceeded) must NOT commit initial_error_.
    const MeshProperties finer = cube_mesh(2);
    const ErrorBasedAdapter tight(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                  std::make_shared<RedGreenRefinement>(RedGreenRefinement::Settings{.max_new_nodes = 1}),
                                  ErrorBasedAdapter::Settings{.relative_tolerance = 0.5});
    auto session = tight.start();
    const auto big_u = smooth_u(finer);
    CHECK_THROWS(std::runtime_error, session->adapt(finer, smooth_input(big_u)), "new nodes");
    // Retry on the same session with a 10x smaller solution: if the failed call had poisoned
    // initial_error_, this would falsely report converged instead of attempting refinement.
    std::vector<double> small_u = big_u;
    for (double& v : small_u) v *= 0.1;
    CHECK_THROWS(std::runtime_error, session->adapt(finer, smooth_input(small_u)), "new nodes");
}

void test_bad_inputs() {
    const auto est = std::make_shared<KellyEstimator>();
    const auto mark = std::make_shared<DorflerMarking>();
    const auto ref = std::make_shared<LongestEdgeBisection>();
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(nullptr, mark, ref), "estimator must not be null");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, nullptr, ref), "marking strategy must not be null");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, nullptr), "refiner must not be null");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, ref, {.absolute_tolerance = 0.0}), "absolute_tolerance");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, ref, {.absolute_tolerance = std::nan("")}), "absolute_tolerance");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, ref, {.relative_tolerance = 0.0}), "relative_tolerance");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, ref, {.relative_tolerance = 1.0}), "relative_tolerance");
    CHECK_THROWS(std::invalid_argument, ErrorBasedAdapter(est, mark, ref, {.max_elements = 0}), "max_elements");
}

// Resuming each cycle from the previous cycle's State (as poisson_adapt does across processes)
// must give exactly the same sequence as one long session: same meshes, statuses, estimates,
// red-green reopenings, and relative_tolerance convergence point.
void test_state_resume_matches_one_session() {
    const ErrorBasedAdapter adapter(std::make_shared<ResidualEstimator>(),
                                    std::make_shared<DorflerMarking>(DorflerMarking::Settings{.theta = 0.5}),
                                    std::make_shared<RedGreenRefinement>(),
                                    ErrorBasedAdapter::Settings{.relative_tolerance = 0.6});

    struct Cycle {
        AdaptationStatus status;
        double           error;
        std::size_t      elements_after;
        double           reopened;
    };
    const auto record = [](const AdaptationResult& r) {
        const double reopened =
            r.info.stages.size() == 3 ? r.info.stages[2].metrics.at("reopened_green_families") : 0.0;
        return Cycle{r.status, estimated_total(r).value(), r.info.elements_after, reopened};
    };

    std::vector<Cycle> continuous;
    {
        MeshProperties mesh = cube_mesh(2);
        auto session = adapter.start();
        for (int c = 0; c < 10; ++c) {
            const auto u = smooth_u(mesh);
            AdaptationResult r = session->adapt(mesh, smooth_input(u));
            continuous.push_back(record(r));
            if (r.status != AdaptationStatus::adapted) break;
            mesh = std::move(*r.mesh);
        }
    }

    std::vector<Cycle> resumed;
    double reopened_total = 0;
    {
        MeshProperties mesh = cube_mesh(2);
        ErrorBasedAdapter::State state;   // imported mesh: empty history, no baseline yet
        CHECK(!state.initial_error && state.history.family.empty());
        for (int c = 0; c < 10; ++c) {
            auto session = adapter.start(state);   // a fresh session per cycle, like a new process
            const auto u = smooth_u(mesh);
            AdaptationResult r = session->adapt(mesh, smooth_input(u));
            resumed.push_back(record(r));
            reopened_total += resumed.back().reopened;
            state = session->state();
            CHECK(state.initial_error.has_value());
            CHECK_NEAR(*state.initial_error, resumed.front().error, 0.0);   // baseline = first cycle's total
            if (r.status != AdaptationStatus::adapted) break;
            CHECK(state.history.family.size() == r.mesh->elements.size());  // history belongs to the new mesh
            mesh = std::move(*r.mesh);
        }
    }

    CHECK(continuous.size() == resumed.size());
    for (std::size_t i = 0; i < std::min(continuous.size(), resumed.size()); ++i) {
        CHECK(continuous[i].status == resumed[i].status);
        CHECK(continuous[i].elements_after == resumed[i].elements_after);
        CHECK(continuous[i].reopened == resumed[i].reopened);
        CHECK_NEAR(continuous[i].error, resumed[i].error, 0.0);
    }
    CHECK(resumed.back().status == AdaptationStatus::converged);   // relative_tolerance works across "processes"
    CHECK(reopened_total >= 1);                                      // history carried across "processes"

    // Without the state, every cycle is its own first cycle: relative_tolerance can never fire.
    {
        MeshProperties mesh = cube_mesh(2);
        for (int c = 0; c < 4; ++c) {
            const auto u = smooth_u(mesh);
            AdaptationResult r = adapter.start()->adapt(mesh, smooth_input(u));
            CHECK(r.status == AdaptationStatus::adapted);
            mesh = std::move(*r.mesh);
        }
    }

    CHECK_THROWS(std::invalid_argument, adapter.start(ErrorBasedAdapter::State{.history = {}, .initial_error = -1.0}),
                 "initial_error");
    CHECK_THROWS(std::invalid_argument,
                 adapter.start(ErrorBasedAdapter::State{.history = {}, .initial_error = std::nan("")}), "initial_error");
}

// eta in ErrorBasedExtras: one indicator per INPUT tet whenever the estimator ran; its norm is
// the estimate stage's "total" metric. When nothing was estimated, extras holds std::monostate.
void test_eta_returned() {
    const MeshProperties mesh = cube_mesh(2);   // 48 tets
    const auto u = smooth_u(mesh);
    const RefinementInput input = smooth_input(u);
    const auto norm = [](const std::vector<double>& eta) {
        double s = 0;
        for (const double e : eta) s += e * e;
        return std::sqrt(s);
    };

    const ErrorBasedAdapter plain(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                  std::make_shared<LongestEdgeBisection>());
    const AdaptationResult adapted = plain.start()->adapt(mesh, input);
    CHECK(adapted.status == AdaptationStatus::adapted);
    const auto* adapted_eb = std::get_if<ErrorBasedExtras>(&adapted.extras);
    CHECK(adapted_eb != nullptr);
    if (adapted_eb) {
        CHECK(adapted_eb->eta.size() == mesh.elements.size());   // input mesh, not the refined one
        CHECK(adapted.mesh->elements.size() != adapted_eb->eta.size());
        CHECK_NEAR(norm(adapted_eb->eta), estimated_total(adapted).value(), 1e-12 * estimated_total(adapted).value());
        CHECK(*std::max_element(adapted_eb->eta.begin(), adapted_eb->eta.end()) ==
              adapted.info.stages[0].metrics.at("max_indicator"));
        const auto expected_marked = DorflerMarking().mark(mesh, adapted_eb->eta);
        CHECK(adapted_eb->marked == expected_marked);
        CHECK(static_cast<double>(adapted_eb->marked.size()) == adapted.info.stages[1].metrics.at("marked"));
    }

    const ErrorBasedAdapter generous(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                     std::make_shared<LongestEdgeBisection>(),
                                     ErrorBasedAdapter::Settings{.absolute_tolerance = 1e9});
    const AdaptationResult converged = generous.start()->adapt(mesh, input);
    const auto* converged_eb = std::get_if<ErrorBasedExtras>(&converged.extras);
    CHECK(converged.status == AdaptationStatus::converged && converged_eb &&
          converged_eb->eta.size() == mesh.elements.size() && converged_eb->marked.empty());

    const ErrorBasedAdapter at_limit(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                     std::make_shared<LongestEdgeBisection>(),
                                     ErrorBasedAdapter::Settings{.max_elements = 40});
    const AdaptationResult limited = at_limit.start()->adapt(mesh, input);
    const auto* limited_eb = std::get_if<ErrorBasedExtras>(&limited.extras);
    CHECK(limited.status == AdaptationStatus::limit_reached && limited_eb &&
          limited_eb->eta.size() == mesh.elements.size() && limited_eb->marked.empty());

    const ErrorBasedAdapter skip(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                 std::make_shared<LongestEdgeBisection>(),
                                 ErrorBasedAdapter::Settings{.max_elements = 40, .estimate_at_limit = false});
    const AdaptationResult skipped = skip.start()->adapt(mesh, input);
    CHECK(skipped.status == AdaptationStatus::limit_reached);
    CHECK(std::holds_alternative<std::monostate>(skipped.extras));   // estimator never ran

    std::vector<double> linear_u;
    for (const auto& p : mesh.nodes) linear_u.push_back(2 * p[0] - p[1] + 3 * p[2]);
    const AdaptationResult unchanged = plain.start()->adapt(mesh, RefinementInput{linear_u, {}, {}});
    const auto* unchanged_eb = std::get_if<ErrorBasedExtras>(&unchanged.extras);
    CHECK(unchanged.status == AdaptationStatus::unchanged && unchanged_eb &&
          unchanged_eb->eta.size() == mesh.elements.size() && unchanged_eb->marked.empty());
}

// An estimator whose indicators add up linearly (IndicatorSum::values): the adapter's total
// is their sum and Dörfler marking accumulates them, not their squares.
void test_linear_indicator_sum() {
    class Ramp final : public ErrorEstimator {
    public:
        Ramp() : ErrorEstimator(PointSourceRule::ignore, IndicatorSum::values) {}
        std::string name() const override { return "ramp"; }

    private:
        std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput&, std::span<const uint8_t>,
                                       ErrorEstimate::Info&) const override {
            std::vector<double> eta(mesh.elements.size());
            for (std::size_t k = 0; k < eta.size(); ++k) eta[k] = double(k + 1);
            return eta;
        }
    };
    const MeshProperties mesh = cube_mesh(2);   // 48 tets
    const std::vector<double> u(mesh.nodes.size(), 0.0);
    const ErrorBasedAdapter adapter(std::make_shared<Ramp>(), std::make_shared<DorflerMarking>(),
                                    std::make_shared<LongestEdgeBisection>());
    const AdaptationResult r = adapter.start()->adapt(mesh, RefinementInput{u, {}, {}});
    CHECK(r.status == AdaptationStatus::adapted);
    CHECK_NEAR(estimated_total(r).value(), 48.0 * 49.0 / 2, 1e-9);
    const auto& eta = std::get<ErrorBasedExtras>(r.extras).eta;
    const double by_values = double(DorflerMarking().mark(mesh, eta, IndicatorSum::values).size());
    const double by_squares = double(DorflerMarking().mark(mesh, eta).size());
    CHECK(by_values != by_squares);
    CHECK(r.info.stages[1].stage == "mark" && r.info.stages[1].metrics.at("marked") == by_values);

    // The trajectory estimator runs in the adapter like any other.
    const auto smooth = smooth_u(mesh);
    const ErrorBasedAdapter trajectory(std::make_shared<TrajectoryEstimator>(), std::make_shared<DorflerMarking>(),
                                       std::make_shared<RedGreenRefinement>());
    const AdaptationResult t = trajectory.start()->adapt(mesh, smooth_input(smooth));
    CHECK(t.status == AdaptationStatus::adapted);
    double sum = 0;
    for (double e : std::get<ErrorBasedExtras>(t.extras).eta) sum += e;
    CHECK_NEAR(estimated_total(t).value(), sum, 1e-12 * sum);
}

int main() {
    test_multi_cycle_and_session_history();
    test_absolute_and_combined_tolerance();
    test_max_elements_limit();
    test_unchanged_and_driver_placeholder();
    test_overflow_and_failed_first_call();
    test_bad_inputs();
    test_state_resume_matches_one_session();
    test_eta_returned();
    test_linear_indicator_sum();
    return finish("adapter");
}
