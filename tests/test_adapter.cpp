// Tests for MeshAdapter, ErrorBasedAdapter, and the PoissonDriver placeholder header.

#include "amr/error_based_adapter.h"
#include "amr/kelly_estimator.h"
#include "amr/longest_edge_bisection.h"
#include "amr/red_green_refinement.h"
#include "amr/residual_estimator.h"
#include "poisson_driver.h"
#include "test_support.h"

#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>

constexpr double pi = std::numbers::pi;

std::vector<double> smooth_u(const MeshProperties& mesh) {
    std::vector<double> u;
    u.reserve(mesh.nodes.size());
    for (const auto& p : mesh.nodes)
        u.push_back(std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]));
    return u;
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
            CHECK(last.info.error_estimate.has_value());
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

    // adapt_once() with explicit RefinementHistory + initial_error matches Session::adapt() identically.
    MeshProperties mesh = cube_mesh(2);
    RefinementHistory history;
    std::optional<double> baseline;
    uint32_t step_cycles = 0;
    AdaptationStatus final_status = AdaptationStatus::unchanged;
    for (; step_cycles < 10; ++step_cycles) {
        const auto u = smooth_u(mesh);
        auto step = adapter.adapt_once(mesh, smooth_input(u), std::move(history), baseline);
        if (!baseline && step.adaptation.info.error_estimate) baseline = *step.adaptation.info.error_estimate;
        history = std::move(step.history);
        final_status = step.adaptation.status;
        if (final_status != AdaptationStatus::adapted) break;
        mesh = std::move(*step.adaptation.mesh);
    }
    CHECK(final_status == AdaptationStatus::converged);
    CHECK(mesh.elements.size() == n1);
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
    CHECK_NEAR(*r.info.error_estimate, initial, 1e-12);

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

    // Default estimate_at_limit = true: runs estimate, reports error_estimate, then stops.
    const ErrorBasedAdapter with_estimate(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                          std::make_shared<LongestEdgeBisection>(),
                                          ErrorBasedAdapter::Settings{.max_elements = 40});
    const AdaptationResult r1 = with_estimate.start()->adapt(mesh, input);
    CHECK(r1.status == AdaptationStatus::limit_reached);
    CHECK(!r1.mesh.has_value());
    CHECK(r1.info.error_estimate.has_value() && *r1.info.error_estimate > 0);
    CHECK(r1.info.stages.size() == 1 && r1.info.stages[0].stage == "estimate");

    // estimate_at_limit = false: stops immediately before running the estimator.
    const ErrorBasedAdapter skip_estimate(std::make_shared<KellyEstimator>(), std::make_shared<DorflerMarking>(),
                                          std::make_shared<LongestEdgeBisection>(),
                                          ErrorBasedAdapter::Settings{.max_elements = 40, .estimate_at_limit = false});
    const AdaptationResult r2 = skip_estimate.start()->adapt(mesh, input);
    CHECK(r2.status == AdaptationStatus::limit_reached);
    CHECK(!r2.mesh.has_value());
    CHECK(!r2.info.error_estimate.has_value());
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
    CHECK_NEAR(*r.info.error_estimate, 0.0, 1e-12);
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
    bool needs_source() const override { return false; }
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
    CHECK(r.info.error_estimate.has_value() && std::isfinite(*r.info.error_estimate));
    CHECK_NEAR(*r.info.error_estimate / (std::sqrt(6.0) * 1e200), 1.0, 1e-12);

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

int main() {
    test_multi_cycle_and_session_history();
    test_absolute_and_combined_tolerance();
    test_max_elements_limit();
    test_unchanged_and_driver_placeholder();
    test_overflow_and_failed_first_call();
    test_bad_inputs();
    return finish("adapter");
}
