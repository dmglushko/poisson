// Tests for the error estimators on structured cube meshes. With no solver yet, u_h is the
// nodal interpolant of a chosen function: exact where the function is piecewise linear
// (the estimators must give 0), and O(h) in energy otherwise.

#include "amr/kelly_estimator.h"
#include "amr/recovery_estimator.h"
#include "amr/residual_estimator.h"
#include "test_support.h"

#include <numbers>
#include <stdexcept>

using Bc = MeshProperties::BoundaryCondition;
using BcType = MeshProperties::BcType;
constexpr double pi = std::numbers::pi;

std::vector<double> interpolate(const MeshProperties& mesh, const std::function<double(const Vec3&)>& u) {
    std::vector<double> values;
    for (const auto& p : mesh.nodes) values.push_back(u(p));
    return values;
}

RefinementInput input_for(const std::vector<double>& u,
                          std::function<double(const std::array<double, 3>&, uint32_t)> f = [](const auto&, uint32_t) { return 0.0; }) {
    return {u, std::move(f), {}};
}

double total(const ErrorEstimator& estimator, const MeshProperties& mesh, const RefinementInput& input) {
    return estimator.estimate(mesh, input).total;
}

// A linear u with Dirichlet data everywhere is exact: every estimator gives 0.
void test_linear_is_exact() {
    const MeshProperties mesh = cube_mesh(3);
    const auto u = interpolate(mesh, [](const Vec3& p) { return 2 * p[0] + 3 * p[1] - p[2]; });
    const RefinementInput input = input_for(u);
    CHECK_NEAR(total(ResidualEstimator(), mesh, input), 0, 1e-12);
    CHECK_NEAR(total(KellyEstimator(), mesh, input), 0, 1e-12);
    CHECK_NEAR(total(RecoveryEstimator(), mesh, input), 0, 1e-12);
    const ErrorEstimate e = ResidualEstimator().estimate(mesh, input);
    CHECK(e.eta.size() == mesh.elements.size());
    CHECK(e.info.estimator == "residual");
    CHECK(e.info.elements == mesh.elements.size());
    CHECK(e.info.details.contains("element_part") && e.info.details.contains("jump_part"));
}

// u = x: Neumann eps du/dn = g on x = 1 must match (g = 1); a wrong g shows up in the
// boundary part. Robin eps du/dn + alpha u = g with alpha = 2, g = 3 holds as well.
void test_natural_conditions() {
    const std::vector<Bc> bcs{{BcType::Dirichlet, 0, {}}, {BcType::Neumann, 1, {}}, {BcType::Neumann, 0, {}},
                              {BcType::Robin, 3, 2.0}, {BcType::Neumann, 0.5, {}}};
    const auto u_of = [](const MeshProperties& mesh) { return interpolate(mesh, [](const Vec3& p) { return p[0]; }); };
    {
        const MeshProperties mesh = cube_mesh(2, {0, 1, 2, 2, 2, 2}, bcs);
        const auto u = u_of(mesh);
        CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u)), 0, 1e-12);
    }
    {
        const MeshProperties mesh = cube_mesh(2, {0, 3, 2, 2, 2, 2}, bcs);   // Robin on x = 1
        const auto u = u_of(mesh);
        CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u)), 0, 1e-12);
    }
    {
        // g = 0.5 instead of 1 on x = 1: R = -0.5 on a unit square, h_F = sqrt(2) / 2 per
        // face, so the boundary part is sqrt(h_F * 0.25 * 1).
        const MeshProperties mesh = cube_mesh(2, {0, 4, 2, 2, 2, 2}, bcs);
        const auto u = u_of(mesh);
        const ErrorEstimate e = ResidualEstimator().estimate(mesh, input_for(u));
        CHECK_NEAR(e.info.details.at("boundary_part"), std::sqrt(std::sqrt(2.0) / 2 * 0.25), 1e-12);
        CHECK_NEAR(e.info.details.at("jump_part"), 0, 1e-12);
        CHECK_NEAR(KellyEstimator().estimate(mesh, input_for(u)).total, e.total, 1e-12);
    }
}

// Screening: -div(grad u) + c u = f with u = x, c = 1 / 0.5^2 = 4 when mesh.screened is true (f = 4x),
// and c = 0 when mesh.screened is false even if the material carries a non-zero screening_length.
void test_screening() {
    MeshProperties mesh = cube_mesh(2, {0, 0, 0, 0, 0, 0}, {{BcType::Dirichlet, 0, {}}}, {{"screened", 1, 0.5, false, {}}});
    const auto u = interpolate(mesh, [](const Vec3& p) { return p[0]; });
    // Unscreened by default: c = 0, so f = 0 is exact and f = 4x is wrong.
    CHECK(!mesh.screened);
    CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u)), 0, 1e-12);
    CHECK(total(ResidualEstimator(), mesh, input_for(u, [](const auto& x, uint32_t) { return 4 * x[0]; })) > 0.1);
    // Screened enabled: c = 4, so f = 4x is exact and f = 0 is wrong.
    mesh.screened = true;
    CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u, [](const auto& x, uint32_t) { return 4 * x[0]; })), 0, 1e-12);
    CHECK(total(ResidualEstimator(), mesh, input_for(u)) > 0.1);
}

// Two materials, eps 1 for x < 0.5 and 4 beyond. u with a kink at the interface and a
// continuous flux eps du/dx = 1 is exact for P1; recovery per material agrees, averaging
// across the interface does not.
void test_interface() {
    const std::vector<MeshProperties::Material> materials{{"a", 1, 0, false, {}}, {"b", 4, 0, false, {}}};
    const MeshProperties mesh = cube_mesh(4, {0, 0, 1, 1, 1, 1}, {{BcType::Dirichlet, 0, {}}, {BcType::Neumann, 0, {}}}, materials, 0.5);
    const auto u = interpolate(mesh, [](const Vec3& p) { return p[0] < 0.5 ? p[0] : 0.5 + (p[0] - 0.5) / 4; });
    const RefinementInput input = input_for(u);
    CHECK_NEAR(total(ResidualEstimator(), mesh, input), 0, 1e-12);
    CHECK_NEAR(total(KellyEstimator(), mesh, input), 0, 1e-12);
    CHECK_NEAR(total(RecoveryEstimator(), mesh, input), 0, 1e-12);
    CHECK_NEAR(total(RecoveryEstimator({.quantity = RecoveryEstimator::Quantity::gradient}), mesh, input), 0, 1e-12);
    const double across_gradient =
        total(RecoveryEstimator({.quantity = RecoveryEstimator::Quantity::gradient, .per_material = false}), mesh, input);
    const double across_flux =
        total(RecoveryEstimator({.quantity = RecoveryEstimator::Quantity::flux, .per_material = false}), mesh, input);
    CHECK(across_gradient > 0.05);
    // The flux eps du/dx is continuous here (u varies in x only), so averaging it is exact.
    CHECK_NEAR(across_flux, 0, 1e-12);
}

// u = sin(pi x) sin(pi y) sin(pi z), f = 3 pi^2 u: the estimates halve when h halves.
void test_rate() {
    const auto exact = [](const Vec3& p) { return std::sin(pi * p[0]) * std::sin(pi * p[1]) * std::sin(pi * p[2]); };
    const auto f = [&](const std::array<double, 3>& x, uint32_t) { return 3 * pi * pi * exact(x); };
    const ResidualEstimator residual;
    const KellyEstimator kelly;
    const RecoveryEstimator recovery;
    std::array<double, 3> previous{};
    for (int n : {4, 8, 16}) {
        const MeshProperties mesh = cube_mesh(n);
        const auto u = interpolate(mesh, exact);
        const RefinementInput input = input_for(u, f);
        const std::array<double, 3> now{total(residual, mesh, input), total(kelly, mesh, input), total(recovery, mesh, input)};
        if (n > 4)
            for (int i = 0; i < 3; ++i) {
                const double ratio = previous[i] / now[i];
                std::cout << "INFO rate n = " << n << ", estimator " << i << ": ratio " << ratio << "\n";
                CHECK(ratio > 1.7 && ratio < 2.3);
            }
        previous = now;
    }
}

void test_metal() {
    const std::vector<MeshProperties::Material> materials{{"oxide", 3.9, 0, false, {}}, {"gate", 1, 0, true, 1.0}};
    const MeshProperties mesh = cube_mesh(4, {0, 0, 0, 0, 0, 0}, {{BcType::Dirichlet, 0, {}}}, materials, 0.5);
    const auto u = interpolate(mesh, [](const Vec3& p) { return std::sin(3 * p[0]) + p[1] * p[1]; });
    const ResidualEstimator residual;
    const KellyEstimator kelly;
    const RecoveryEstimator recovery;
    for (const ErrorEstimator* estimator : {static_cast<const ErrorEstimator*>(&residual),
                                            static_cast<const ErrorEstimator*>(&kelly),
                                            static_cast<const ErrorEstimator*>(&recovery)}) {
        const ErrorEstimate e = estimator->estimate(mesh, input_for(u));
        bool metal_zero = true, oxide_positive = false;
        for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
            if (mesh.elements[k].material == 1) metal_zero = metal_zero && e.eta[k] == 0;
            else oxide_positive = oxide_positive || e.eta[k] > 0;
        }
        CHECK(metal_zero);
        CHECK(oxide_positive);
    }
}

void test_point_sources() {
    const MeshProperties mesh = cube_mesh(4);
    const auto u = interpolate(mesh, [](const Vec3& p) { return p[0] * p[1] + p[2] * p[2]; });
    RefinementInput input = input_for(u);
    input.point_sources.push_back({{0.51, 0.49, 0.52}, 1.0, {}});
    const ErrorEstimate excluded = KellyEstimator().estimate(mesh, input);
    const ErrorEstimate ignored = KellyEstimator({.point_sources = PointSourceRule::ignore}).estimate(mesh, input);
    // Every tet that shares a node with the tet containing the point is zero.
    const uint32_t count = excluded.info.excluded_elements;
    CHECK(count >= 6 && count < mesh.elements.size());
    uint32_t zeros = 0;
    for (double eta : excluded.eta) zeros += eta == 0;
    CHECK(zeros >= count);
    CHECK(ignored.info.excluded_elements == 0);
    CHECK(ignored.total > excluded.total);
    RefinementInput hinted = input;
    hinted.point_sources[0].position = {2, 2, 2};
    const ErrorEstimate outside = KellyEstimator().estimate(mesh, hinted);
    CHECK(outside.info.warnings.size() == 1);
    CHECK(outside.info.excluded_elements == 0);
    hinted.point_sources[0].element = 0;
    CHECK(KellyEstimator().estimate(mesh, hinted).info.excluded_elements > 0);

    // Zero-weight point source must not mask its patch.
    RefinementInput zero_delta = input_for(u);
    zero_delta.point_sources.push_back({{0.51, 0.49, 0.52}, 0.0, {}});
    const ErrorEstimate zero_est = KellyEstimator().estimate(mesh, zero_delta);
    CHECK(zero_est.info.excluded_elements == 0);
    CHECK_NEAR(zero_est.total, ignored.total, 1e-12);

    // Residual and Kelly stage breakdowns must agree with the post-exclusion total.
    const ErrorEstimate res_excl = ResidualEstimator().estimate(mesh, input);
    const double res_parts = std::hypot(res_excl.info.details.at("element_part"),
                                        std::hypot(res_excl.info.details.at("jump_part"), res_excl.info.details.at("boundary_part")));
    CHECK_NEAR(res_parts, res_excl.total, 1e-12);
    const double kelly_parts = std::hypot(excluded.info.details.at("jump_part"), excluded.info.details.at("boundary_part"));
    CHECK_NEAR(kelly_parts, excluded.total, 1e-12);
}

void test_large_scale_and_many_materials() {
    // 65,536 materials must not wrap uint16_t and hang RecoveryEstimator.
    MeshProperties mesh = cube_mesh(1);
    mesh.materials.resize(65536, {"m", 1.0, 0.0, false, {}});
    const auto u = interpolate(mesh, [](const Vec3& p) { return p[0] * p[1]; });
    const ErrorEstimate rec = RecoveryEstimator().estimate(mesh, input_for(u));
    CHECK(std::isfinite(rec.total) && rec.total > 0);

    // Six indicators of 1e200 have a representable norm sqrt(6)*1e200; scaled norm must not overflow.
    class HugeIndicatorEstimator final : public ErrorEstimator {
    public:
        HugeIndicatorEstimator() : ErrorEstimator(PointSourceRule::ignore) {}
        std::string name() const override { return "huge"; }

    private:
        std::vector<double> indicators(const MeshProperties& m, const RefinementInput&, std::span<const uint8_t>,
                                       ErrorEstimate::Info&) const override {
            return std::vector<double>(m.elements.size(), 1e200);
        }
        bool needs_source() const override { return false; }
    };
    const MeshProperties small = cube_mesh(1);
    const std::vector<double> zeros(small.nodes.size(), 0.0);
    const ErrorEstimate huge_est = HugeIndicatorEstimator().estimate(small, input_for(zeros));
    CHECK(std::isfinite(huge_est.total));
    CHECK_NEAR(huge_est.total / (std::sqrt(6.0) * 1e200), 1.0, 1e-12);
}

void test_bad_input() {
    const MeshProperties mesh = cube_mesh(1);
    std::vector<double> u(mesh.nodes.size(), 0);
    std::vector<double> short_u(3, 0);
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, input_for(short_u)), "u has 3 values for 8 nodes");
    u[2] = std::nan("");
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, input_for(u)), "u at node 2 is not finite");
    u[2] = 0;
    CHECK_THROWS(std::invalid_argument, ResidualEstimator().estimate(mesh, RefinementInput{u, {}, {}}), "needs the source f");
    CHECK(KellyEstimator().estimate(mesh, RefinementInput{u, {}, {}}).total == 0);   // Kelly does not need it
    CHECK_THROWS(std::invalid_argument,
                 ResidualEstimator().estimate(mesh, input_for(u, [](const auto&, uint32_t) { return std::nan(""); })),
                 "the source is nan");
    RefinementInput bad = input_for(u);
    bad.point_sources.push_back({{0.5, 0.5, 0.5}, 1, 99});
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, bad), "point source 0 names tet 99");
    bad.point_sources[0] = {{std::nan(""), 0, 0}, 1, {}};
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, bad), "point source 0 is not finite");
    CHECK_THROWS(std::invalid_argument, ResidualEstimator({.tet_degree = 21}), "ResidualEstimator::Settings::tet_degree");
    CHECK_THROWS(std::invalid_argument, KellyEstimator({.face_degree = -1}), "KellyEstimator::Settings::face_degree");
    CHECK_THROWS(std::invalid_argument, RecoveryEstimator({.tet_degree = 99}), "RecoveryEstimator::Settings::tet_degree");
    MeshProperties flat = mesh;
    std::swap(flat.elements[0].node[0], flat.elements[0].node[1]);
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(flat, input_for(u)), "tet 0 has volume");
    // Out-of-bounds indices in MeshProperties must throw std::invalid_argument cleanly.
    MeshProperties bad_node = mesh;
    bad_node.elements[0].node[0] = 99;
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(bad_node, input_for(u)), "tet 0 has node 99");
    MeshProperties bad_mat = mesh;
    bad_mat.elements[0].material = 9;
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(bad_mat, input_for(u)), "tet 0 has material 9");
    MeshProperties bad_bc = mesh;
    bad_bc.boundary_elements[0].boundary_condition = 9;
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(bad_bc, input_for(u)), "boundary element 0 has boundary condition 9");
    MeshProperties bad_bnode = mesh;
    bad_bnode.boundary_elements[0].node[0] = 99;
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(bad_bnode, input_for(u)), "boundary element 0 has node 99");
    MeshProperties bad_eps = mesh;
    bad_eps.materials[0].eps_r = 0.0;
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(bad_eps, input_for(u)), "material 0 has eps_r");
}

int main() {
    test_linear_is_exact();
    test_natural_conditions();
    test_screening();
    test_interface();
    test_rate();
    test_metal();
    test_point_sources();
    test_large_scale_and_many_materials();
    test_bad_input();
    return finish("estimator");
}
