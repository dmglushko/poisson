// Tests for the error estimators on structured cube meshes. With no solver yet, u_h is the
// nodal interpolant of a chosen function: exact where the function is piecewise linear
// (the estimators must give 0), and O(h) in energy otherwise.

#include "amr/kelly_estimator.h"
#include "amr/recovery_estimator.h"
#include "amr/residual_estimator.h"
#include "amr/trajectory_estimator.h"
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

// One tet, (0,0,0), (1,0,0), (0,1,0), (0,0,1), with Dirichlet faces opposite nodes 0 and 1:
// only the bubble of edge (0, 1) is free. Face (0,1,2) (z = 0) has `bottom` (Robin unless
// replaced), face (0,1,3) (y = 0) is natural.
MeshProperties one_bubble_tet(const Bc& bottom) {
    MeshProperties mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.elements = {{{0, 1, 2, 3}, 0}};
    mesh.boundary_conditions = {{BcType::Dirichlet, 0, {}}, bottom};
    mesh.boundary_elements = {{{1, 3, 2}, 0}, {{0, 2, 3}, 0}, {{0, 2, 1}, 1}};
    mesh.materials = {{"one", 1, 0, false, {}}};
    return mesh;
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
    // Screened by default: c = 4, so f = 4x is exact and f = 0 is wrong.
    CHECK(mesh.screened);
    CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u, [](const auto& x, uint32_t) { return 4 * x[0]; })), 0, 1e-12);
    CHECK(total(ResidualEstimator(), mesh, input_for(u)) > 0.1);
    // Screening disabled: c = 0, so f = 0 is exact and f = 4x is wrong.
    mesh.screened = false;
    CHECK_NEAR(total(ResidualEstimator(), mesh, input_for(u)), 0, 1e-12);
    CHECK(total(ResidualEstimator(), mesh, input_for(u, [](const auto& x, uint32_t) { return 4 * x[0]; })) > 0.1);
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

// Exact values where the integrand varies over the element (the closed forms, not a
// constant residual). One tet (0,0,0), (1,0,0), (0,1,0), (0,0,1), u = x (grad u = (1,0,0)),
// Robin alpha = 2, g = 0 on the bottom z = 0: the flux there is 0, so R_F = -2x, and
// ||R_F||^2 = 4 * (integral of x^2 over the unit right triangle = 1/12) = 1/3. h_F = sqrt(2);
// the other faces are Dirichlet or natural with zero flux.
void test_linear_boundary_residual() {
    const MeshProperties mesh = one_bubble_tet({BcType::Robin, 0, 2.0});
    const std::vector<double> u{0, 1, 0, 0};
    const double expected = std::sqrt(std::sqrt(2.0) / 3);
    const ErrorEstimate kelly = KellyEstimator().estimate(mesh, input_for(u));
    CHECK_NEAR(kelly.total, expected, 1e-14);
    CHECK_NEAR(kelly.info.details.at("boundary_part"), expected, 1e-14);
    const ErrorEstimate residual = ResidualEstimator().estimate(mesh, input_for(u));
    CHECK_NEAR(residual.info.details.at("boundary_part"), expected, 1e-14);
}

// Two tets sharing face (1, 2, 3): A = (0,0,0), (1,0,0), (0,1,0), (0,0,1) with |A| = 1/6 and
// B = (1,0,0), (0,1,0), (0,0,1), (1,1,1) with |B| = 1/3. u = x on A and x + (x + y + z - 1)
// on B, so grad u_B - grad u_A = D = (1, 1, 1). The recovered gradient at the shared nodes is
// (|A| g_A + |B| g_B) / (|A| + |B|) = g_A + 2D/3; at the other nodes it is the tet's own. So
// G - grad u_h is linear with node values (0, 2D/3, 2D/3, 2D/3) on A and (-D/3 x3, 0) on B:
//   eta_A^2 = |A|/20 (3 (4/9) + 4) |D|^2 = 2/15,   eta_B^2 = |B|/20 (3/9 + 1) |D|^2 = 1/15.
void test_recovery_linear_difference() {
    MeshProperties mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    std::array<uint32_t, 4> a{0, 1, 2, 3}, b{1, 2, 3, 4};
    if (signed_volume(mesh, a) < 0) std::swap(a[2], a[3]);
    if (signed_volume(mesh, b) < 0) std::swap(b[2], b[3]);
    mesh.elements = {{a, 0}, {b, 0}};
    mesh.materials = {{"one", 1, 0, false, {}}};
    const std::vector<double> u{0, 1, 0, 0, 3};
    for (const auto quantity : {RecoveryEstimator::Quantity::gradient, RecoveryEstimator::Quantity::flux}) {
        const ErrorEstimate e = RecoveryEstimator({.quantity = quantity}).estimate(mesh, input_for(u));
        CHECK_NEAR(e.eta[0], std::sqrt(2.0 / 15), 1e-14);
        CHECK_NEAR(e.eta[1], std::sqrt(1.0 / 15), 1e-14);
    }
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

    // Fixed-radius exclusion via MeshLocator::query_sphere: larger radius excludes more tets,
    // overlapping spheres do not double-count, zero-weight sources are skipped, and spheres
    // outside the mesh warn without excluding anything.
    const ErrorEstimate r_small =
        KellyEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.05})
            .estimate(mesh, input);
    const ErrorEstimate r_big =
        KellyEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, input);
    CHECK(r_small.info.excluded_elements >= 1);
    CHECK(r_big.info.excluded_elements > r_small.info.excluded_elements);
    CHECK(r_big.info.excluded_elements < mesh.elements.size());
    CHECK(r_big.total < r_small.total);
    CHECK_NEAR(std::hypot(r_big.info.details.at("jump_part"), r_big.info.details.at("boundary_part")), r_big.total, 1e-12);

    // Works identically across all estimators (Residual, Recovery, Trajectory).
    const ErrorEstimate res_rad =
        ResidualEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, input);
    const ErrorEstimate rec_rad =
        RecoveryEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, input);
    const ErrorEstimate traj_rad =
        TrajectoryEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, input);
    CHECK(res_rad.info.excluded_elements == r_big.info.excluded_elements);
    CHECK(rec_rad.info.excluded_elements == r_big.info.excluded_elements);
    CHECK(traj_rad.info.excluded_elements == r_big.info.excluded_elements);
    CHECK_NEAR(std::hypot(res_rad.info.details.at("element_part"),
                          std::hypot(res_rad.info.details.at("jump_part"), res_rad.info.details.at("boundary_part"))),
               res_rad.total, 1e-12);

    // Duplicate/overlapping point sources do not double-count excluded_elements.
    RefinementInput dup = input;
    dup.point_sources.push_back(input.point_sources[0]);
    const ErrorEstimate dup_rad =
        KellyEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, dup);
    CHECK(dup_rad.info.excluded_elements == r_big.info.excluded_elements);

    // Zero-weight source excludes nothing under exclude_radius too.
    const ErrorEstimate zero_rad =
        KellyEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.35})
            .estimate(mesh, zero_delta);
    CHECK(zero_rad.info.excluded_elements == 0);
    CHECK_NEAR(zero_rad.total, ignored.total, 1e-12);

    // Sphere completely outside the mesh emits a warning and excludes 0 tets.
    RefinementInput far_away = input_for(u);
    far_away.point_sources.push_back({{5.0, 5.0, 5.0}, 1.0, {}});
    const ErrorEstimate far_rad =
        KellyEstimator({.point_sources = PointSourceRule::exclude_radius, .point_source_exclusion_radius = 0.2})
            .estimate(mesh, far_away);
    CHECK(far_rad.info.warnings.size() == 1);
    CHECK(far_rad.info.excluded_elements == 0);

    // Invalid exclusion radii are rejected by the constructor.
    CHECK_THROWS(std::invalid_argument,
                 (KellyEstimator(KellyEstimator::Settings{.point_sources = PointSourceRule::exclude_radius,
                                                          .point_source_exclusion_radius = 0.0})),
                 "positive");
    CHECK_THROWS(std::invalid_argument,
                 (KellyEstimator(KellyEstimator::Settings{.point_sources = PointSourceRule::exclude_radius,
                                                          .point_source_exclusion_radius = -0.5})),
                 "positive");
    CHECK_THROWS(std::invalid_argument,
                 (ResidualEstimator(ResidualEstimator::Settings{.point_sources = PointSourceRule::exclude_radius,
                                                                .point_source_exclusion_radius = std::nan("")})),
                 "positive");
    CHECK_THROWS(std::invalid_argument,
                 (TrajectoryEstimator(TrajectoryEstimator::Settings{.point_sources = PointSourceRule::exclude_radius,
                                                                    .point_source_exclusion_radius = 1e200})),
                 "overflow");
    CHECK_THROWS(std::invalid_argument,
                 (RecoveryEstimator(RecoveryEstimator::Settings{.point_sources = PointSourceRule::exclude_patch,
                                                                .point_source_exclusion_radius = -1.0})),
                 "non-negative");
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
    CHECK(ResidualEstimator().estimate(mesh, RefinementInput{u, {}, {}}).total == 0);
    CHECK(KellyEstimator().estimate(mesh, RefinementInput{u, {}, {}}).total == 0);
    CHECK_THROWS(std::invalid_argument,
                 ResidualEstimator().estimate(mesh, input_for(u, [](const auto&, uint32_t) { return std::nan(""); })),
                 "the source is nan");
    RefinementInput bad = input_for(u);
    bad.point_sources.push_back({{0.5, 0.5, 0.5}, 1, 99});
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, bad), "point source 0 names tet 99");
    bad.point_sources[0] = {{std::nan(""), 0, 0}, 1, {}};
    CHECK_THROWS(std::invalid_argument, KellyEstimator().estimate(mesh, bad), "point source 0 is not finite");
    CHECK_THROWS(std::invalid_argument, ResidualEstimator({.tet_degree = 21}), "ResidualEstimator::Settings::tet_degree");
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

// The trajectory estimator: zero where u_h is exact, total = sum of eta, and the right size
// and rate for a smooth u. The exact contribution of K is |integral over K of grad(u - u_h)|,
// and with grad u linear that is |K| |grad u(centroid) - grad u_h|.
void test_trajectory() {
    const TrajectoryEstimator trajectory;
    CHECK(trajectory.name() == "trajectory");
    CHECK(trajectory.indicator_sum() == IndicatorSum::values);
    CHECK(ResidualEstimator().indicator_sum() == IndicatorSum::squares);

    // Exact cases: linear u with Dirichlet data; u = x with Neumann g = 1 and Robin on x = 1;
    // the kinked u with continuous flux across a material interface.
    {
        const MeshProperties mesh = cube_mesh(3);
        const auto u = interpolate(mesh, [](const Vec3& p) { return 2 * p[0] + 3 * p[1] - p[2]; });
        CHECK_NEAR(total(trajectory, mesh, input_for(u)), 0, 1e-12);
    }
    const std::vector<Bc> bcs{{BcType::Dirichlet, 0, {}}, {BcType::Neumann, 1, {}}, {BcType::Neumann, 0, {}},
                              {BcType::Robin, 3, 2.0}, {BcType::Neumann, 0.5, {}}};
    for (uint16_t side : {1, 3}) {
        const MeshProperties mesh = cube_mesh(2, {0, side, 2, 2, 2, 2}, bcs);
        const auto u = interpolate(mesh, [](const Vec3& p) { return p[0]; });
        CHECK_NEAR(total(trajectory, mesh, input_for(u)), 0, 1e-12);
    }
    {
        const MeshProperties mesh = cube_mesh(2, {0, 4, 2, 2, 2, 2}, bcs);   // wrong g = 0.5
        const auto u = interpolate(mesh, [](const Vec3& p) { return p[0]; });
        CHECK(total(trajectory, mesh, input_for(u)) > 1e-3);
    }
    {
        const std::vector<MeshProperties::Material> materials{{"a", 1, 0, false, {}}, {"b", 4, 0, false, {}}};
        const MeshProperties mesh =
            cube_mesh(4, {0, 0, 1, 1, 1, 1}, {{BcType::Dirichlet, 0, {}}, {BcType::Neumann, 0, {}}}, materials, 0.5);
        const auto u = interpolate(mesh, [](const Vec3& p) { return p[0] < 0.5 ? p[0] : 0.5 + (p[0] - 0.5) / 4; });
        CHECK_NEAR(total(trajectory, mesh, input_for(u)), 0, 1e-12);
    }

    // u = x^2 + y z + 2 z^2 with Dirichlet data, f = -lap u = -6.
    const auto exact = [](const Vec3& p) { return p[0] * p[0] + p[1] * p[2] + 2 * p[2] * p[2]; };
    const auto grad_exact = [](const Vec3& p) { return Vec3{2 * p[0], p[2], p[1] + 4 * p[2]}; };
    const auto f = [](const std::array<double, 3>&, uint32_t) { return -6.0; };
    double previous = 0;
    for (int n : {2, 4, 8}) {
        const MeshProperties mesh = cube_mesh(n);
        const auto u = interpolate(mesh, exact);
        const ErrorEstimate e = trajectory.estimate(mesh, input_for(u, f));
        CHECK(e.sum == IndicatorSum::values);
        double sum = 0, true_total = 0;
        for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
            sum += e.eta[k];
            const auto& t = mesh.elements[k].node;
            Vec3 centroid{};
            for (uint32_t node : t)
                for (int d = 0; d < 3; ++d) centroid[d] += mesh.nodes[node][d] / 4;
            // grad u_h from the 3x3 system of the edge differences (Cramer).
            std::array<Vec3, 3> edge;
            Vec3 du;
            for (int i = 0; i < 3; ++i) {
                for (int d = 0; d < 3; ++d) edge[i][d] = mesh.nodes[t[i + 1]][d] - mesh.nodes[t[0]][d];
                du[i] = u[t[i + 1]] - u[t[0]];
            }
            const auto det = [](const std::array<Vec3, 3>& m) {
                return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            };
            const double d0 = det(edge);
            Vec3 grad_h;
            for (int c = 0; c < 3; ++c) {
                auto m = edge;
                for (int i = 0; i < 3; ++i) m[i][c] = du[i];
                grad_h[c] = det(m) / d0;
            }
            const Vec3 g = grad_exact(centroid);
            const double volume = std::abs(d0) / 6;
            true_total += volume * std::sqrt((g[0] - grad_h[0]) * (g[0] - grad_h[0]) + (g[1] - grad_h[1]) * (g[1] - grad_h[1]) +
                                             (g[2] - grad_h[2]) * (g[2] - grad_h[2]));
        }
        CHECK_NEAR(e.total, sum, 1e-12 * sum);
        std::cout << "INFO trajectory n = " << n << ": estimate " << e.total << ", true " << true_total
                  << ", effectivity " << e.total / true_total << "\n";
        CHECK(e.total / true_total > 0.8 && e.total / true_total < 1.25);
        if (n > 2) {
            std::cout << "INFO trajectory rate n = " << n << ": ratio " << previous / e.total << "\n";
            CHECK(previous / e.total > 1.7 && previous / e.total < 2.3);
        }
        previous = e.total;
        // scale multiplies every indicator.
        const ErrorEstimate scaled = TrajectoryEstimator({.scale = 2.5}).estimate(mesh, input_for(u, f));
        CHECK_NEAR(scaled.total, 2.5 * e.total, 1e-12 * e.total);
    }

    // Metal tets are zero, and an empty source means f = 0.
    const std::vector<MeshProperties::Material> materials{{"oxide", 3.9, 0, false, {}}, {"gate", 1, 0, true, 1.0}};
    const MeshProperties metal = cube_mesh(4, {0, 0, 0, 0, 0, 0}, {{BcType::Dirichlet, 0, {}}}, materials, 0.5);
    const auto u = interpolate(metal, [](const Vec3& p) { return std::sin(3 * p[0]) + p[1] * p[1]; });
    const ErrorEstimate e = trajectory.estimate(metal, input_for(u));
    bool metal_zero = true, oxide_positive = false;
    for (uint32_t k = 0; k < metal.elements.size(); ++k) {
        if (metal.elements[k].material == 1) metal_zero = metal_zero && e.eta[k] == 0;
        else oxide_positive = oxide_positive || e.eta[k] > 0;
    }
    CHECK(metal_zero);
    CHECK(oxide_positive);
    CHECK_NEAR(trajectory.estimate(metal, RefinementInput{u, {}, {}}).total, e.total, 1e-12 * e.total);
    for (double scale : {0.0, -1.0, std::nan("")})
        CHECK_THROWS(std::invalid_argument, TrajectoryEstimator({.scale = scale}), "TrajectoryEstimator::Settings::scale");
    CHECK_THROWS(std::invalid_argument, TrajectoryEstimator({.tet_degree = -1}), "TrajectoryEstimator::Settings::tet_degree");
}

// The analytic values of the one free bubble phi = 4 lambda_0 lambda_1 (|K| = 1/6,
// grad lambda_0 = (-1,-1,-1), grad lambda_1 = (1,0,0)):
//   integral of |grad phi|^2 = 16 |K| ((3 + 1) / 10 + 2 (-1) / 20) = 0.8,
//   integral over the bottom (area 1/2) of phi^2 = area 8/45, of phi = area / 3,
//   integral of grad phi = |K| (0, -1, -1), so eta = sqrt(2)/6 |a|.
void test_trajectory_one_bubble() {
    const std::vector<double> zero(4, 0.0);
    // Robin alpha = 2, g = 3 with u = 0, f = 0: load 3 * area / 3 = 0.5, matrix 0.8 + 2 * 8/90.
    const MeshProperties robin = one_bubble_tet({BcType::Robin, 3, 2.0});
    const double a_robin = 0.5 / (0.8 + 2 * 0.5 * 8 / 45);
    CHECK_NEAR(TrajectoryEstimator().estimate(robin, input_for(zero)).total, std::sqrt(2.0) / 6 * a_robin, 1e-12);
    // Neumann g = 3: no mass term.
    const MeshProperties neumann = one_bubble_tet({BcType::Neumann, 3, {}});
    CHECK_NEAR(TrajectoryEstimator().estimate(neumann, input_for(zero)).total, std::sqrt(2.0) / 6 * 0.5 / 0.8, 1e-12);
    // f = 1, natural bottom: load integral of phi = 4 |K| / 20 = 1/30.
    MeshProperties natural = one_bubble_tet({BcType::Neumann, 0, {}});
    const auto one = [](const std::array<double, 3>&, uint32_t) { return 1.0; };
    const double eta_one = TrajectoryEstimator().estimate(natural, input_for(zero, one)).total;
    CHECK_NEAR(eta_one, std::sqrt(2.0) / 6 * (1.0 / 30) / 0.8, 1e-12);
    // Huge and tiny sources: eta scales with f, no overflow or underflow in |integral|.
    const auto huge = [](const std::array<double, 3>&, uint32_t) { return 1e200; };
    const auto tiny = [](const std::array<double, 3>&, uint32_t) { return 1e-200; };
    CHECK_NEAR(TrajectoryEstimator().estimate(natural, input_for(zero, huge)).total / 1e200, eta_one, 1e-12);
    CHECK_NEAR(TrajectoryEstimator().estimate(natural, input_for(zero, tiny)).total * 1e200, eta_one, 1e-12);
    // u = x, Robin alpha = 2, g = 0: the bottom residual is r = -2x = -2 lambda_1, so the load
    // is -2 * 4 * (integral over the bottom of lambda_0 lambda_1^2 = 2 area 2!/5! = 1/60).
    const std::vector<double> u_x{0, 1, 0, 0};
    const MeshProperties robin_zero_g = one_bubble_tet({BcType::Robin, 0, 2.0});
    CHECK_NEAR(TrajectoryEstimator().estimate(robin_zero_g, input_for(u_x)).total,
               std::sqrt(2.0) / 6 * (2 * 4.0 / 60) / (0.8 + 2 * 0.5 * 8 / 45), 1e-14);
    // Screening c = 1 / 0.5^2 = 4 with f = 1, u = 0: the mass term is
    // c * 16 * (integral of lambda_0^2 lambda_1^2 = 6 |K| 2! 2! / 7! = 1/1260) = 4 * 4/315.
    MeshProperties screened = one_bubble_tet({BcType::Neumann, 0, {}});
    screened.materials[0].screening_length = 0.5;
    CHECK_NEAR(TrajectoryEstimator().estimate(screened, input_for(zero, one)).total,
               std::sqrt(2.0) / 6 * (1.0 / 30) / (0.8 + 4 * 4.0 / 315), 1e-14);
    // Boundary data near the overflow limit: the face terms scale with g and stay finite.
    // Kelly squares g, so 4e153 (its square 1.6e307 is still representable); the trajectory
    // load is linear in g, so 9e307 on a tet twice the size (face area 2).
    const auto kelly_g = [&](double g) {
        return KellyEstimator().estimate(one_bubble_tet({BcType::Neumann, g, {}}), input_for(zero)).total;
    };
    CHECK_NEAR(kelly_g(4e153) / 4e153, kelly_g(1), 1e-14);
    const auto doubled = [](const Bc& bottom) {
        MeshProperties mesh = one_bubble_tet(bottom);
        for (auto& p : mesh.nodes)
            for (double& v : p) v *= 2;
        return mesh;
    };
    const auto trajectory_g = [&](double g) {
        return TrajectoryEstimator().estimate(doubled({BcType::Neumann, g, {}}), input_for(zero)).total;
    };
    CHECK_NEAR(trajectory_g(9e307) / 9e307, trajectory_g(1), 1e-14);
    // A huge Robin alpha: the local problem stays regular. Load g area / 3 = 2/3, matrix
    // 2 * 0.8 (stiffness scales with h) + alpha area 8/45, |K| = 4/3, |grad lambda_0 + grad lambda_1| = sqrt(2)/2.
    const double huge_alpha = 1e308;
    CHECK_NEAR(TrajectoryEstimator().estimate(doubled({BcType::Robin, 1, huge_alpha}), input_for(zero)).total /
                   (4.0 / 3 * std::sqrt(2.0) / 2 * (2.0 / 3) / (1.6 + huge_alpha * (2.0 * 8 / 45))),
               1, 1e-10);
    // A negative Robin alpha is rejected (the local problem could be singular).
    CHECK_THROWS(std::invalid_argument, TrajectoryEstimator().estimate(one_bubble_tet({BcType::Robin, 3, -9.0}), input_for(zero)),
                 "negative Robin alpha");

    // A metal tet below the bottom face instead of the boundary element: at a prescribed
    // voltage the face is Dirichlet for the dielectric (the free bubble is dropped, eta = 0);
    // a floating metal is skipped (natural, the f = 1 value above).
    for (const bool fixed : {true, false}) {
        MeshProperties mesh = one_bubble_tet({BcType::Neumann, 0, {}});
        mesh.boundary_elements.pop_back();
        mesh.nodes.push_back({0.2, 0.2, -1});
        std::array<uint32_t, 4> metal{0, 1, 2, 4};
        if (signed_volume(mesh, metal) < 0) std::swap(metal[1], metal[2]);
        mesh.elements.push_back({metal, 1});
        mesh.materials.push_back({"gate", 1, 0, true, fixed ? std::optional(1.0) : std::nullopt});
        const std::vector<double> u(5, 0.0);
        const ErrorEstimate e = TrajectoryEstimator().estimate(mesh, input_for(u, one));
        CHECK(e.eta[1] == 0);
        if (fixed) CHECK(e.eta[0] == 0);
        else CHECK_NEAR(e.eta[0], eta_one, 1e-12);
    }
}

// Three coupled bubbles, so the off-diagonal stiffness, mass and Robin entries all matter.
// The unit tet with a Dirichlet face opposite node 0 (only the bubbles of edges (0, j) are
// free), Robin alpha = 2, g = 0 on the other three faces, eps = 1, c = 1 / 0.5^2 = 4, f = 1,
// u = 0. By symbolic integration of the 3x3 local problem: eta = 7 sqrt(3) / 1072.
void test_trajectory_coupled_bubbles() {
    MeshProperties mesh;
    mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    mesh.elements = {{{0, 1, 2, 3}, 0}};
    mesh.boundary_conditions = {{BcType::Dirichlet, 0, {}}, {BcType::Robin, 0, 2.0}};
    mesh.boundary_elements = {{{1, 3, 2}, 0}, {{0, 2, 1}, 1}, {{0, 1, 3}, 1}, {{0, 3, 2}, 1}};
    mesh.materials = {{"one", 1, 0.5, false, {}}};
    const std::vector<double> zero(4, 0.0);
    const auto one = [](const std::array<double, 3>&, uint32_t) { return 1.0; };
    CHECK_NEAR(TrajectoryEstimator().estimate(mesh, input_for(zero, one)).total, 7 * std::sqrt(3.0) / 1072, 1e-15);
}

// Two tets sharing face (1, 2, 3), permittivities 1 and 4, a piecewise-linear u with a nonzero
// weighted flux jump. Checks the face-to-edge mapping (eta must not depend on the local node
// order) and an interior Dirichlet tag (it decouples the two local problems).
void test_trajectory_two_tets() {
    const auto two_tets = [](double eps_b, bool permuted, bool interior_dirichlet) {
        MeshProperties mesh;
        mesh.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
        std::array<uint32_t, 4> a{0, 1, 2, 3}, b{1, 2, 3, 4};
        if (signed_volume(mesh, a) < 0) std::swap(a[2], a[3]);
        if (signed_volume(mesh, b) < 0) std::swap(b[2], b[3]);
        if (permuted) {   // 3-cycles keep the orientation
            a = {a[1], a[2], a[0], a[3]};
            b = {b[0], b[2], b[3], b[1]};
        }
        mesh.elements = {{a, 0}, {b, 1}};
        mesh.materials = {{"a", 1, 0, false, {}}, {"b", eps_b, 0, false, {}}};
        if (interior_dirichlet) {
            mesh.boundary_conditions = {{BcType::Dirichlet, 0, {}}};
            mesh.boundary_elements = {{{1, 2, 3}, 0}};
        }
        return mesh;
    };
    const std::vector<double> u{0.3, -1, 2, 0.5, 1.7}, u_moved{0.3, -1, 2, 0.5, -5};
    const TrajectoryEstimator trajectory;
    const ErrorEstimate plain = trajectory.estimate(two_tets(4, false, false), input_for(u));
    const ErrorEstimate permuted = trajectory.estimate(two_tets(4, true, false), input_for(u));
    CHECK(plain.eta[0] > 0 && plain.eta[1] > 0);
    for (int k = 0; k < 2; ++k) CHECK_NEAR(permuted.eta[k], plain.eta[k], 1e-12 * plain.eta[k]);
    // Without the tag, tet 0 sees tet 1 through the jump; with it, it does not.
    CHECK(std::abs(trajectory.estimate(two_tets(10, false, false), input_for(u_moved)).eta[0] - plain.eta[0]) >
          1e-6 * plain.eta[0]);
    const double tagged = trajectory.estimate(two_tets(4, false, true), input_for(u)).eta[0];
    CHECK(std::abs(tagged - plain.eta[0]) > 1e-6 * plain.eta[0]);
    CHECK_NEAR(trajectory.estimate(two_tets(10, true, true), input_for(u_moved)).eta[0], tagged, 1e-12 * tagged);
}

// An empty input.source means f = 0 and must match an explicit zero source (using the
// closed-form -c u_h path when screened, and skipping the element integral when unscreened).
void test_empty_source() {
    const std::vector<Bc> bcs{{BcType::Dirichlet, 0, {}}, {BcType::Robin, 1.5, 2.0}, {BcType::Neumann, -0.5, {}}};
    const std::vector<MeshProperties::Material> mats{{"screened", 2.5, 0.4, false, {}}, {"unscreened", 3.0, 0.0, false, {}}};
    MeshProperties mesh = cube_mesh(3, {0, 1, 2, 2, 0, 1}, bcs, mats, 0.5);
    const auto u = interpolate(mesh, [](const Vec3& p) { return p[0] * p[0] + 2 * p[1] * p[2] - 0.3 * p[2] + 0.7; });
    const RefinementInput zero_fn = input_for(u);
    const RefinementInput empty_fn{u, {}, {}};

    const ResidualEstimator residual;
    const TrajectoryEstimator trajectory;

    for (const bool screened : {true, false}) {
        mesh.screened = screened;
        const ErrorEstimate res_fn = residual.estimate(mesh, zero_fn);
        const ErrorEstimate res_empty = residual.estimate(mesh, empty_fn);
        CHECK_NEAR(res_empty.total, res_fn.total, 1e-13 * res_fn.total);
        CHECK_NEAR(res_empty.info.details.at("element_part"), res_fn.info.details.at("element_part"),
                   screened ? 1e-13 * res_fn.info.details.at("element_part") : 1e-15);
        if (!screened) CHECK(res_empty.info.details.at("element_part") == 0);
        else CHECK(res_empty.info.details.at("element_part") > 0);
        for (std::size_t k = 0; k < mesh.elements.size(); ++k)
            CHECK_NEAR(res_empty.eta[k], res_fn.eta[k], 1e-13 * std::max(1.0, res_fn.eta[k]));

        const ErrorEstimate traj_fn = trajectory.estimate(mesh, zero_fn);
        const ErrorEstimate traj_empty = trajectory.estimate(mesh, empty_fn);
        CHECK_NEAR(traj_empty.total, traj_fn.total, 1e-13 * traj_fn.total);
        for (std::size_t k = 0; k < mesh.elements.size(); ++k)
            CHECK_NEAR(traj_empty.eta[k], traj_fn.eta[k], 1e-13 * std::max(1.0, traj_fn.eta[k]));
    }
}

int main() {
    test_linear_is_exact();
    test_natural_conditions();
    test_screening();
    test_interface();
    test_linear_boundary_residual();
    test_recovery_linear_difference();
    test_rate();
    test_metal();
    test_point_sources();
    test_large_scale_and_many_materials();
    test_bad_input();
    test_empty_source();
    test_trajectory();
    test_trajectory_one_bubble();
    test_trajectory_coupled_bubbles();
    test_trajectory_two_tets();
    return finish("estimator");
}
