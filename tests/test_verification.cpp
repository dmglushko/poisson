// Tests for the verification cases: the exact solutions satisfy the equation, the boundary
// and interface conditions, and match closed forms; invalid settings are rejected; the
// generated input files import into a mesh that matches the case (with gmsh, if installed);
// the error evaluator and the sampler measure the right thing.

#include "amr/red_green_refinement.h"
#include "common/constants.h"
#include "importer/importer.h"
#include "verification/case_inputs.h"
#include "verification/error_evaluator.h"
#include "verification/field_sampler.h"
#include "verification/image_cases.h"
#include "verification/layered_cases.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using Vec3 = std::array<double, 3>;
using BcType = MeshProperties::BcType;
using BC = MeshProperties::BoundaryCondition;

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED line " << __LINE__ << ": " #condition "\n"; \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

void check_near(int line, double actual, double expected, double tolerance) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::cerr << "FAILED line " << line << ": " << actual << " != " << expected << " (tolerance " << tolerance << ")\n";
        ++failures;
    }
}
#define CHECK_NEAR(actual, expected, tolerance) check_near(__LINE__, actual, expected, tolerance)

void check_invalid(int line, const std::function<void()>& call, const std::string& expected) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            std::cerr << "FAILED line " << line << ": expected '" << expected << "' in: " << error.what() << "\n";
            ++failures;
        }
        return;
    }
    std::cerr << "FAILED line " << line << ": no std::invalid_argument, expected '" << expected << "'\n";
    ++failures;
}
#define CHECK_INVALID(call, expected) check_invalid(__LINE__, [&] { call; }, expected)

constexpr double K = poisson::constants::coulomb_constant_nm;
const BC dirichlet0{BcType::Dirichlet, 0, std::nullopt};

BC dirichlet(double u) { return {BcType::Dirichlet, u, std::nullopt}; }
BC neumann(double g) { return {BcType::Neumann, g, std::nullopt}; }
BC robin(double alpha, double g) { return {BcType::Robin, g, alpha}; }

VerificationLayer dielectric(const std::string& name, double thickness, double eps, double lambda = 0) {
    return {name, thickness, eps, lambda, false, std::nullopt};
}
VerificationLayer metal(const std::string& name, double thickness, std::optional<double> voltage = std::nullopt) {
    return {name, thickness, 1, 0, true, voltage};
}

// c = eps_r / lambda^2 where screened, else 0.
double reaction(const VerificationCase& c, std::size_t m) {
    const auto& material = c.problem().materials[m];
    return c.problem().screened && material.screening_length > 0
               ? material.eps_r / (material.screening_length * material.screening_length)
               : 0;
}

// The exact solution satisfies -eps lap u + c u = 0 at x (finite differences, relative to the
// size of the terms) and grad_u matches the finite-difference gradient.
void check_pde(int line, const VerificationCase& c, const Vec3& x, std::size_t m, double tolerance = 1e-5) {
    const double h = 1e-4;
    const double u0 = c.u(x, m);
    double lap = 0, scale = std::abs(reaction(c, m) * u0);
    const Vec3 g = c.grad_u(x, m);
    for (int k = 0; k < 3; ++k) {
        Vec3 a = x, b = x;
        a[k] += h;
        b[k] -= h;
        const double ua = c.u(a, m), ub = c.u(b, m);
        lap += (ua - 2 * u0 + ub) / (h * h);
        const double fd = (ua - ub) / (2 * h);
        scale = std::max(scale, std::abs(fd) / 1.0);
        check_near(line, g[k], fd, 1e-6 * (1 + std::abs(fd)));
    }
    const double eps = c.problem().materials[m].eps_r;
    const double residual = -eps * lap + reaction(c, m) * u0;
    if (!(std::abs(residual) <= tolerance * (1 + eps * scale + std::abs(reaction(c, m) * u0)))) {
        std::cerr << "FAILED line " << line << ": PDE residual " << residual << " at (" << x[0] << ", " << x[1] << ", "
                  << x[2] << ") in material " << m << "\n";
        ++failures;
    }
}
#define CHECK_PDE(...) check_pde(__LINE__, __VA_ARGS__)

// ---------------------------------------------------------------- planar layers

void test_planar() {
    // A1: one dielectric between two Dirichlet faces: a straight line.
    {
        PlanarLayersCase c({{dielectric("oxide", 2, 3.9)}, dirichlet(0.5), dirichlet(1.5), 1, 1, true});
        CHECK_NEAR(c.u({0.5, 0.3, 0.3}, 0), 0.75, 1e-14);
        CHECK_NEAR(c.grad_u({0.5, 0.3, 0.3}, 0)[0], 0.5, 1e-14);
        CHECK(c.problem().boundary_conditions.size() == 3);   // 0.5V, 1.5V, D0
        CHECK(c.problem().charges.empty() && c.problem().default_exclusion_radius == 0);
    }
    // A2: Dirichlet V0 at x = 0, Neumann g at x = L: u = V0 + g / eps x.
    {
        PlanarLayersCase c({{dielectric("oxide", 2, 4)}, dirichlet(1), neumann(0.2), 1, 1, true});
        CHECK_NEAR(c.u({1.5, 0, 0}, 0), 1 + 0.2 / 4 * 1.5, 1e-14);
    }
    // A3: Robin at x = L: b = (g - alpha V0) / (eps + alpha L).
    {
        const double eps = 3.9, alpha = 0.6, g = 0.25, V0 = 0.4, L = 2;
        PlanarLayersCase c({{dielectric("oxide", L, eps)}, dirichlet(V0), robin(alpha, g), 1, 1, true});
        const double b = (g - alpha * V0) / (eps + alpha * L);
        CHECK_NEAR(c.u({1.2, 0, 0}, 0), V0 + b * 1.2, 1e-14);
    }
    // A4: a metal at a prescribed voltage at the start, its end face insulated.
    {
        PlanarLayersCase c({{metal("gate", 1, 0.7), dielectric("oxide", 2, 3.9)}, neumann(0), dirichlet(0.1), 1, 1, true});
        CHECK_NEAR(c.u({0.5, 0, 0}, 0), 0.7, 1e-15);
        CHECK_NEAR(c.u({2, 0, 0}, 1), 0.7 + (0.1 - 0.7) * 0.5, 1e-14);
        CHECK(c.problem().materials[0].is_metal && c.problem().materials[0].voltage == 0.7);
    }
    // B: two dielectrics: the same flux D through both.
    {
        const double e1 = 3.9, e2 = 11.7, a = 1, L = 3, V0 = 0, V1 = 1;
        PlanarLayersCase c({{dielectric("a", a, e1), dielectric("b", L - a, e2)}, dirichlet(V0), dirichlet(V1), 1, 1, true});
        const double D = (V1 - V0) / (a / e1 + (L - a) / e2);
        CHECK_NEAR(c.u({0.5, 0, 0}, 0), V0 + D * 0.5 / e1, 1e-14);
        CHECK_NEAR(c.u({2, 0, 0}, 1), V0 + D * a / e1 + D * 1 / e2, 1e-14);
        CHECK_NEAR(c.u({a, 0, 0}, 0), c.u({a, 0, 0}, 1), 1e-14);
    }
    // C1 / C2: a floating metal passes the flux on; a metal at a voltage splits the problem.
    {
        const double e1 = 2, e2 = 5, a = 1, b = 2, L = 4, V0 = 0, V1 = 1;
        PlanarLayersCase floating(
            {{dielectric("left", a, e1), metal("plate", b - a), dielectric("right", L - b, e2)}, dirichlet(V0), dirichlet(V1), 1, 1, true});
        const double D = (V1 - V0) / (a / e1 + (L - b) / e2);
        const double Vm = V0 + D * a / e1;
        CHECK_NEAR(floating.u({1.5, 0, 0}, 1), Vm, 1e-14);
        CHECK_NEAR(floating.u({3, 0, 0}, 2), Vm + D / e2, 1e-14);
        PlanarLayersCase fixed(
            {{dielectric("left", a, e1), metal("plate", b - a, 0.25), dielectric("right", L - b, e2)}, dirichlet(V0), dirichlet(V1), 1, 1, true});
        CHECK_NEAR(fixed.u({0.5, 0, 0}, 0), 0.125, 1e-14);
        CHECK_NEAR(fixed.u({3, 0, 0}, 2), 0.25 + 0.75 * 0.5, 1e-14);
    }
    // D1: screened slab between two voltages.
    {
        const double lam = 0.8, L = 3, V0 = 0.3, V1 = 1.7;
        PlanarLayersCase c({{dielectric("doped", L, 11.7, lam)}, dirichlet(V0), dirichlet(V1), 1, 1, true});
        const double x = 1.1;
        CHECK_NEAR(c.u({x, 0, 0}, 0), (V0 * std::sinh((L - x) / lam) + V1 * std::sinh(x / lam)) / std::sinh(L / lam), 1e-13);
        CHECK_PDE(c, {x, 0.5, 0.5}, 0);
        // Screening off: the same material gives a straight line.
        PlanarLayersCase off({{dielectric("doped", L, 11.7, lam)}, dirichlet(V0), dirichlet(V1), 1, 1, false});
        CHECK_NEAR(off.u({x, 0, 0}, 0), V0 + (V1 - V0) * x / L, 1e-14);
    }
    // D2: Robin alpha = eps / lambda acts as an infinitely long slab: u = V0 exp(-x / lambda).
    {
        const double eps = 3.9, lam = 1.2, V0 = 0.8;
        PlanarLayersCase c({{dielectric("doped", 3, eps, lam)}, dirichlet(V0), robin(eps / lam, 0), 1, 1, true});
        CHECK_NEAR(c.u({2.2, 0, 0}, 0), V0 * std::exp(-2.2 / lam), 1e-13);
    }
    // D3: two screened layers; the interface potential has a closed form.
    {
        const double e1 = 3.9, l1 = 1.3, e2 = 11.7, l2 = 0.7, a = 2, L = 5, V0 = 0.3, V1 = 1.7;
        PlanarLayersCase c({{dielectric("a", a, e1, l1), dielectric("b", L - a, e2, l2)}, dirichlet(V0), dirichlet(V1), 1, 1, true});
        const double s1 = std::sinh(a / l1), c1 = std::cosh(a / l1), s2 = std::sinh((L - a) / l2), c2 = std::cosh((L - a) / l2);
        const double Vm = (e1 / l1 * V0 / s1 + e2 / l2 * V1 / s2) / (e1 / l1 * c1 / s1 + e2 / l2 * c2 / s2);
        CHECK_NEAR(c.u({a, 0, 0}, 0), Vm, 1e-13);
        CHECK_NEAR(c.u({a, 0, 0}, 1), Vm, 1e-13);
        CHECK_NEAR(e1 * c.grad_u({a, 0, 0}, 0)[0], e2 * c.grad_u({a, 0, 0}, 1)[0], 1e-12);
        CHECK_PDE(c, {1, 0.5, 0.5}, 0);
        CHECK_PDE(c, {4, 0.5, 0.5}, 1);
    }
    // A thick, strongly screened layer stays finite (no overflow in the basis).
    {
        PlanarLayersCase c({{dielectric("doped", 400, 1, 0.5)}, dirichlet(1), dirichlet(2), 1, 1, true});
        CHECK_NEAR(c.u({0, 0, 0}, 0), 1, 1e-12);
        CHECK_NEAR(c.u({400, 0, 0}, 0), 2, 1e-12);
        CHECK_NEAR(c.u({200, 0, 0}, 0), 0, 1e-12);
    }

    // Invalid settings.
    CHECK_INVALID(PlanarLayersCase({{}, dirichlet0, dirichlet0, 1, 1, true}), "at least one layer");
    CHECK_INVALID(PlanarLayersCase({{dielectric("1x", 1, 1)}, dirichlet0, dirichlet0, 1, 1, true}), "must start with a letter");
    CHECK_INVALID(PlanarLayersCase({{dielectric("vacuum", 1, 1)}, dirichlet0, dirichlet0, 1, 1, true}), "special meaning");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1), dielectric("a", 1, 1)}, dirichlet0, dirichlet0, 1, 1, true}),
                  "two layers are named 'a'");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 0, 1)}, dirichlet0, dirichlet0, 1, 1, true}), "thickness must be positive");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1), metal("m", 1), metal("n", 1, 1.0), dielectric("b", 1, 1)},
                                    dirichlet0, dirichlet0, 1, 1, true}),
                  "both metals");
    CHECK_INVALID(PlanarLayersCase({{metal("m", 1, 1.0), dielectric("b", 1, 1)}, dirichlet0, dirichlet0, 1, 1, true}),
                  "must then be Neumann 0");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1)}, neumann(0), neumann(1), 1, 1, true}), "no unique solution");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1)}, dirichlet0, BC{BcType::Robin, 0, -1.0}, 1, 1, true}),
                  "alpha must not be negative");
    CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1)}, dirichlet0, BC{BcType::Robin, 0, std::nullopt}, 1, 1, true}),
                  "needs alpha");
    VerificationLayer bad = dielectric("a", 1, 1);
    bad.voltage = 1;
    CHECK_INVALID(PlanarLayersCase({{bad}, dirichlet0, dirichlet0, 1, 1, true}), "is not a metal");
    // All-Neumann is fine with screening.
    PlanarLayersCase screened({{dielectric("a", 1, 1, 0.5)}, neumann(0.1), neumann(0), 1, 1, true});
    CHECK_PDE(screened, {0.5, 0.5, 0.5}, 0);
}

// ---------------------------------------------------------------- spherical layers

void test_spherical() {
    // E1 / E2: a shell between two voltages.
    {
        const double R1 = 1, R2 = 4, V1 = 0.5, V2 = -0.2;
        SphericalLayersCase c({{dielectric("shell", R2 - R1, 3.9)}, R1, dirichlet(V1), dirichlet(V2), 0, true});
        const double B = (V1 - V2) / (1 / R1 - 1 / R2), A = V1 - B / R1;
        CHECK_NEAR(c.u({0, 2.2, 0}, 0), A + B / 2.2, 1e-13);
        const double lam = 0.8;
        SphericalLayersCase s({{dielectric("shell", R2 - R1, 3.9, lam)}, R1, dirichlet(V1), dirichlet(V2), 0, true});
        const double r = 2.2;
        const double expected =
            (V1 * R1 * std::sinh((R2 - r) / lam) + V2 * R2 * std::sinh((r - R1) / lam)) / (r * std::sinh((R2 - R1) / lam));
        CHECK_NEAR(s.u({r / std::sqrt(3.0), r / std::sqrt(3.0), r / std::sqrt(3.0)}, 0), expected, 1e-13);
        CHECK_PDE(s, {1.2, -1.1, 1.0}, 0);
        CHECK(s.problem().charges.empty());
        CHECK_INVALID(SphericalLayersCase({{dielectric("shell", 1, 1)}, 1, std::nullopt, dirichlet0, 0, true}),
                      "needs an inner boundary condition");
        CHECK_INVALID(SphericalLayersCase({{dielectric("shell", 1, 1)}, 1, dirichlet0, dirichlet0, -1, true}),
                      "central charge needs a dielectric ball");
    }
    // F1-F3: the charge at the centre of a ball, with each kind of outer condition.
    {
        const double eps = 3.9, lam = 2, R = 10;
        const int q = -1;
        const auto yukawa = [&](double r) { return q * K / eps * std::exp(-r / lam) / r; };
        const double g = -q * K * std::exp(-R / lam) * (1 / (R * R) + 1 / (lam * R));
        SphericalLayersCase f1({{dielectric("ball", R, eps, lam)}, 0, std::nullopt, dirichlet(yukawa(R)), q, true});
        SphericalLayersCase f2({{dielectric("ball", R, eps, lam)}, 0, std::nullopt, neumann(g), q, true});
        SphericalLayersCase f3({{dielectric("ball", R, eps, lam)}, 0, std::nullopt, robin(eps * (1 / R + 1 / lam), 0), q, true});
        for (const SphericalLayersCase* c : {&f1, &f2, &f3}) {
            CHECK_NEAR(c->u({3, 0, 4}, 0), yukawa(5), 1e-12);
            CHECK_NEAR(c->u({0.01, 0, 0}, 0), yukawa(0.01), 1e-9);
            CHECK_PDE(*c, {2, 3, 1}, 0);
            CHECK(c->problem().charges.size() == 1 && c->problem().charges[0].value == q);
            CHECK(c->problem().default_exclusion_radius == 1);
        }
        // Unscreened Neumann has no unique solution; unscreened Robin does.
        CHECK_INVALID(SphericalLayersCase({{dielectric("ball", R, eps)}, 0, std::nullopt, neumann(g), q, true}), "no unique solution");
        SphericalLayersCase coulomb({{dielectric("ball", R, eps)}, 0, std::nullopt, robin(eps / R, 0), q, true});
        CHECK_NEAR(coulomb.u({0, 0, 5}, 0), q * K / (eps * 5), 1e-12);
    }
    // F4: a dielectric ball inside a shell, grounded outside, unscreened.
    {
        const double e1 = 2, e2 = 5, a = 1.5, R = 4;
        const int q = 1;
        SphericalLayersCase c({{dielectric("core", a, e1), dielectric("shell", R - a, e2)}, 0, std::nullopt, dirichlet0, q, true});
        CHECK_NEAR(c.u({1, 0, 0}, 0), q * K / (e1 * 1) + q * K * (1 / (e2 * a) - 1 / (e2 * R) - 1 / (e1 * a)), 1e-12);
        CHECK_NEAR(c.u({0, 3, 0}, 1), q * K / e2 * (1.0 / 3 - 1 / R), 1e-12);
    }
    // A floating metal shell keeps no net charge: outside it, the field is that of the bare charge.
    {
        const double eps = 2, R = 6;
        const int q = 1;
        SphericalLayersCase c(
            {{dielectric("core", 2, eps), metal("cage", 1), dielectric("outside", R - 3, eps)}, 0, std::nullopt, dirichlet0, q, true});
        CHECK_NEAR(c.u({0, 0, 4}, 2), q * K / eps * (1.0 / 4 - 1 / R), 1e-12);
        CHECK_NEAR(c.u({0, 2.5, 0}, 1), q * K / eps * (1.0 / 3 - 1 / R), 1e-12);
        CHECK(c.grad_u({0, 2.5, 0}, 1) == (Vec3{0, 0, 0}));
    }
    // Screened core and shell with different screening lengths and a charge: the conditions hold.
    {
        SphericalLayersCase c({{dielectric("core", 2, 3, 0.7), dielectric("shell", 3, 9, 1.4)}, 0, std::nullopt, robin(0.3, 0.01), -2, true});
        const double r = 2, h = 1e-6;
        CHECK_NEAR(c.u({0, 0, r}, 0), c.u({0, 0, r}, 1), 1e-12);
        CHECK_NEAR(3 * c.grad_u({0, 0, r}, 0)[2], 9 * c.grad_u({0, 0, r}, 1)[2], 1e-10);
        // One-sided inside the shell: beyond r = 5 the formula is held at its edge value.
        const double du = (c.u({0, 0, 5}, 1) - c.u({0, 0, 5 - 2 * h}, 1)) / (2 * h);
        CHECK_NEAR(9 * du + 0.3 * c.u({0, 0, 5}, 1), 0.01, 1e-7);
        CHECK_NEAR(9 * c.grad_u({0, 0, 5}, 1)[2] + 0.3 * c.u({0, 0, 5}, 1), 0.01, 1e-12);
        CHECK(c.u({0, 0, 5.5}, 1) == c.u({0, 0, 5}, 1));
        CHECK_PDE(c, {0.5, 0.6, 0.7}, 0);
        CHECK_PDE(c, {2.5, 2.0, 1.0}, 1);
    }
}

// ---------------------------------------------------------------- image cases

void test_images() {
    {
        GroundedBallImageCase c({5, 3.9, -1, {0.3, -1.1, 1.6}});
        std::mt19937 rng(7);
        std::normal_distribution<double> n;
        for (int i = 0; i < 10; ++i) {
            Vec3 x{n(rng), n(rng), n(rng)};
            const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            for (auto& v : x) v *= 5 / r;
            CHECK_NEAR(c.u(x, 0), 0, 1e-14);
        }
        CHECK_PDE(c, {-2, 1, -1}, 0);
        // The charge as stored in .cha (float32).
        CHECK(c.problem().charges[0].position[0] == static_cast<double>(0.3f));
        GroundedBallImageCase centred({5, 2, 1, {0, 0, 0}});
        CHECK_NEAR(centred.u({0, 0, 2}, 0), K / 2 * (0.5 - 0.2), 1e-14);
        CHECK_INVALID(GroundedBallImageCase({5, 2, 1, {5, 0, 0}}), "inside the ball");
    }
    {
        HalfSpaceImageCase dir({3.9, 1.5, -1, 2, 20, 20, HalfSpaceImageCase::Plane::Dirichlet, true});
        CHECK_NEAR(dir.u({0.7, -0.3, 0}, 0), 0, 1e-15);
        CHECK_PDE(dir, {0.4, 0.5, 1}, 0);
        HalfSpaceImageCase neu({3.9, 1.5, -1, 2, 20, 20, HalfSpaceImageCase::Plane::Neumann, true});
        CHECK_NEAR(neu.grad_u({0.7, -0.3, 0}, 0)[2], 0, 1e-14);
        CHECK(dir.problem().boundary_error_bound > 0 && dir.problem().boundary_error_bound < 1e-6);
        CHECK(neu.problem().boundary_conditions.size() == 2);
    }
    {
        DielectricInterfaceCase c({3.9, 11.7, 1.5, -1, 1, 20, 20, true});
        const Vec3 x{0.7, 0.4, 0};
        CHECK_NEAR(c.u(x, 0), c.u(x, 1), 1e-14);
        CHECK_NEAR(3.9 * c.grad_u(x, 0)[2], 11.7 * c.grad_u(x, 1)[2], 1e-12);
        CHECK_PDE(c, {0.3, 0.2, 0.5}, 0);
        CHECK_PDE(c, {0.3, 0.2, -0.5}, 1);
        CHECK(c.problem().materials[0].name == "upper" && c.problem().materials[1].name == "lower");
        CHECK(c.material_index("lower") == 1);
        CHECK_INVALID(c.material_index("middle"), "no material named 'middle'");
        CHECK_INVALID(c.u(x, 2), "out of range");
    }
}

// ---------------------------------------------------------------- review regressions

void test_regressions() {
    // A single metal layer at a voltage is that voltage everywhere.
    PlanarLayersCase metal_only({{metal("gate", 1, 5.0)}, neumann(0), neumann(0), 1, 1, true});
    CHECK(metal_only.u({0.5, 0.5, 0.5}, 0) == 5.0);
    SphericalLayersCase metal_ball({{metal("core", 1, 2.0)}, 0, std::nullopt, neumann(0), 0, true});
    CHECK(metal_ball.u({0, 0, 0.5}, 0) == 2.0);

    // Weak screening is unique and solved accurately (no false "no unique solution").
    {
        const double eps = 3.9, lam = 1e4;
        SphericalLayersCase shell({{dielectric("shell", 1, eps, lam)}, 1, neumann(0), neumann(1), 0, true});
        // u carries a large constant (about (r / lambda)^-2), so the difference step is not tiny.
        const double h = 1e-3, du = (shell.u({0, 0, 1.5 + h}, 0) - shell.u({0, 0, 1.5 - h}, 0)) / (2 * h);
        CHECK_NEAR(eps * shell.grad_u({0, 0, 2}, 0)[2], 1, 1e-6);
        CHECK_NEAR(shell.grad_u({0, 0, 1}, 0)[2], 0, 1e-6);
        CHECK_NEAR(du, shell.grad_u({0, 0, 1.5}, 0)[2], 1e-5);
        PlanarLayersCase almost_linear({{dielectric("doped", 1, 1, 1e11)}, dirichlet(0), dirichlet(1), 1, 1, true});
        CHECK_NEAR(almost_linear.u({0.25, 0, 0}, 0), 0.25, 1e-12);
        // Screening this weak as the only thing fixing the potential is rejected, not solved wrongly.
        CHECK_INVALID(PlanarLayersCase({{dielectric("a", 1, 1, 1e8), dielectric("b", 1, 1)}, neumann(-1), neumann(1), 1, 1, true}),
                      "too ill-conditioned");
        // Still rejected when nothing fixes the potential.
        CHECK_INVALID(SphericalLayersCase({{dielectric("shell", 1, eps)}, 1, neumann(0), neumann(1), 0, true}), "no unique solution");
    }

    // Names the importer would rename are rejected.
    CHECK_INVALID(PlanarLayersCase({{dielectric("detectorOxide", 1, 1)}, dirichlet0, dirichlet0, 1, 1, true}), "special meaning");

    // Extreme charges do not overflow; a distance that rounds to 0 in float32 is rejected; the
    // default exclusion radius fits the box.
    HalfSpaceImageCase big({1, 1, std::numeric_limits<int32_t>::min(), 2, 20, 20, HalfSpaceImageCase::Plane::Dirichlet, true});
    CHECK(big.problem().boundary_error_bound > 0);
    GroundedBallImageCase big_ball({5, 1, std::numeric_limits<int32_t>::min(), {1, 0, 0}});
    CHECK(big_ball.u({-2, 0, 0}, 0) < 0);
    CHECK_INVALID(HalfSpaceImageCase({1, 1, -1, 1e-50, 20, 20, HalfSpaceImageCase::Plane::Dirichlet, true}), "0 < distance");
    CHECK_INVALID(DielectricInterfaceCase({1, 4, 1, -1, 1e-50, 20, 20, true}), "0 < distance");
    HalfSpaceImageCase narrow({1, 1, -1, 2, 0.4, 2.2, HalfSpaceImageCase::Plane::Dirichlet, true});
    CHECK(narrow.problem().default_exclusion_radius <= 0.1 + 1e-12);

    // A field with NaN is rejected.
    PlanarLayersCase c({{dielectric("a", 1, 1)}, dirichlet(0), dirichlet(1), 1, 1, true});
    MeshProperties tet;
    tet.nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tet.elements = {{{0, 1, 2, 3}, 0}};
    tet.materials = c.problem().materials;
    tet.neighbours = {MeshProperties::no_neighbours};
    CHECK_INVALID(ErrorEvaluator().evaluate(tet, {std::nan(""), 0, 0, 0}, c), "non-finite");
    CHECK_INVALID(FieldSampler().sample(tet, {0, 0, 0, std::numeric_limits<double>::infinity()}, {{0.1, 0.1, 0.1}}), "non-finite");
    // Its untagged outer faces are reported.
    bool untagged = false;
    for (const auto& m : mesh_mismatches(tet, c)) untagged = untagged || m.find("4 outer tet faces have no boundary condition") != std::string::npos;
    CHECK(untagged);
}

// ---------------------------------------------------------------- input files and meshes

bool have_gmsh() { return std::system("gmsh --version > /dev/null 2>&1") == 0; }

fs::path write(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
    return path;
}

// Writes the case's inputs, meshes them with gmsh and imports the mesh.
MeshProperties mesh_case(const VerificationCase& c, const fs::path& dir, const std::string& tag, double h_max,
                         const std::string& extra = "") {
    const CaseInputsResult inputs = make_case_inputs(c);
    const fs::path geo = write(dir / (tag + ".geo"), inputs.geometry);
    const fs::path yaml = write(dir / (tag + ".yaml"), inputs.materials);
    const fs::path cha = write(dir / (tag + ".cha"), inputs.charges);
    const fs::path msh = dir / (tag + ".msh");
    const std::string command = "gmsh -setnumber h_max " + std::to_string(h_max) + " " + extra + " " + geo.string() +
                                " -3 -format msh22 -nt 1 -o " + msh.string() + " > " + (dir / (tag + ".log")).string() + " 2>&1";
    if (std::system(command.c_str()) != 0) throw std::runtime_error("gmsh failed: " + command);
    Importer::Settings settings;
    settings.screened = c.problem().screened;
    const MeshImportResult imported = Importer(settings).import_mesh(
        msh, {yaml}, c.problem().charges.empty() ? std::nullopt : std::optional<fs::path>(cha));
    for (const auto& warning : imported.info.warnings) std::cerr << "  " << tag << ": " << warning << "\n";
    return imported.mesh;
}

void check_matches(int line, const MeshProperties& mesh, const VerificationCase& c) {
    for (const auto& mismatch : mesh_mismatches(mesh, c)) {
        std::cerr << "FAILED line " << line << ": " << mismatch << "\n";
        ++failures;
    }
}
#define CHECK_MATCHES(mesh, c) check_matches(__LINE__, mesh, c)

void test_inputs_without_gmsh() {
    PlanarLayersCase c({{dielectric("a", 1, 3.9, 2), metal("gate", 1, -0.5), dielectric("b", 1, 2)}, dirichlet(0), robin(0.5, 0.1), 2, 3, true});
    const CaseInputsResult inputs = make_case_inputs(c);
    CHECK(inputs.materials.find("name: a\ndielectric:\n    epsilon_r: 3.9\n    screening_length: 2 nm\n") != std::string::npos);
    CHECK(inputs.materials.find("name: gate\nband_structure:\n    model: metal\n") != std::string::npos);
    CHECK(inputs.geometry.find("Physical Volume(\"gate -0.5V\")") != std::string::npos);
    CHECK(inputs.geometry.find("Physical Surface(\"R0.5,0.1\")") != std::string::npos);
    CHECK(inputs.geometry.find("Physical Surface(\"D0\") = {s_y0[], s_y1[], s_z0[], s_z1[]};") != std::string::npos);
    CHECK(inputs.charges.empty() && inputs.info.materials == 3 && inputs.info.case_name == "planar_layers");

    GroundedBallImageCase ball({5, 3.9, -3, {0.1, 0.2, 0.3}});
    const CaseInputsResult b = make_case_inputs(ball);
    CHECK(b.charges.size() == 16);
    float x = 0;
    int32_t q = 0;
    std::memcpy(&x, b.charges.data(), 4);
    std::memcpy(&q, b.charges.data() + 12, 4);
    CHECK(x == 0.1f && q == -3);
    CHECK(b.geometry.find("If (h_charge > 0)") != std::string::npos);
}

void test_meshes(const fs::path& dir) {
    const ErrorEvaluator evaluator;

    // Linear cases: the exact solution is piecewise linear, so its interpolant is exact on any
    // mesh whose element faces lie on the interfaces, including a floating metal.
    {
        PlanarLayersCase c({{dielectric("left", 1, 2), metal("plate", 0.5), dielectric("right", 1.5, 5)}, dirichlet(0), robin(0.5, 0.1), 1, 1, true});
        const MeshProperties mesh = mesh_case(c, dir, "planar", 0.3);
        CHECK_MATCHES(mesh, c);
        const ExactInterpolationResult exact = interpolate_exact(mesh, c);
        CHECK(exact.info.warnings.empty());
        const ErrorEvaluationResult e = evaluator.evaluate(mesh, exact.u, c);
        CHECK(e.info.l2_error < 1e-12 && e.info.h1_error < 1e-11 && e.info.max_error < 1e-13);
        CHECK(e.info.metal_potential_error < 1e-13 && e.info.metal_elements > 0);
        CHECK_NEAR(e.info.measured_volume, 2.5, 1e-9);   // the metal is not measured
        CHECK(e.info.l2_exact > 0 && e.info.h1_exact > 0);

        // A constant offset shows up as exactly that error.
        std::vector<double> shifted = exact.u;
        for (auto& v : shifted) v += 1e-3;
        const ErrorEvaluationResult s = evaluator.evaluate(mesh, shifted, c);
        CHECK_NEAR(s.info.l2_error, 1e-3 * std::sqrt(2.5), 1e-12);
        CHECK_NEAR(s.info.max_error, 1e-3, 1e-12);
        CHECK_NEAR(s.info.metal_potential_error, 1e-3, 1e-12);
        CHECK(s.info.h1_error < 1e-10);
        // A gradient error of 1e-3 in x over the measured 2.5 nm^3 gives exactly that H1 error.
        std::vector<double> tilted = exact.u;
        for (std::size_t n = 0; n < tilted.size(); ++n) tilted[n] += 1e-3 * mesh.nodes[n][0];
        const ErrorEvaluationResult t = evaluator.evaluate(mesh, tilted, c);
        CHECK_NEAR(t.info.h1_error, 1e-3 * std::sqrt(2.5), 1e-12);

        // Sampling along the slab gives the exact values back.
        const FieldSampler sampler;
        const SamplingResult line = sampler.sample(mesh, exact.u, FieldSampler::line({0.01, 0.5, 0.5}, {2.99, 0.5, 0.5}, 50), &c);
        CHECK(line.info.outside == 0);
        double worst = 0;
        for (std::size_t i = 0; i < 50; ++i) worst = std::max(worst, std::abs(line.samples.u[i] - line.samples.u_exact[i]));
        CHECK(worst < 1e-12);
        const SamplingResult outside = sampler.sample(mesh, exact.u, {{5, 0, 0}});
        CHECK(outside.info.outside == 1 && !outside.samples.found[0] && std::isnan(outside.samples.u[0]));
        CHECK(outside.samples.u_exact.empty());

        // A mesh of another case is caught.
        PlanarLayersCase other({{dielectric("left", 1, 3), metal("plate", 0.5), dielectric("right", 1.5, 5)}, dirichlet(0), robin(0.5, 0.1), 1, 1, true});
        CHECK(!mesh_mismatches(mesh, other).empty());
        CHECK(mesh_mismatches(mesh, other)[0].find("eps_r") != std::string::npos);
    }

    // Swapped boundary conditions, a different layer thickness and a missing boundary triangle
    // are all caught, though the sets of materials and conditions are the same.
    {
        PlanarLayersCase c({{dielectric("a", 1, 2), dielectric("b", 1, 5)}, dirichlet(0), dirichlet(1), 1, 1, true});
        MeshProperties mesh = mesh_case(c, dir, "swap", 0.4);
        CHECK_MATCHES(mesh, c);
        const auto has = [&](const VerificationCase& other, const std::string& text) {
            for (const auto& m : mesh_mismatches(mesh, other))
                if (m.find(text) != std::string::npos) return true;
            return false;
        };
        PlanarLayersCase swapped({{dielectric("a", 1, 2), dielectric("b", 1, 5)}, dirichlet(1), dirichlet(0), 1, 1, true});
        CHECK(has(swapped, "boundary triangles have the wrong condition"));
        PlanarLayersCase thicker({{dielectric("a", 1.2, 2), dielectric("b", 0.8, 5)}, dirichlet(0), dirichlet(1), 1, 1, true});
        CHECK(has(thicker, "tets lie outside the region of their material"));
        PlanarLayersCase wider({{dielectric("a", 1, 2), dielectric("b", 1, 5)}, dirichlet(0), dirichlet(1), 2, 1, true});
        CHECK(has(wider, "do not lie on the boundary of the case"));
        mesh.boundary_elements.pop_back();
        CHECK(has(c, "1 outer tet faces have no boundary condition"));
    }
    // Spheres: a too small ball is caught; uniform AMR refinement (new boundary nodes on
    // flat facets, inside the sphere) still matches; swapped shell conditions are caught.
    {
        SphericalLayersCase big({{dielectric("ball", 10, 3.9, 2)}, 0, std::nullopt, dirichlet(1), 0, true});
        SphericalLayersCase small({{dielectric("ball", 6, 3.9, 2)}, 0, std::nullopt, dirichlet(1), 0, true});
        const MeshProperties mesh = mesh_case(small, dir, "small_ball", 2);
        CHECK_MATCHES(mesh, small);
        const auto mismatches = mesh_mismatches(mesh, big);
        bool reach = false, sphere = false;
        for (const auto& m : mismatches) {
            reach = reach || m.find("reaches radius") != std::string::npos;
            sphere = sphere || m.find("lies on the sphere of radius 10") != std::string::npos;
        }
        CHECK(reach && sphere);

        SphericalLayersCase shell({{dielectric("shell", 0.5, 3.9)}, 10, dirichlet(0), dirichlet(1), 0, true});
        const MeshProperties coarse = mesh_case(shell, dir, "thin_shell", 3);
        CHECK_MATCHES(coarse, shell);
        std::vector<uint32_t> all(coarse.elements.size());
        for (uint32_t e = 0; e < all.size(); ++e) all[e] = e;
        MeshProperties refined = RedGreenRefinement().refine(coarse, all).mesh;
        refined = RedGreenRefinement().refine(refined, std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}).mesh;
        CHECK_MATCHES(refined, shell);
        SphericalLayersCase swapped({{dielectric("shell", 0.5, 3.9)}, 10, dirichlet(1), dirichlet(0), 0, true});
        CHECK(!mesh_mismatches(coarse, swapped).empty());
        CHECK(!mesh_mismatches(refined, swapped).empty());

        // Strong screening: points of the shell on facets of the inner sphere stay finite.
        SphericalLayersCase strong({{dielectric("core", 2, 3.9), dielectric("skin", 2, 3.9, 1e-5)}, 0, std::nullopt, dirichlet(1), 0, true});
        const MeshProperties m = mesh_case(strong, dir, "strong", 1);
        const ErrorEvaluationResult e = evaluator.evaluate(m, interpolate_exact(m, strong).u, strong);
        CHECK(e.info.excluded_points == 0 && std::isfinite(e.info.l2_error));
    }
    // Two volumes of one material name with different voltages are both checked.
    {
        PlanarLayersCase c({{dielectric("a", 1, 2), metal("gate", 0.5, 5.0), dielectric("b", 1, 3)}, dirichlet(0), dirichlet(1), 1, 1, true});
        MeshProperties mesh = mesh_case(c, dir, "gate", 0.5);
        CHECK_MATCHES(mesh, c);
        const std::size_t gate = c.material_index("gate");
        uint16_t index = 0;
        while (mesh.materials[index].name != "gate") ++index;
        MeshProperties::Material other = mesh.materials[index];
        other.voltage = 7.0;
        mesh.materials.push_back(other);
        for (auto& element : mesh.elements)
            if (element.material == index) {
                element.material = static_cast<uint16_t>(mesh.materials.size() - 1);
                break;
            }
        bool voltage = false;
        for (const auto& m : mesh_mismatches(mesh, c)) voltage = voltage || m.find("has voltage 7") != std::string::npos;
        CHECK(voltage && gate == 1);
    }

    // A thin layer still gets every face and volume named.
    {
        PlanarLayersCase c({{dielectric("a", 1, 2), dielectric("film", 1e-3, 5), dielectric("b", 1, 3)}, dirichlet(0), dirichlet(1), 1, 1, true});
        const MeshProperties mesh = mesh_case(c, dir, "thin", 0.3);
        CHECK_MATCHES(mesh, c);
        CHECK(evaluator.evaluate(mesh, interpolate_exact(mesh, c).u, c).info.l2_error < 1e-12);
    }

    // Screened slab: the interpolation error converges at second order in L2.
    {
        PlanarLayersCase c({{dielectric("doped", 3, 11.7, 0.5)}, dirichlet(0.3), dirichlet(1.7), 1, 1, true});
        double previous = 0;
        for (const double h : {0.2, 0.1}) {
            const MeshProperties mesh = mesh_case(c, dir, "screened", h);
            CHECK_MATCHES(mesh, c);
            const ErrorEvaluationResult e = evaluator.evaluate(mesh, interpolate_exact(mesh, c).u, c);
            if (previous > 0) {
                const double rate = std::log2(previous / e.info.l2_error);
                if (rate < 1.6) {
                    std::cerr << "FAILED: screened L2 rate " << rate << "\n";
                    ++failures;
                }
            }
            previous = e.info.l2_error;
        }
    }

    // A charge at the centre of a shell-wrapped ball, and the charge cases with images.
    {
        SphericalLayersCase c({{dielectric("core", 2, 3.9, 2), dielectric("shell", 2, 11.7, 2)}, 0, std::nullopt, dirichlet0, -1, true});
        const MeshProperties mesh = mesh_case(c, dir, "ball", 0.8);
        CHECK_MATCHES(mesh, c);
        const ErrorEvaluationResult e = evaluator.evaluate(mesh, interpolate_exact(mesh, c).u, c);
        CHECK(e.info.excluded_points > 0 && e.info.exclusion_radius == 1);
        const ErrorEvaluationResult none = ErrorEvaluator({0.0, 4}).evaluate(mesh, interpolate_exact(mesh, c).u, c);
        CHECK(!none.info.warnings.empty() && none.info.warnings[0].find("no exclusion radius") != std::string::npos);
        CHECK(e.info.warnings.empty());
        // Tets that touch the charge reach beyond 1 nm at this size; beyond 2 nm the
        // interpolant is close to the exact solution.
        // interpolant converges at second order there.
        const ErrorEvaluator far_evaluator({2.0, 4});
        const ErrorEvaluationResult far = far_evaluator.evaluate(mesh, interpolate_exact(mesh, c).u, c);
        const MeshProperties fine = mesh_case(c, dir, "ball_fine", 0.4);
        const ErrorEvaluationResult far_fine = far_evaluator.evaluate(fine, interpolate_exact(fine, c).u, c);
        const double rate = std::log2(far.info.l2_error / far_fine.info.l2_error);
        CHECK(far.info.exclusion_radius == 2 && far.info.l2_error < 0.1 * far.info.l2_exact);
        CHECK(rate > 1.6);
        CHECK(e.info.l2_error > far.info.l2_error);
        SphericalLayersCase shell({{dielectric("shell", 3, 3.9)}, 1, robin(0.5, 0), dirichlet(1), 0, true});
        CHECK_MATCHES(mesh_case(shell, dir, "shell", 0.8), shell);
    }
    {
        GroundedBallImageCase c({5, 3.9, -1, {1, 0.5, -0.5}});
        const MeshProperties mesh = mesh_case(c, dir, "image", 1.0, "-setnumber h_charge 0.2 -setnumber r_fine 0.5 -setnumber r_coarse 2");
        CHECK_MATCHES(mesh, c);
        HalfSpaceImageCase h({3.9, 1, -1, 1, 6, 6, HalfSpaceImageCase::Plane::Neumann, true});
        CHECK_MATCHES(mesh_case(h, dir, "half_space", 1.5), h);
        DielectricInterfaceCase i({3.9, 11.7, 1, 1, 1, 6, 6, true});
        CHECK_MATCHES(mesh_case(i, dir, "interface", 1.5), i);
    }
    // Screening off at import is caught.
    {
        PlanarLayersCase c({{dielectric("doped", 1, 11.7, 0.5)}, dirichlet(0), dirichlet(1), 1, 1, true});
        MeshProperties mesh = mesh_case(c, dir, "flag", 0.5);
        mesh.screened = false;
        const auto mismatches = mesh_mismatches(mesh, c);
        CHECK(mismatches.size() == 1 && mismatches[0].find("unscreened") != std::string::npos);
    }
}

void test_evaluator_settings() {
    CHECK_INVALID(ErrorEvaluator(ErrorEvaluator::Settings{-1, 99}), "quadrature_degree");
    CHECK_INVALID(FieldSampler::line({0, 0, 0}, {1, 0, 0}, 1), "at least 2 points");
    const auto pts = FieldSampler::line({0, 0, 0}, {1, 2, 3}, 3);
    CHECK((pts[1] == Vec3{0.5, 1, 1.5}) && (pts[2] == Vec3{1, 2, 3}));
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "poisson_test_verification";
    fs::create_directories(dir);
    try {
        test_planar();
        test_spherical();
        test_images();
        test_regressions();
        test_inputs_without_gmsh();
        test_evaluator_settings();
        if (have_gmsh()) test_meshes(dir);
        else std::cout << "gmsh not found: skipping the mesh tests\n";
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << "\n";
        ++failures;
    }
    if (failures == 0) std::cout << "all tests passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
