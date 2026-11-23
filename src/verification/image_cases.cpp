#include "verification/image_cases.h"

#include "case_support.h"
#include "common/constants.h"

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace verification_detail;
using BcType = MeshProperties::BcType;

namespace {

constexpr double K = poisson::constants::coulomb_constant_nm;

const MeshProperties::BoundaryCondition grounded{BcType::Dirichlet, 0, std::nullopt};
const MeshProperties::BoundaryCondition insulated{BcType::Neumann, 0, std::nullopt};

MeshProperties::Material dielectric(const std::string& name, double eps_r, double screening_length) {
    MeshProperties::Material m;
    m.name = name;
    m.eps_r = eps_r;
    m.screening_length = screening_length;
    return m;
}

// Adds a * grad G(|x - c|) to g.
void add_yukawa_gradient(std::array<double, 3>& g, const std::array<double, 3>& x, const std::array<double, 3>& c, double k,
                         double a) {
    const double r = distance(x, c);
    if (r == 0) return;
    const double d = a * yukawa_derivative(r, k) / r;
    for (int i = 0; i < 3; ++i) g[i] += d * (x[i] - c[i]);
}

// The six faces of the box [x0, x1] x [y0, y1] x [z0, z1], as .geo list variables named
// s_<prefix>x0[] ... s_<prefix>z1[].
std::string box_faces(const std::array<double, 3>& lo, const std::array<double, 3>& hi) {
    std::ostringstream geo;
    const char* axis = "xyz";
    for (int a = 0; a < 3; ++a)
        for (int side = 0; side < 2; ++side) {
            std::array<std::string, 3> l, h;
            for (int b = 0; b < 3; ++b) {
                l[b] = number(lo[b]) + " - e";
                h[b] = number(hi[b]) + " + e";
            }
            const double plane = side == 0 ? lo[a] : hi[a];
            l[a] = number(plane) + " - e";
            h[a] = number(plane) + " + e";
            geo << "s_" << axis[a] << side << "[] = Surface In BoundingBox{" << l[0] << ", " << l[1] << ", " << l[2] << ", "
                << h[0] << ", " << h[1] << ", " << h[2] << "};\n";
        }
    return geo.str();
}

}   // namespace

// ---------------------------------------------------------------- GroundedBallImageCase

GroundedBallImageCase::GroundedBallImageCase(Settings settings) : settings_(settings) {
    const std::string_view where = "GroundedBallImageCase";
    require_positive(where, settings_.radius, "radius");
    require_positive(where, settings_.eps_r, "eps_r");
    if (settings_.charge == 0) invalid(where, "the charge must not be zero");
    for (const double c : settings_.position) require_finite(where, c, "the charge position");
    charge_position_ = as_stored(settings_.position);
    const double d = std::sqrt(charge_position_[0] * charge_position_[0] + charge_position_[1] * charge_position_[1] +
                               charge_position_[2] * charge_position_[2]);
    if (d >= settings_.radius) invalid(where, "the charge must lie inside the ball");
    if (d > 0) {
        image_charge_ = -static_cast<double>(settings_.charge) * settings_.radius / d;
        const double scale = settings_.radius * settings_.radius / (d * d);
        image_position_ = {charge_position_[0] * scale, charge_position_[1] * scale, charge_position_[2] * scale};
    }

    problem_.materials = {dielectric("medium", settings_.eps_r, 0)};
    problem_.boundary_conditions = {grounded};
    problem_.charges = {{charge_position_, settings_.charge}};
    problem_.screened = true;   // no screening length, so the flag changes nothing
    problem_.default_exclusion_radius = std::min(1.0, 0.5 * (settings_.radius - d));

    std::ostringstream geo;
    geo << geo_header() << "Sphere(1) = {0, 0, 0, " << number(settings_.radius) << "};\n"
        << "Physical Volume(\"medium\") = {1};\n"
        << "Physical Surface(\"" << surface_name(grounded) << "\") = {1};\n"
        << geo_refinement({charge_position_});
    problem_.geometry = geo.str();
}

double GroundedBallImageCase::exact_u(const std::array<double, 3>& x, std::size_t) const {
    const double a = K / settings_.eps_r;
    const double r = distance(x, charge_position_);
    if (image_charge_ == 0) return a * settings_.charge * (1 / r - 1 / settings_.radius);   // charge at the centre
    return a * (settings_.charge / r + image_charge_ / distance(x, image_position_));
}

std::array<double, 3> GroundedBallImageCase::exact_grad(const std::array<double, 3>& x, std::size_t) const {
    const double a = K / settings_.eps_r;
    std::array<double, 3> g{0, 0, 0};
    add_yukawa_gradient(g, x, charge_position_, 0, a * settings_.charge);
    if (image_charge_ != 0) add_yukawa_gradient(g, x, image_position_, 0, a * image_charge_);
    return g;
}

// ---------------------------------------------------------------- HalfSpaceImageCase

HalfSpaceImageCase::HalfSpaceImageCase(Settings settings) : settings_(settings) {
    const std::string_view where = "HalfSpaceImageCase";
    require_positive(where, settings_.eps_r, "eps_r");
    require_non_negative(where, settings_.screening_length, "screening_length");
    if (settings_.charge == 0) invalid(where, "the charge must not be zero");
    require_positive(where, settings_.distance, "distance");
    require_positive(where, settings_.half_width, "half_width");
    require_positive(where, settings_.height, "height");
    d_ = as_stored({0, 0, settings_.distance})[2];
    if (d_ <= 0 || d_ >= settings_.height) invalid(where, "the charge must lie inside the box (0 < distance < height)");
    k_ = settings_.screened && settings_.screening_length > 0 ? 1 / settings_.screening_length : 0;

    const bool dirichlet = settings_.plane == Plane::Dirichlet;
    problem_.materials = {dielectric("medium", settings_.eps_r, settings_.screening_length)};
    problem_.boundary_conditions = dirichlet ? std::vector{grounded} : std::vector{grounded, insulated};
    problem_.charges = {{{0, 0, d_}, settings_.charge}};
    problem_.screened = settings_.screened;
    problem_.default_exclusion_radius =
        std::min({1.0, 0.5 * d_, 0.5 * settings_.half_width, 0.5 * (settings_.height - d_)});
    // On the far faces r+ >= nearest and r- >= r+, so |G(r+) -+ G(r-)| <= (1 or 2) G(nearest).
    const double nearest = std::min(settings_.half_width, settings_.height - d_);
    problem_.boundary_error_bound =
        std::abs(static_cast<double>(settings_.charge)) * K / settings_.eps_r * (dirichlet ? 1 : 2) * yukawa(nearest, k_);

    const double W = settings_.half_width, H = settings_.height;
    std::ostringstream geo;
    geo << geo_header() << "Box(1) = {" << number(-W) << ", " << number(-W) << ", 0, " << number(2 * W) << ", "
        << number(2 * W) << ", " << number(H) << "};\n"
        << "e = " << number(0.25 * std::min(W, H)) << ";\n"
        << box_faces({-W, -W, 0}, {W, W, H}) << "Physical Volume(\"medium\") = {1};\n";
    std::vector<PhysicalGroup> surfaces;
    add_to_group(surfaces, surface_name(dirichlet ? grounded : insulated), "s_z0[]");
    for (const char* face : {"s_x0[]", "s_x1[]", "s_y0[]", "s_y1[]", "s_z1[]"}) add_to_group(surfaces, surface_name(grounded), face);
    geo << geo_physicals("Surface", surfaces) << geo_refinement({{0, 0, d_}});
    problem_.geometry = geo.str();
}

double HalfSpaceImageCase::exact_u(const std::array<double, 3>& x, std::size_t) const {
    const double image = settings_.plane == Plane::Dirichlet ? -1 : 1;
    return settings_.charge * K / settings_.eps_r *
           (yukawa(distance(x, {0, 0, d_}), k_) + image * yukawa(distance(x, {0, 0, -d_}), k_));
}

std::array<double, 3> HalfSpaceImageCase::exact_grad(const std::array<double, 3>& x, std::size_t) const {
    const double a = settings_.charge * K / settings_.eps_r;
    const double image = settings_.plane == Plane::Dirichlet ? -1 : 1;
    std::array<double, 3> g{0, 0, 0};
    add_yukawa_gradient(g, x, {0, 0, d_}, k_, a);
    add_yukawa_gradient(g, x, {0, 0, -d_}, k_, a * image);
    return g;
}

// ---------------------------------------------------------------- DielectricInterfaceCase

DielectricInterfaceCase::DielectricInterfaceCase(Settings settings) : settings_(settings) {
    const std::string_view where = "DielectricInterfaceCase";
    require_positive(where, settings_.eps_upper, "eps_upper");
    require_positive(where, settings_.eps_lower, "eps_lower");
    require_non_negative(where, settings_.screening_length, "screening_length");
    if (settings_.charge == 0) invalid(where, "the charge must not be zero");
    require_positive(where, settings_.distance, "distance");
    require_positive(where, settings_.half_width, "half_width");
    require_positive(where, settings_.half_height, "half_height");
    d_ = as_stored({0, 0, settings_.distance})[2];
    if (d_ <= 0 || d_ >= settings_.half_height)
        invalid(where, "the charge must lie inside the box (0 < distance < half_height)");
    k_ = settings_.screened && settings_.screening_length > 0 ? 1 / settings_.screening_length : 0;

    problem_.materials = {dielectric("upper", settings_.eps_upper, settings_.screening_length),
                          dielectric("lower", settings_.eps_lower, settings_.screening_length)};
    problem_.boundary_conditions = {grounded};
    problem_.charges = {{{0, 0, d_}, settings_.charge}};
    problem_.screened = settings_.screened;
    problem_.default_exclusion_radius =
        std::min({1.0, 0.5 * d_, 0.5 * settings_.half_width, 0.5 * (settings_.half_height - d_)});
    const double e1 = settings_.eps_upper, e2 = settings_.eps_lower, rho = (e1 - e2) / (e1 + e2);
    const double nearest = std::min(settings_.half_width, settings_.half_height - d_);
    problem_.boundary_error_bound =
        std::abs(static_cast<double>(settings_.charge)) * K * std::max((1 + std::abs(rho)) / e1, 2 / (e1 + e2)) * yukawa(nearest, k_);

    const double W = settings_.half_width, H = settings_.half_height;
    std::ostringstream geo;
    geo << geo_header() << "Box(1) = {" << number(-W) << ", " << number(-W) << ", 0, " << number(2 * W) << ", "
        << number(2 * W) << ", " << number(H) << "};\n"
        << "Box(2) = {" << number(-W) << ", " << number(-W) << ", " << number(-H) << ", " << number(2 * W) << ", "
        << number(2 * W) << ", " << number(H) << "};\n"
        << "BooleanFragments{ Volume{1, 2}; Delete; }{}\n"
        << "e = " << number(0.25 * std::min(W, H)) << ";\n"
        << "v_upper[] = Volume In BoundingBox{" << number(-W) << " - e, " << number(-W) << " - e, -e, " << number(W)
        << " + e, " << number(W) << " + e, " << number(H) << " + e};\n"
        << "v_lower[] = Volume In BoundingBox{" << number(-W) << " - e, " << number(-W) << " - e, " << number(-H)
        << " - e, " << number(W) << " + e, " << number(W) << " + e, e};\n"
        << box_faces({-W, -W, -H}, {W, W, H})
        << "Physical Volume(\"upper\") = {v_upper[]};\n"
        << "Physical Volume(\"lower\") = {v_lower[]};\n";
    std::vector<PhysicalGroup> surfaces;
    for (const char* face : {"s_x0[]", "s_x1[]", "s_y0[]", "s_y1[]", "s_z0[]", "s_z1[]"})
        add_to_group(surfaces, surface_name(grounded), face);
    geo << geo_physicals("Surface", surfaces) << geo_refinement({{0, 0, d_}});
    problem_.geometry = geo.str();
}

double DielectricInterfaceCase::exact_u(const std::array<double, 3>& x, std::size_t material) const {
    const double e1 = settings_.eps_upper, e2 = settings_.eps_lower, rho = (e1 - e2) / (e1 + e2);
    const double g_plus = yukawa(distance(x, {0, 0, d_}), k_);
    if (material == 0) return settings_.charge * K / e1 * (g_plus + rho * yukawa(distance(x, {0, 0, -d_}), k_));
    return 2.0 * settings_.charge * K / (e1 + e2) * g_plus;
}

std::array<double, 3> DielectricInterfaceCase::exact_grad(const std::array<double, 3>& x, std::size_t material) const {
    const double e1 = settings_.eps_upper, e2 = settings_.eps_lower, rho = (e1 - e2) / (e1 + e2);
    std::array<double, 3> g{0, 0, 0};
    if (material == 0) {
        add_yukawa_gradient(g, x, {0, 0, d_}, k_, settings_.charge * K / e1);
        add_yukawa_gradient(g, x, {0, 0, -d_}, k_, settings_.charge * K / e1 * rho);
    } else {
        add_yukawa_gradient(g, x, {0, 0, d_}, k_, 2.0 * settings_.charge * K / (e1 + e2));
    }
    return g;
}

// ---------------------------------------------------------------- geometry checks

std::vector<std::string> GroundedBallImageCase::geometry_mismatches(const MeshProperties& mesh,
                                                                    const std::vector<std::optional<std::size_t>>& material) const {
    return spherical_geometry_mismatches(mesh, material, {"medium"}, {{0, settings_.radius}}, {{settings_.radius, grounded}});
}

std::vector<std::string> HalfSpaceImageCase::geometry_mismatches(const MeshProperties& mesh,
                                                                 const std::vector<std::optional<std::size_t>>& material) const {
    const double W = settings_.half_width, H = settings_.height, tol = 1e-9 * std::max(W, H);
    const Vec3 lo{-W, -W, 0}, hi{W, W, H};
    const auto condition_on = [&](const std::array<Vec3, 3>& triangle) -> std::optional<MeshProperties::BoundaryCondition> {
        if (!inside_box(triangle, lo, hi, tol)) return std::nullopt;
        const auto [axis, side] = box_face(triangle, lo, hi, tol);
        if (axis < 0) return std::nullopt;
        if (axis == 2 && side == 0) return settings_.plane == Plane::Dirichlet ? grounded : insulated;
        return grounded;
    };
    const auto fits = [&](const std::array<Vec3, 4>& tet, std::size_t) { return inside_box(tet, lo, hi, tol); };
    return flat_geometry_mismatches(mesh, material, condition_on, fits);
}

std::vector<std::string> DielectricInterfaceCase::geometry_mismatches(const MeshProperties& mesh,
                                                                      const std::vector<std::optional<std::size_t>>& material) const {
    const double W = settings_.half_width, H = settings_.half_height, tol = 1e-9 * std::max(W, H);
    const Vec3 lo{-W, -W, -H}, hi{W, W, H};
    const auto condition_on = [&](const std::array<Vec3, 3>& triangle) -> std::optional<MeshProperties::BoundaryCondition> {
        if (!inside_box(triangle, lo, hi, tol) || box_face(triangle, lo, hi, tol).first < 0) return std::nullopt;
        return grounded;
    };
    const auto fits = [&](const std::array<Vec3, 4>& tet, std::size_t m) {
        return m == 0 ? inside_box(tet, {-W, -W, 0}, hi, tol) : inside_box(tet, lo, {W, W, 0}, tol);
    };
    return flat_geometry_mismatches(mesh, material, condition_on, fits);
}
