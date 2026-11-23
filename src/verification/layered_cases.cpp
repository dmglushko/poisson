#include "verification/layered_cases.h"

#include "case_support.h"
#include "common/constants.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace verification_detail;
using BcType = MeshProperties::BcType;

namespace {

enum class Geometry { Planar, Spherical };

// One layer of the solved problem. s is the coordinate across the layers: x (planar) or r.
struct SolvedLayer {
    double start = 0, end = 0;
    double eps = 1;
    double k = 0;                 // 1 / screening length, or 0
    bool   metal = false;
    bool   ball = false;          // spherical layer that starts at r = 0
    std::optional<double> voltage;
    double particular = 0;        // q K / eps for a ball holding the central charge
    int    first = -1;            // index of the first unknown of this layer
    int    count = 0;             // number of unknowns: 2 (dielectric), 1 (ball, floating metal), 0
    double c[2] = {0, 0};         // solved coefficients (dielectric) or c[0] = potential (metal)
};

// The two solutions g_j of g'' = k^2 g on a layer, and g_j'. For a thin or weakly screened
// layer (k L < 1) the centred pair cosh(k t), sinh(k t) / k (t from the layer middle) keeps
// them distinct as k -> 0 (they become 1 and t). Otherwise the decaying pair
// exp(-k (s - start)), exp(-k (end - s)) keeps them at most 1, so thick layers cannot overflow.
void slab_basis(const SolvedLayer& layer, int j, double s, double& g, double& dg) {
    const double k = layer.k;
    if (k * (layer.end - layer.start) < 1) {
        const double t = s - 0.5 * (layer.start + layer.end);
        if (j == 0) {
            g = std::cosh(k * t);
            dg = k * std::sinh(k * t);
        } else {
            g = k == 0 ? t : std::sinh(k * t) / k;
            dg = std::cosh(k * t);
        }
    } else if (j == 0) {
        g = std::exp(-k * (s - layer.start));
        dg = -k * g;
    } else {
        g = std::exp(-k * (layer.end - s));
        dg = k * g;
    }
}

// Basis function j of a dielectric layer and its derivative d/ds at s: g_j (planar) or g_j / r
// (spherical, since r u satisfies the planar equation), or the regular solution of a ball.
void basis(Geometry geometry, const SolvedLayer& layer, int j, double s, double& value, double& derivative) {
    const double k = layer.k;
    if (geometry == Geometry::Planar) {
        slab_basis(layer, j, s, value, derivative);
        return;
    }
    if (layer.ball) {   // the regular solution only: a constant, or sinh(k r) / r (scaled by 2 exp(-k end))
        if (k == 0) {
            value = 1;
            derivative = 0;
            return;
        }
        const double ks = k * s, scale = 2 * std::exp(-k * layer.end);
        if (ks < 1e-3) {
            value = scale * k * (1 + ks * ks / 6);
            derivative = scale * k * k * ks / 3;
        } else if (ks < 1) {
            value = scale * std::sinh(ks) / s;
            derivative = scale * (ks * std::cosh(ks) - std::sinh(ks)) / (s * s);
        } else {
            const double a = std::exp(-k * (layer.end - s)), b = std::exp(-k * (layer.end + s));
            value = (a - b) / s;
            derivative = k * (a + b) / s - (a - b) / (s * s);
        }
        return;
    }
    double g = 0, dg = 0;
    slab_basis(layer, j, s, g, dg);
    value = g / s;
    derivative = dg / s - g / (s * s);
}

// The central charge's own potential q K exp(-k r) / (eps r) and its derivative.
void particular(const SolvedLayer& layer, double s, double& value, double& derivative) {
    value = derivative = 0;
    if (layer.particular == 0) return;
    value = layer.particular * yukawa(s, layer.k);
    derivative = layer.particular * yukawa_derivative(s, layer.k);
}

// A linear form in the unknowns: sum of terms + constant.
struct Form {
    std::vector<std::pair<int, double>> terms;
    double constant = 0;

    Form& add(const Form& other, double factor) {
        for (const auto& [i, a] : other.terms) terms.push_back({i, a * factor});
        constant += other.constant * factor;
        return *this;
    }
};

Form value_form(Geometry geometry, const SolvedLayer& layer, double s) {
    Form f;
    if (layer.metal) {
        if (layer.count == 1) f.terms.push_back({layer.first, 1});
        else f.constant = *layer.voltage;
        return f;
    }
    for (int j = 0; j < layer.count; ++j) {
        double v = 0, d = 0;
        basis(geometry, layer, j, s, v, d);
        f.terms.push_back({layer.first + j, v});
    }
    double v = 0, d = 0;
    particular(layer, s, v, d);
    f.constant = v;
    return f;
}

Form derivative_form(Geometry geometry, const SolvedLayer& layer, double s) {
    Form f;
    if (layer.metal) return f;
    for (int j = 0; j < layer.count; ++j) {
        double v = 0, d = 0;
        basis(geometry, layer, j, s, v, d);
        f.terms.push_back({layer.first + j, d});
    }
    double v = 0, d = 0;
    particular(layer, s, v, d);
    f.constant = d;
    return f;
}

// The total flux eps du/ds through the plane (planar) or sphere (r^2 eps du/dr, without 4 pi).
Form flux_form(Geometry geometry, const SolvedLayer& layer, double s) {
    Form f;
    const double area = geometry == Geometry::Planar ? 1 : s * s;
    return f.add(derivative_form(geometry, layer, s), layer.eps * area);
}

// A boundary condition at s on the outside of `layer`; outward = +1 at the end, -1 at the start.
Form boundary_form(Geometry geometry, const SolvedLayer& layer, double s, double outward,
                   const MeshProperties::BoundaryCondition& condition) {
    Form f;
    switch (condition.type) {
    case BcType::Dirichlet: f.add(value_form(geometry, layer, s), 1); break;
    case BcType::Neumann: f.add(derivative_form(geometry, layer, s), outward * layer.eps); break;
    case BcType::Robin:
        f.add(derivative_form(geometry, layer, s), outward * layer.eps).add(value_form(geometry, layer, s), *condition.alpha);
        break;
    }
    f.constant -= condition.value;   // form = 0
    return f;
}

void validate_layers(std::string_view where, const std::vector<VerificationLayer>& layers) {
    if (layers.empty()) invalid(where, "at least one layer is needed");
    std::set<std::string> names;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        require_material_name(where, layer.name);
        if (!names.insert(layer.name).second) invalid(where, "two layers are named '" + layer.name + "'");
        require_positive(where, layer.thickness, "layer thickness");
        if (!layer.metal) {
            require_positive(where, layer.eps_r, "eps_r");
            require_non_negative(where, layer.screening_length, "screening_length");
            if (layer.voltage) invalid(where, "layer '" + layer.name + "' has a voltage but is not a metal");
        } else if (layer.voltage) {
            require_finite(where, *layer.voltage, "the metal voltage");
        }
        if (layer.metal && i > 0 && layers[i - 1].metal)
            invalid(where, "layers '" + layers[i - 1].name + "' and '" + layer.name + "' are both metals; merge them");
    }
}

void require_insulated(std::string_view where, const MeshProperties::BoundaryCondition& condition, const char* face) {
    if (condition.type != BcType::Neumann || condition.value != 0)
        invalid(where, std::string("a metal layer ends at the ") + face + " face, which must then be Neumann 0");
}

// Sets up the unknowns and the equations, solves them, and stores the coefficients.
void solve(std::string_view where, Geometry geometry, std::vector<SolvedLayer>& layers,
           const std::optional<MeshProperties::BoundaryCondition>& start_condition,
           const MeshProperties::BoundaryCondition& end_condition) {
    int unknowns = 0;
    for (auto& layer : layers) {
        layer.first = unknowns;
        layer.count = layer.metal ? (layer.voltage ? 0 : 1) : (layer.ball ? 1 : 2);
        unknowns += layer.count;
    }

    std::vector<Form> equations;
    for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
        const SolvedLayer& a = layers[i];
        const SolvedLayer& b = layers[i + 1];
        const double s = a.end;
        equations.push_back(value_form(geometry, a, s).add(value_form(geometry, b, s), -1));
        if (!a.metal && !b.metal) equations.push_back(flux_form(geometry, a, s).add(flux_form(geometry, b, s), -1));
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i].metal || layers[i].voltage) continue;
        // No net charge on a floating metal: the flux into it equals the flux out of it.
        Form f;
        if (i + 1 < layers.size()) f.add(flux_form(geometry, layers[i + 1], layers[i].end), 1);
        if (i > 0) f.add(flux_form(geometry, layers[i - 1], layers[i].start), -1);
        if (f.terms.empty()) invalid(where, "a floating metal needs a dielectric next to it");
        equations.push_back(f);
    }
    if (start_condition && !layers.front().metal)
        equations.push_back(boundary_form(geometry, layers.front(), layers.front().start, -1, *start_condition));
    if (!layers.back().metal)
        equations.push_back(boundary_form(geometry, layers.back(), layers.back().end, +1, end_condition));

    // Unique exactly when something fixes the additive constant: the difference of two
    // solutions has zero energy, so it is a constant, which a Dirichlet end, a Robin end with
    // alpha > 0, a metal at a voltage or a screened dielectric forces to 0.
    const auto fixes = [](const MeshProperties::BoundaryCondition& c) {
        return c.type == BcType::Dirichlet || (c.type == BcType::Robin && *c.alpha > 0);
    };
    bool fixed = false;        // by a condition or a metal voltage
    double screening = 0;      // max (k L)^2 over the dielectrics, L the size of the whole problem
    const double size = layers.back().end - layers.front().start;
    for (auto& layer : layers) {
        if (layer.metal && layer.voltage) layer.c[0] = *layer.voltage;
        fixed = fixed || (layer.metal && layer.voltage);
        if (!layer.metal) screening = std::max(screening, layer.k * size * layer.k * size);
    }
    if (start_condition && !layers.front().metal) fixed = fixed || fixes(*start_condition);
    if (!layers.back().metal) fixed = fixed || fixes(end_condition);
    if (!fixed && screening == 0)
        invalid(where, "the problem has no unique solution (nothing fixes the potential: add a Dirichlet or Robin "
                       "condition, a metal with a voltage, or screening)");
    // Fixed by screening alone, the constant is set by terms of order (L / lambda)^2 relative
    // to the rest; below 1e-10 they are lost to rounding (the basis degenerates to the
    // unscreened one), so the answer would be wrong while looking accurate.
    if (!fixed && screening < 1e-10)
        invalid(where, "the problem is too ill-conditioned to solve accurately: only screening fixes the potential, and "
                       "the screening length is too large for the size (L / lambda)^2 = " + number(screening));

    if (unknowns == 0) return;
    if (static_cast<int>(equations.size()) != unknowns)
        throw std::logic_error(std::string(where) + ": " + std::to_string(equations.size()) + " equations for " +
                               std::to_string(unknowns) + " unknowns");

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(unknowns, unknowns);
    Eigen::VectorXd rhs(unknowns);
    for (int row = 0; row < unknowns; ++row) {
        for (const auto& [col, a] : equations[row].terms) matrix(row, col) += a;
        rhs(row) = -equations[row].constant;
    }
    // Rows can differ in scale by orders of magnitude (fluxes carry r^2); equilibrate them.
    for (int row = 0; row < unknowns; ++row) {
        const double norm = matrix.row(row).cwiseAbs().maxCoeff();
        if (norm > 0) {
            matrix.row(row) /= norm;
            rhs(row) /= norm;
        }
    }
    // Unique, but possibly too ill-conditioned for double precision: when only a screening
    // length far above the size fixes the potential, the constant is set by a term of order
    // (L / lambda)^2 and is lost to rounding although the residual stays small.
    const Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    const Eigen::VectorXd z = lu.solve(rhs);
    const double residual = (matrix * z - rhs).cwiseAbs().maxCoeff();
    const double scale = matrix.cwiseAbs().rowwise().sum().maxCoeff() * z.cwiseAbs().maxCoeff() + rhs.cwiseAbs().maxCoeff();
    if (!z.allFinite() || !(residual <= 1e-9 * scale) || !(lu.rcond() >= 1e-12))
        invalid(where, "the problem is too ill-conditioned to solve accurately (estimated reciprocal condition number " +
                           number(lu.rcond()) + ")");

    for (auto& layer : layers) {
        if (layer.metal) {
            if (!layer.voltage) layer.c[0] = z(layer.first);
            continue;
        }
        for (int j = 0; j < layer.count; ++j) layer.c[j] = z(layer.first + j);
    }
}

double layer_value(Geometry geometry, const SolvedLayer& layer, double s) {
    if (layer.metal) return layer.c[0];
    double u = 0, v = 0, d = 0;
    for (int j = 0; j < layer.count; ++j) {
        basis(geometry, layer, j, s, v, d);
        u += layer.c[j] * v;
    }
    particular(layer, s, v, d);
    return u + v;
}

double layer_derivative(Geometry geometry, const SolvedLayer& layer, double s) {
    if (layer.metal) return 0;
    double du = 0, v = 0, d = 0;
    for (int j = 0; j < layer.count; ++j) {
        basis(geometry, layer, j, s, v, d);
        du += layer.c[j] * d;
    }
    particular(layer, s, v, d);
    return du + d;
}

std::vector<SolvedLayer> make_layers(const std::vector<VerificationLayer>& layers, double start, bool screened) {
    std::vector<SolvedLayer> out;
    double s = start;
    for (const auto& layer : layers) {
        SolvedLayer solved;
        solved.start = s;
        solved.end = s + layer.thickness;
        solved.eps = layer.metal ? 1 : layer.eps_r;
        solved.k = !layer.metal && screened && layer.screening_length > 0 ? 1 / layer.screening_length : 0;
        solved.metal = layer.metal;
        solved.voltage = layer.voltage;
        out.push_back(solved);
        s = solved.end;
    }
    return out;
}

std::vector<MeshProperties::Material> layer_materials(const std::vector<VerificationLayer>& layers) {
    std::vector<MeshProperties::Material> out;
    for (const auto& layer : layers) {
        MeshProperties::Material m;
        m.name = layer.name;
        m.is_metal = layer.metal;
        m.voltage = layer.voltage;
        if (!layer.metal) {
            m.eps_r = layer.eps_r;
            m.screening_length = layer.screening_length;
        }
        out.push_back(m);
    }
    return out;
}

double min_thickness(const std::vector<VerificationLayer>& layers) {
    double t = layers.front().thickness;
    for (const auto& layer : layers) t = std::min(t, layer.thickness);
    return t;
}

}   // namespace

// ---------------------------------------------------------------- PlanarLayersCase

struct PlanarLayersCase::Solution {
    std::vector<SolvedLayer> layers;
};

PlanarLayersCase::~PlanarLayersCase() = default;

PlanarLayersCase::PlanarLayersCase(Settings settings) : settings_(std::move(settings)) {
    const std::string_view where = "PlanarLayersCase";
    validate_layers(where, settings_.layers);
    require_boundary_condition(where, settings_.left);
    require_boundary_condition(where, settings_.right);
    require_positive(where, settings_.width, "width");
    require_positive(where, settings_.height, "height");
    if (settings_.layers.front().metal) require_insulated(where, settings_.left, "x = 0");
    if (settings_.layers.back().metal) require_insulated(where, settings_.right, "x = L");

    solution_ = std::make_unique<Solution>();
    solution_->layers = make_layers(settings_.layers, 0, settings_.screened);
    solve(where, Geometry::Planar, solution_->layers, settings_.left, settings_.right);

    problem_.materials = layer_materials(settings_.layers);
    problem_.screened = settings_.screened;
    const MeshProperties::BoundaryCondition insulated{BcType::Neumann, 0, std::nullopt};
    add_condition(problem_.boundary_conditions, settings_.left);
    add_condition(problem_.boundary_conditions, settings_.right);
    add_condition(problem_.boundary_conditions, insulated);

    const double L = solution_->layers.back().end, W = settings_.width, H = settings_.height;
    // Selection boxes reach a quarter of the smallest size: past OpenCASCADE's tolerance, short of any other face.
    const double e = 0.25 * std::min({min_thickness(settings_.layers), W, H});
    std::ostringstream geo;
    geo << geo_header();
    for (std::size_t i = 0; i < settings_.layers.size(); ++i) {
        const auto& layer = solution_->layers[i];
        geo << "Box(" << i + 1 << ") = {" << number(layer.start) << ", 0, 0, " << number(layer.end - layer.start) << ", "
            << number(W) << ", " << number(H) << "};\n";
    }
    if (settings_.layers.size() > 1) geo << "BooleanFragments{ Volume{1:" << settings_.layers.size() << "}; Delete; }{}\n";
    geo << "e = " << number(e) << ";\n";
    std::vector<PhysicalGroup> volumes, surfaces;
    for (std::size_t i = 0; i < settings_.layers.size(); ++i) {
        const auto& layer = solution_->layers[i];
        geo << "v_" << i << "[] = Volume In BoundingBox{" << number(layer.start) << " - e, -e, -e, " << number(layer.end)
            << " + e, " << number(W) << " + e, " << number(H) << " + e};\n";
        add_to_group(volumes, volume_name(problem_.materials[i]), "v_" + std::to_string(i) + "[]");
    }
    const std::string l = number(L), w = number(W), h = number(H);
    geo << "s_left[] = Surface In BoundingBox{-e, -e, -e, e, " << w << " + e, " << h << " + e};\n"
        << "s_right[] = Surface In BoundingBox{" << l << " - e, -e, -e, " << l << " + e, " << w << " + e, " << h << " + e};\n"
        << "s_y0[] = Surface In BoundingBox{-e, -e, -e, " << l << " + e, e, " << h << " + e};\n"
        << "s_y1[] = Surface In BoundingBox{-e, " << w << " - e, -e, " << l << " + e, " << w << " + e, " << h << " + e};\n"
        << "s_z0[] = Surface In BoundingBox{-e, -e, -e, " << l << " + e, " << w << " + e, e};\n"
        << "s_z1[] = Surface In BoundingBox{-e, -e, " << h << " - e, " << l << " + e, " << w << " + e, " << h << " + e};\n";
    add_to_group(surfaces, surface_name(settings_.left), "s_left[]");
    add_to_group(surfaces, surface_name(settings_.right), "s_right[]");
    for (const char* side : {"s_y0[]", "s_y1[]", "s_z0[]", "s_z1[]"}) add_to_group(surfaces, surface_name(insulated), side);
    geo << geo_physicals("Volume", volumes) << geo_physicals("Surface", surfaces);
    problem_.geometry = geo.str();
}

double PlanarLayersCase::exact_u(const std::array<double, 3>& x, std::size_t material) const {
    return layer_value(Geometry::Planar, solution_->layers[material], x[0]);
}

std::array<double, 3> PlanarLayersCase::exact_grad(const std::array<double, 3>& x, std::size_t material) const {
    return {layer_derivative(Geometry::Planar, solution_->layers[material], x[0]), 0, 0};
}

// ---------------------------------------------------------------- SphericalLayersCase

struct SphericalLayersCase::Solution {
    std::vector<SolvedLayer> layers;
};

SphericalLayersCase::~SphericalLayersCase() = default;

SphericalLayersCase::SphericalLayersCase(Settings settings) : settings_(std::move(settings)) {
    const std::string_view where = "SphericalLayersCase";
    validate_layers(where, settings_.layers);
    require_non_negative(where, settings_.inner_radius, "inner_radius");
    require_boundary_condition(where, settings_.outer);
    if (settings_.inner_radius > 0) {
        if (!settings_.inner) invalid(where, "a shell (inner_radius > 0) needs an inner boundary condition");
        require_boundary_condition(where, *settings_.inner);
        if (settings_.layers.front().metal) require_insulated(where, *settings_.inner, "inner");
    } else if (settings_.inner) {
        invalid(where, "an inner boundary condition needs inner_radius > 0");
    }
    if (settings_.layers.back().metal) require_insulated(where, settings_.outer, "outer");
    if (settings_.charge != 0 && (settings_.inner_radius > 0 || settings_.layers.front().metal))
        invalid(where, "the central charge needs a dielectric ball (inner_radius = 0)");

    solution_ = std::make_unique<Solution>();
    solution_->layers = make_layers(settings_.layers, settings_.inner_radius, settings_.screened);
    SolvedLayer& first = solution_->layers.front();
    first.ball = settings_.inner_radius == 0;
    if (settings_.charge != 0) first.particular = settings_.charge * poisson::constants::coulomb_constant_nm / first.eps;
    solve(where, Geometry::Spherical, solution_->layers, settings_.inner, settings_.outer);

    problem_.materials = layer_materials(settings_.layers);
    problem_.screened = settings_.screened;
    if (settings_.inner) add_condition(problem_.boundary_conditions, *settings_.inner);
    add_condition(problem_.boundary_conditions, settings_.outer);
    if (settings_.charge != 0) {
        problem_.charges.push_back({{0, 0, 0}, settings_.charge});
        problem_.default_exclusion_radius = std::min(1.0, 0.5 * settings_.layers.front().thickness);
    }

    // Spheres for every radius, fragmented into conforming shells. Shell i is what lies
    // within the box of its outer radius but not within the box of its inner radius.
    const auto& layers = solution_->layers;
    std::vector<double> radii;
    if (settings_.inner_radius > 0) radii.push_back(settings_.inner_radius);
    for (const auto& layer : layers) radii.push_back(layer.end);
    double smallest = min_thickness(settings_.layers);
    if (settings_.inner_radius > 0) smallest = std::min(smallest, settings_.inner_radius);
    const double e = 0.25 * smallest;

    std::ostringstream geo;
    geo << geo_header();
    for (std::size_t i = 0; i < radii.size(); ++i) geo << "Sphere(" << i + 1 << ") = {0, 0, 0, " << number(radii[i]) << "};\n";
    if (settings_.inner_radius > 0) {
        geo << "BooleanDifference{ Volume{2:" << radii.size() << "}; Delete; }{ Volume{1}; Delete; }\n";
        if (radii.size() > 2) geo << "BooleanFragments{ Volume{:}; Delete; }{}\n";
    } else if (radii.size() > 1) {
        geo << "BooleanFragments{ Volume{1:" << radii.size() << "}; Delete; }{}\n";
    }
    geo << "e = " << number(e) << ";\n";
    const auto in_box = [&](const char* kind, const std::string& var, double r) {
        const std::string a = number(r);
        geo << var << "[] = " << kind << " In BoundingBox{-" << a << " - e, -" << a << " - e, -" << a << " - e, " << a
            << " + e, " << a << " + e, " << a << " + e};\n";
    };
    for (std::size_t i = 0; i < radii.size(); ++i) {
        in_box("Volume", "vb_" + std::to_string(i), radii[i]);
        in_box("Surface", "sb_" + std::to_string(i), radii[i]);
    }
    std::vector<PhysicalGroup> volumes, surfaces;
    const std::size_t offset = settings_.inner_radius > 0 ? 1 : 0;   // radii[offset + i] is the outer radius of layer i
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string var = "v_" + std::to_string(i);
        geo << var << "[] = vb_" << offset + i << "[];\n";
        if (offset + i > 0) geo << var << "[] -= {vb_" << offset + i - 1 << "[]};\n";
        add_to_group(volumes, volume_name(problem_.materials[i]), var + "[]");
    }
    geo << "s_outer[] = sb_" << radii.size() - 1 << "[];\n";
    if (radii.size() > 1) geo << "s_outer[] -= {sb_" << radii.size() - 2 << "[]};\n";
    add_to_group(surfaces, surface_name(settings_.outer), "s_outer[]");
    if (settings_.inner) {
        geo << "s_inner[] = sb_0[];\n";
        add_to_group(surfaces, surface_name(*settings_.inner), "s_inner[]");
    }
    geo << geo_physicals("Volume", volumes) << geo_physicals("Surface", surfaces);
    std::vector<Vec3> refine;
    for (const auto& charge : problem_.charges) refine.push_back(charge.position);
    geo << geo_refinement(refine);
    problem_.geometry = geo.str();
}

// A meshed sphere puts points of a layer slightly outside it (a shell tet on a flat facet of
// its inner sphere). There the layer's formula is taken at the nearest radius of the layer:
// extending it instead would overflow for strong screening (exp(k * depth)).
double SphericalLayersCase::exact_u(const std::array<double, 3>& x, std::size_t material) const {
    const auto& layer = solution_->layers[material];
    const double r = std::clamp(radius(x), layer.start, layer.end);
    return layer_value(Geometry::Spherical, layer, r);
}

std::array<double, 3> SphericalLayersCase::exact_grad(const std::array<double, 3>& x, std::size_t material) const {
    const auto& layer = solution_->layers[material];
    const double distance = radius(x);
    if (distance == 0) return {0, 0, 0};
    const double d = layer_derivative(Geometry::Spherical, layer, std::clamp(distance, layer.start, layer.end)) / distance;
    return {d * x[0], d * x[1], d * x[2]};
}

// ---------------------------------------------------------------- geometry checks

std::vector<std::string> PlanarLayersCase::geometry_mismatches(const MeshProperties& mesh,
                                                               const std::vector<std::optional<std::size_t>>& material) const {
    const double L = solution_->layers.back().end, W = settings_.width, H = settings_.height;
    const double tol = 1e-9 * std::max({L, W, H});
    const auto condition_on = [&](const std::array<Vec3, 3>& triangle) -> std::optional<MeshProperties::BoundaryCondition> {
        if (on_plane(triangle, 0, 0, tol)) return settings_.left;
        if (on_plane(triangle, 0, L, tol)) return settings_.right;
        if (!inside_box(triangle, {0, 0, 0}, {L, W, H}, tol) || box_face(triangle, {0, 0, 0}, {L, W, H}, tol).first < 0)
            return std::nullopt;
        return MeshProperties::BoundaryCondition{BcType::Neumann, 0, std::nullopt};
    };
    const auto fits = [&](const std::array<Vec3, 4>& tet, std::size_t m) {
        const auto& layer = solution_->layers[m];
        return inside_box(tet, {layer.start, 0, 0}, {layer.end, W, H}, tol);
    };
    return flat_geometry_mismatches(mesh, material, condition_on, fits);
}

std::vector<std::string> SphericalLayersCase::geometry_mismatches(const MeshProperties& mesh,
                                                                  const std::vector<std::optional<std::size_t>>& material) const {
    std::vector<std::string> names;
    std::vector<std::pair<double, double>> layers;
    for (std::size_t m = 0; m < solution_->layers.size(); ++m) {
        names.push_back(problem_.materials[m].name);
        layers.emplace_back(solution_->layers[m].start, solution_->layers[m].end);
    }
    std::vector<SphereBoundary> spheres{{solution_->layers.back().end, settings_.outer}};
    if (settings_.inner) spheres.push_back({settings_.inner_radius, *settings_.inner});
    return spherical_geometry_mismatches(mesh, material, names, layers, spheres);
}
