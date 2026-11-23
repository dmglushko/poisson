// The trajectory-displacement estimator (Mitchell-Villarrubia): a Bank-Weiser local problem
// per tet in its six edge bubbles, then the magnitude of the integral of grad e_K.

#include "amr/trajectory_estimator.h"

#include "estimator_geometry.h"
#include "common/quadrature.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

using namespace amr_detail;

namespace {

// The six edges of a tet, as pairs of local nodes; bubble e is 4 lambda_a lambda_b.
constexpr std::array<std::array<int, 2>, 6> edges{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

// Whether edge e lies on local face f (the face opposite node f).
bool on_face(int e, int f) { return edges[e][0] != f && edges[e][1] != f; }

using Vector6 = std::array<double, 6>;
using Matrix6 = std::array<Vector6, 6>;

// Solves A x = b on the unknowns with free[i]; the others are 0. A restricted to the free
// unknowns is symmetric positive definite (the bubbles hold no constant; eps > 0, c >= 0,
// alpha >= 0), so Gaussian elimination with partial pivoting is plenty. Empty if a pivot
// is not clearly positive relative to the diagonal (a singular local problem).
std::optional<Vector6> solve(Matrix6 a, Vector6 b, const std::array<bool, 6>& free) {
    std::array<int, 6> index{};
    int n = 0;
    double diagonal = 0;
    for (int i = 0; i < 6; ++i)
        if (free[i]) {
            index[n++] = i;
            diagonal = std::max(diagonal, std::abs(a[i][i]));
        }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int r = col + 1; r < n; ++r)
            if (std::abs(a[index[r]][index[col]]) > std::abs(a[index[pivot]][index[col]])) pivot = r;
        std::swap(a[index[col]], a[index[pivot]]);
        std::swap(b[index[col]], b[index[pivot]]);
        const double d = a[index[col]][index[col]];
        if (!(std::abs(d) > 1e-12 * diagonal)) return std::nullopt;
        for (int r = col + 1; r < n; ++r) {
            const double m = a[index[r]][index[col]] / d;
            for (int c = col; c < n; ++c) a[index[r]][index[c]] -= m * a[index[col]][index[c]];
            b[index[r]] -= m * b[index[col]];
        }
    }
    Vector6 x{};
    for (int col = n - 1; col >= 0; --col) {
        double s = b[index[col]];
        for (int c = col + 1; c < n; ++c) s -= a[index[col]][index[c]] * x[index[c]];
        x[index[col]] = s / a[index[col]][index[col]];
    }
    return x;
}

// The local index in tet `element` of `node`.
int local_index(const MeshProperties& mesh, uint32_t element, uint32_t node) {
    const auto& n = mesh.elements[element].node;
    for (int i = 0; i < 4; ++i)
        if (n[i] == node) return i;
    throw std::logic_error("trajectory estimator: a face node is not in its tet");
}

// The local face of tet `element` that consists of `face_nodes`: the one opposite its other node.
int local_face(const MeshProperties& mesh, uint32_t element, const std::array<uint32_t, 3>& face_nodes) {
    int sum = 0 + 1 + 2 + 3;
    for (uint32_t node : face_nodes) sum -= local_index(mesh, element, node);
    return sum;
}

} // namespace

TrajectoryEstimator::TrajectoryEstimator() : TrajectoryEstimator(Settings{}) {}

TrajectoryEstimator::TrajectoryEstimator(Settings settings)
    : ErrorEstimator(settings.point_sources, IndicatorSum::values), settings_(settings) {
    check_degree(settings_.tet_degree, "TrajectoryEstimator::Settings::tet_degree");
    check_degree(settings_.face_degree, "TrajectoryEstimator::Settings::face_degree");
    if (!(std::isfinite(settings_.scale) && settings_.scale > 0))
        throw std::invalid_argument("TrajectoryEstimator::Settings::scale must be finite and positive, got " +
                                    std::to_string(settings_.scale));
}

std::vector<double> TrajectoryEstimator::indicators(const MeshProperties& mesh, const RefinementInput& input,
                                                    std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const {
    const std::string who = name() + " estimator";
    const std::vector<TetGeometry> geometry = tet_geometry(mesh, who);
    const std::size_t count = mesh.elements.size();
    std::vector<Coefficients> material(mesh.materials.size());
    for (std::size_t m = 0; m < mesh.materials.size(); ++m) material[m] = coefficients(mesh.materials[m], mesh.screened);
    const auto coef = [&](uint32_t k) -> const Coefficients& { return material[mesh.elements[k].material]; };
    std::vector<Vec3> grad(count);
    for (uint32_t k = 0; k < count; ++k) grad[k] = gradient(mesh, geometry[k], k, input.u);

    // 1. The face terms of every local problem: the load from the faces, the Dirichlet faces
    //    (bit f of `dirichlet`: local face f) and the Robin alpha of each local face.
    std::vector<Vector6> load(count, Vector6{});
    std::vector<uint8_t> dirichlet(count, 0);
    std::vector<std::array<double, 4>> robin(count, std::array<double, 4>{});
    const auto face_rule = triangle_rule(settings_.face_degree);
    uint32_t natural_interior = 0;
    for (const Face& face : faces(mesh)) {
        const uint32_t k = face.element;
        const Coefficients& ck = coef(k);
        const bool metal_k = ck.metal;
        const bool metal_l = face.other && coef(*face.other).metal;
        if (metal_k && (!face.other || metal_l)) continue;   // no dielectric side
        const std::optional<MeshProperties::BoundaryCondition> bc =
            face.boundary_condition ? std::optional(mesh.boundary_conditions[*face.boundary_condition]) : std::nullopt;
        // e = 0 on the face for each dielectric side.
        const auto constrain = [&] {
            if (!metal_k) dirichlet[k] |= uint8_t(1u << face.local);
            if (face.other && !metal_l) dirichlet[*face.other] |= uint8_t(1u << local_face(mesh, *face.other, face.node));
        };
        if (bc && bc->type == MeshProperties::BcType::Dirichlet) {
            constrain();
            continue;
        }
        if (metal_k || metal_l) {
            // A metal at a prescribed voltage is a Dirichlet surface for its dielectric
            // neighbour. A floating metal's potential is part of the solution: skipped, as in
            // the residual estimator.
            const uint32_t metal = metal_k ? k : *face.other;
            if (mesh.materials[mesh.elements[metal].material].voltage) constrain();
            continue;
        }

        // n out of k: -grad lambda of the node opposite the face, normalised.
        const Vec3& g = geometry[k].grad_lambda[face.local];
        const double g_norm = norm(g);
        const Vec3 normal{-g[0] / g_norm, -g[1] / g_norm, -g[2] / g_norm};
        const double area = 0.5 * norm(cross(sub(mesh.nodes[face.node[1]], mesh.nodes[face.node[0]]),
                                             sub(mesh.nodes[face.node[2]], mesh.nodes[face.node[0]])));

        if (face.other) {
            const uint32_t l = *face.other;
            if (bc) ++natural_interior;
            // Half the flux mismatch, the same for both sides (each with its own outward normal):
            // (eps_L grad u_L - eps_K grad u_K) . n_K / 2. A bubble on the face integrates to area / 3.
            const double half = 0.5 * (coef(l).eps * dot(grad[l], normal) - ck.eps * dot(grad[k], normal));
            const int fl = local_face(mesh, l, face.node);
            for (int e = 0; e < 6; ++e) {
                if (on_face(e, face.local)) load[k][e] += half * area / 3;
                if (on_face(e, fl)) load[l][e] += half * area / 3;
            }
            continue;
        }

        // Outer face: Neumann, Robin, or natural (no boundary element: g = 0). face.node lists
        // k's nodes other than face.local in order, so the face barycentrics map to k's.
        const double g_value = bc ? bc->value : 0;
        const double alpha = bc && bc->type == MeshProperties::BcType::Robin ? bc->alpha.value_or(0) : 0;
        robin[k][face.local] = alpha;
        const double flux = ck.eps * dot(grad[k], normal);
        for (const auto& q : face_rule) {
            std::array<double, 4> lambda{};
            for (int j = 0, f = 0; j < 4; ++j)
                if (j != face.local) lambda[j] = q.barycentric[f++];
            double u_h = 0;
            for (int i = 0; i < 3; ++i) u_h += q.barycentric[i] * input.u[face.node[i]];
            const double r = g_value - alpha * u_h - flux;
            for (int e = 0; e < 6; ++e)
                if (on_face(e, face.local))
                    load[k][e] += q.weight * area * r * 4 * lambda[edges[e][0]] * lambda[edges[e][1]];
        }
    }
    if (natural_interior > 0)
        info.warnings.push_back(std::to_string(natural_interior) +
                                " interior faces carry a Neumann or Robin boundary element; they were treated as "
                                "ordinary interior faces");

    // 2. Per tet: the element load and matrix, the 6x6 solve, and |integral of grad e_K|.
    const auto matrix_rule = tet_rule(4);   // exact for the bubble products (degree 4)
    const auto load_rule = tet_rule(settings_.tet_degree);
    const auto robin_rule = triangle_rule(4);
    std::vector<double> eta(count, 0);
    for (uint32_t k = 0; k < count; ++k) {
        const Coefficients& c = coef(k);
        if (c.metal || excluded[k]) continue;
        const TetGeometry& geo = geometry[k];
        const auto& n = mesh.elements[k].node;

        std::array<bool, 6> free{};
        bool any_free = false;
        for (int e = 0; e < 6; ++e) {
            free[e] = true;
            for (int f = 0; f < 4; ++f)
                if ((dirichlet[k] >> f & 1u) && on_face(e, f)) free[e] = false;
            any_free = any_free || free[e];
        }
        if (!any_free) continue;

        Matrix6 a{};
        for (const auto& q : matrix_rule) {
            const auto& l = q.barycentric;
            std::array<Vec3, 6> dphi;
            Vector6 phi;
            for (int e = 0; e < 6; ++e) {
                const int i = edges[e][0], j = edges[e][1];
                phi[e] = 4 * l[i] * l[j];
                for (int d = 0; d < 3; ++d) dphi[e][d] = 4 * (l[j] * geo.grad_lambda[i][d] + l[i] * geo.grad_lambda[j][d]);
            }
            for (int r = 0; r < 6; ++r)
                for (int s = 0; s < 6; ++s)
                    a[r][s] += q.weight * geo.volume * (c.eps * dot(dphi[r], dphi[s]) + c.c * phi[r] * phi[s]);
        }
        for (int f = 0; f < 4; ++f) {
            if (robin[k][f] == 0) continue;
            std::array<uint32_t, 3> face_node{};
            for (int j = 0, m = 0; j < 4; ++j)
                if (j != f) face_node[m++] = n[j];
            const double area = 0.5 * norm(cross(sub(mesh.nodes[face_node[1]], mesh.nodes[face_node[0]]),
                                                 sub(mesh.nodes[face_node[2]], mesh.nodes[face_node[0]])));
            for (const auto& q : robin_rule) {
                std::array<double, 4> l{};
                for (int j = 0, m = 0; j < 4; ++j)
                    if (j != f) l[j] = q.barycentric[m++];
                Vector6 phi;
                for (int e = 0; e < 6; ++e) phi[e] = 4 * l[edges[e][0]] * l[edges[e][1]];
                for (int r = 0; r < 6; ++r)
                    for (int s = 0; s < 6; ++s) a[r][s] += q.weight * area * robin[k][f] * phi[r] * phi[s];
            }
        }

        Vector6 b = load[k];
        for (const auto& q : load_rule) {
            const Vec3 x = point_at(mesh, k, q.barycentric);
            const double f = input.source(x, k);
            if (!std::isfinite(f))
                throw std::invalid_argument(who + ": the source is " + std::to_string(f) + " at (" +
                                            std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " +
                                            std::to_string(x[2]) + ") in tet " + std::to_string(k));
            double u_h = 0;
            for (int i = 0; i < 4; ++i) u_h += q.barycentric[i] * input.u[n[i]];
            const double r = f - c.c * u_h;
            for (int e = 0; e < 6; ++e)
                b[e] += q.weight * geo.volume * r * 4 * q.barycentric[edges[e][0]] * q.barycentric[edges[e][1]];
        }

        const std::optional<Vector6> solved = solve(a, b, free);
        if (!solved) throw std::runtime_error(who + ": the local problem of tet " + std::to_string(k) + " is singular");
        const Vector6& coefficient = *solved;
        // The integral over K of grad(4 lambda_i lambda_j) is |K| (grad lambda_i + grad lambda_j).
        Vec3 integral{};
        for (int e = 0; e < 6; ++e)
            for (int d = 0; d < 3; ++d)
                integral[d] += coefficient[e] * geo.volume *
                               (geo.grad_lambda[edges[e][0]][d] + geo.grad_lambda[edges[e][1]][d]);
        eta[k] = settings_.scale * std::hypot(integral[0], integral[1], integral[2]);   // no overflow when squaring
    }
    return eta;
}
