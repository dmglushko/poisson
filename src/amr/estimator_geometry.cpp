#include "estimator_geometry.h"

#include "quadrature.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace amr_detail {

std::vector<TetGeometry> tet_geometry(const MeshProperties& mesh, const std::string& who) {
    std::vector<TetGeometry> result(mesh.elements.size());
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto& n = mesh.elements[k].node;
        const Vec3& p0 = mesh.nodes[n[0]];
        const Vec3 e1 = sub(mesh.nodes[n[1]], p0), e2 = sub(mesh.nodes[n[2]], p0), e3 = sub(mesh.nodes[n[3]], p0);
        const double det = dot(e1, cross(e2, e3));   // 6 V
        TetGeometry& g = result[k];
        g.volume = det / 6;
        if (!(std::isfinite(g.volume) && g.volume > 0))
            throw std::invalid_argument(who + ": tet " + std::to_string(k) + " has volume " +
                                        std::to_string(g.volume) + "; a positive volume is needed");
        // Rows of the inverse of [e1 e2 e3] are the gradients of lambda_1..3.
        const Vec3 c23 = cross(e2, e3), c31 = cross(e3, e1), c12 = cross(e1, e2);
        for (int d = 0; d < 3; ++d) {
            g.grad_lambda[1][d] = c23[d] / det;
            g.grad_lambda[2][d] = c31[d] / det;
            g.grad_lambda[3][d] = c12[d] / det;
            g.grad_lambda[0][d] = -(g.grad_lambda[1][d] + g.grad_lambda[2][d] + g.grad_lambda[3][d]);
        }
        double longest = 0;
        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b) longest = std::max(longest, norm(sub(mesh.nodes[n[a]], mesh.nodes[n[b]])));
        g.diameter = longest;
    }
    return result;
}

Vec3 gradient(const MeshProperties& mesh, const TetGeometry& g, uint32_t element, std::span<const double> u) {
    Vec3 result{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const double value = u[mesh.elements[element].node[i]];
        for (int d = 0; d < 3; ++d) result[d] += value * g.grad_lambda[i][d];
    }
    return result;
}

Vec3 point_at(const MeshProperties& mesh, uint32_t element, const std::array<double, 4>& lambda) {
    Vec3 x{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const Vec3& p = mesh.nodes[mesh.elements[element].node[i]];
        for (int d = 0; d < 3; ++d) x[d] += lambda[i] * p[d];
    }
    return x;
}

Coefficients coefficients(const MeshProperties::Material& material) {
    Coefficients c;
    c.eps = material.eps_r;
    c.c = material.screening_length > 0 ? material.eps_r / (material.screening_length * material.screening_length) : 0;
    c.metal = material.is_metal;
    return c;
}

std::vector<Face> faces(const MeshProperties& mesh) {
    using Key = std::array<uint32_t, 3>;
    const auto sorted = [](Key key) {
        std::sort(key.begin(), key.end());
        return key;
    };
    struct TetFace {
        Key      key;
        uint32_t element;
        uint8_t  local;
    };
    std::vector<TetFace> all;
    all.reserve(4 * mesh.elements.size());
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto& n = mesh.elements[k].node;
        for (uint8_t i = 0; i < 4; ++i) {
            Key face{};
            for (int j = 0, f = 0; j < 4; ++j)
                if (j != i) face[f++] = n[j];
            all.push_back({sorted(face), k, i});
        }
    }
    std::sort(all.begin(), all.end(), [](const TetFace& a, const TetFace& b) {
        return a.key != b.key ? a.key < b.key : a.element < b.element;
    });

    std::vector<std::pair<Key, uint16_t>> boundary;
    boundary.reserve(mesh.boundary_elements.size());
    for (const auto& b : mesh.boundary_elements) boundary.emplace_back(sorted(b.node), b.boundary_condition);
    std::sort(boundary.begin(), boundary.end());

    std::vector<Face> result;
    result.reserve(2 * mesh.elements.size() + 1);
    for (std::size_t i = 0; i < all.size();) {
        const bool pair = i + 1 < all.size() && all[i + 1].key == all[i].key;
        Face face;
        const auto& n = mesh.elements[all[i].element].node;
        for (int j = 0, f = 0; j < 4; ++j)
            if (j != all[i].local) face.node[f++] = n[j];
        face.element = all[i].element;
        face.local = all[i].local;
        if (pair) face.other = all[i + 1].element;
        const auto found = std::lower_bound(boundary.begin(), boundary.end(), std::make_pair(all[i].key, uint16_t{0}));
        if (found != boundary.end() && found->first == all[i].key) face.boundary_condition = found->second;
        result.push_back(face);
        i += pair ? 2 : 1;
    }
    return result;
}

double triangle_diameter(const MeshProperties& mesh, const std::array<uint32_t, 3>& node) {
    const Vec3 &a = mesh.nodes[node[0]], &b = mesh.nodes[node[1]], &c = mesh.nodes[node[2]];
    return std::max({norm(sub(a, b)), norm(sub(b, c)), norm(sub(c, a))});
}

std::optional<uint32_t> locate(const MeshProperties& mesh, const std::vector<TetGeometry>& geometry, const Vec3& x) {
    std::optional<uint32_t> best;
    double best_min = -std::numeric_limits<double>::infinity();
    for (uint32_t k = 0; k < mesh.elements.size(); ++k) {
        const auto& n = mesh.elements[k].node;
        // lambda_i(x) = grad lambda_i . (x - p_j) for a node j != i, where lambda_i = 0.
        double smallest = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i)
            smallest = std::min(smallest, dot(geometry[k].grad_lambda[i], sub(x, mesh.nodes[n[(i + 1) % 4]])));
        if (smallest > best_min) {
            best_min = smallest;
            best = k;
        }
    }
    if (!best || best_min < -1e-9) return std::nullopt;
    return best;
}

void check_degree(int degree, const char* setting) {
    if (degree < 0 || degree > max_quadrature_degree)
        throw std::invalid_argument(std::string(setting) + " must be in [0, " + std::to_string(max_quadrature_degree) +
                                    "], got " + std::to_string(degree));
}

} // namespace amr_detail
