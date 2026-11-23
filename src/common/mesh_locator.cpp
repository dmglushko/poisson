#include "common/mesh_locator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

using Vec3 = std::array<double, 3>;

constexpr uint32_t leaf_size = 8;
constexpr double   bary_tol = 1e-9;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
bool finite_vec(const Vec3& v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

double dist2_point_box(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    double d2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        if (p[d] < lo[d]) {
            const double diff = lo[d] - p[d];
            d2 += diff * diff;
        } else if (p[d] > hi[d]) {
            const double diff = p[d] - hi[d];
            d2 += diff * diff;
        }
    }
    return d2;
}

bool point_in_padded_box(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    for (int d = 0; d < 3; ++d) {
        const double span = hi[d] - lo[d];
        const double pad = bary_tol * std::max(1.0, std::max(span, std::max(std::abs(lo[d]), std::abs(hi[d]))));
        if (p[d] < lo[d] - pad || p[d] > hi[d] + pad) return false;
    }
    return true;
}

bool boxes_overlap(const Vec3& a_lo, const Vec3& a_hi, const Vec3& b_lo, const Vec3& b_hi) {
    for (int d = 0; d < 3; ++d)
        if (a_hi[d] < b_lo[d] || b_hi[d] < a_lo[d]) return false;
    return true;
}

// Squared Euclidean distance from `p` to the closed triangle (a, b, c) (Ericson, §5.1.5).
double dist2_point_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return dot(ap, ap);

    const Vec3 bp = sub(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return dot(bp, bp);

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const Vec3 q{a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]};
        const Vec3 diff = sub(p, q);
        return dot(diff, diff);
    }

    const Vec3 cp = sub(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return dot(cp, cp);

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const Vec3 q{a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]};
        const Vec3 diff = sub(p, q);
        return dot(diff, diff);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3 bc = sub(c, b);
        const Vec3 q{b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]};
        const Vec3 diff = sub(p, q);
        return dot(diff, diff);
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    const Vec3 q{a[0] + v * ab[0] + w * ac[0], a[1] + v * ab[1] + w * ac[1], a[2] + v * ab[2] + w * ac[2]};
    const Vec3 diff = sub(p, q);
    return dot(diff, diff);
}

} // namespace

MeshLocator::MeshLocator(const MeshProperties& mesh) : mesh_(&mesh) {
    const std::size_t n_nodes = mesh.nodes.size();
    const std::size_t n_tets = mesh.elements.size();

    for (std::size_t i = 0; i < n_nodes; ++i)
        if (!finite_vec(mesh.nodes[i]))
            throw std::invalid_argument("MeshLocator: node " + std::to_string(i) + " has non-finite coordinates");

    if (mesh.neighbours.size() != n_tets)
        throw std::invalid_argument("MeshLocator: neighbours has " + std::to_string(mesh.neighbours.size()) +
                                    " entries for " + std::to_string(n_tets) + " tets");
    for (std::size_t k = 0; k < n_tets; ++k)
        for (const uint32_t nb : mesh.neighbours[k])
            if (nb != MeshProperties::no_neighbour && nb >= n_tets)
                throw std::invalid_argument("MeshLocator: tet " + std::to_string(k) + " names neighbour " +
                                            std::to_string(nb) + " out of range");

    if (n_tets == 0) return;

    grad_lambda_.resize(n_tets);
    tet_boxes_.resize(n_tets);
    std::vector<Vec3> centroids(n_tets);

    for (std::size_t k = 0; k < n_tets; ++k) {
        const auto& n = mesh.elements[k].node;
        for (uint32_t node : n)
            if (node >= n_nodes)
                throw std::invalid_argument("MeshLocator: tet " + std::to_string(k) + " has node " +
                                            std::to_string(node) + ", but the mesh has " + std::to_string(n_nodes) +
                                            " nodes");

        const Vec3& p0 = mesh.nodes[n[0]];
        const Vec3& p1 = mesh.nodes[n[1]];
        const Vec3& p2 = mesh.nodes[n[2]];
        const Vec3& p3 = mesh.nodes[n[3]];
        const Vec3 e1 = sub(p1, p0), e2 = sub(p2, p0), e3 = sub(p3, p0);
        const double det = dot(e1, cross(e2, e3));
        if (!(std::isfinite(det) && det > 0.0))
            throw std::invalid_argument("MeshLocator: tet " + std::to_string(k) + " has volume " +
                                        std::to_string(det / 6.0) + "; a positive volume is needed");

        const Vec3 c23 = cross(e2, e3), c31 = cross(e3, e1), c12 = cross(e1, e2);
        auto& g = grad_lambda_[k];
        for (int d = 0; d < 3; ++d) {
            g[1][d] = c23[d] / det;
            g[2][d] = c31[d] / det;
            g[3][d] = c12[d] / det;
            g[0][d] = -(g[1][d] + g[2][d] + g[3][d]);
        }

        Aabb box{p0, p0};
        for (int i = 1; i < 4; ++i) {
            const Vec3& p = mesh.nodes[n[i]];
            for (int d = 0; d < 3; ++d) {
                box.lo[d] = std::min(box.lo[d], p[d]);
                box.hi[d] = std::max(box.hi[d], p[d]);
            }
        }
        tet_boxes_[k] = box;
        for (int d = 0; d < 3; ++d) centroids[k][d] = 0.25 * (p0[d] + p1[d] + p2[d] + p3[d]);
    }

    order_.resize(n_tets);
    std::iota(order_.begin(), order_.end(), 0u);
    nodes_.reserve(2 * n_tets / leaf_size + 2);
    build_node(0, static_cast<uint32_t>(n_tets), centroids);
}

uint32_t MeshLocator::build_node(uint32_t begin, uint32_t end, const std::vector<Vec3>& centroids) {
    Aabb box = tet_boxes_[order_[begin]];
    for (uint32_t i = begin + 1; i < end; ++i) {
        const Aabb& b = tet_boxes_[order_[i]];
        for (int d = 0; d < 3; ++d) {
            box.lo[d] = std::min(box.lo[d], b.lo[d]);
            box.hi[d] = std::max(box.hi[d], b.hi[d]);
        }
    }

    const uint32_t idx = static_cast<uint32_t>(nodes_.size());
    const uint32_t count = end - begin;
    if (count <= leaf_size) {
        nodes_.push_back(Node{box, begin, 0, count});
        return idx;
    }

    // Split along the axis with the largest centroid spread at the median for guaranteed O(log N) depth.
    Vec3 clo = centroids[order_[begin]], chi = clo;
    for (uint32_t i = begin + 1; i < end; ++i) {
        const Vec3& c = centroids[order_[i]];
        for (int d = 0; d < 3; ++d) {
            clo[d] = std::min(clo[d], c[d]);
            chi[d] = std::max(chi[d], c[d]);
        }
    }
    int axis = 0;
    if (chi[1] - clo[1] > chi[axis] - clo[axis]) axis = 1;
    if (chi[2] - clo[2] > chi[axis] - clo[axis]) axis = 2;

    const uint32_t mid = begin + count / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                     [&](uint32_t a, uint32_t b) { return centroids[a][axis] < centroids[b][axis]; });

    nodes_.push_back({});   // reserve slot; fill after recursive calls so reallocation cannot invalidate refs
    const uint32_t left = build_node(begin, mid, centroids);
    const uint32_t right = build_node(mid, end, centroids);
    nodes_[idx] = Node{box, left, right, 0};
    return idx;
}

std::array<double, 4> MeshLocator::barycentric(uint32_t element, const Vec3& x) const {
    const auto& n = mesh_->elements[element].node;
    const auto& g = grad_lambda_[element];
    std::array<double, 4> lambda{};
    for (int i = 0; i < 4; ++i) lambda[i] = dot(g[i], sub(x, mesh_->nodes[n[(i + 1) % 4]]));
    return lambda;
}

std::optional<MeshLocator::Location> MeshLocator::locate(const Vec3& x) const {
    if (!finite_vec(x)) throw std::invalid_argument("MeshLocator::locate: point is not finite");
    if (nodes_.empty()) return std::nullopt;

    std::optional<Location> best;
    double best_min = -std::numeric_limits<double>::infinity();

    uint32_t stack[64];
    int sp = 0;
    if (point_in_padded_box(x, nodes_[0].box.lo, nodes_[0].box.hi)) stack[sp++] = 0;

    while (sp > 0) {
        const Node& node = nodes_[stack[--sp]];
        if (node.count > 0) {
            for (uint32_t i = node.left; i < node.left + node.count; ++i) {
                const uint32_t k = order_[i];
                if (!point_in_padded_box(x, tet_boxes_[k].lo, tet_boxes_[k].hi)) continue;
                const auto lambda = barycentric(k, x);
                const double smallest = std::min({lambda[0], lambda[1], lambda[2], lambda[3]});
                if (smallest >= -bary_tol &&
                    (smallest > best_min || (smallest == best_min && (!best || k < best->element)))) {
                    best_min = smallest;
                    best = Location{k, lambda};
                }
            }
            continue;
        }
        if (point_in_padded_box(x, nodes_[node.left].box.lo, nodes_[node.left].box.hi)) stack[sp++] = node.left;
        if (point_in_padded_box(x, nodes_[node.right].box.lo, nodes_[node.right].box.hi)) stack[sp++] = node.right;
    }
    return best;
}

std::optional<MeshLocator::Location> MeshLocator::locate_near(const Vec3& x, uint32_t hint_element) const {
    if (!finite_vec(x)) throw std::invalid_argument("MeshLocator::locate_near: point is not finite");
    if (hint_element >= mesh_->elements.size())
        throw std::invalid_argument("MeshLocator::locate_near: hint element " + std::to_string(hint_element) +
                                    " is out of range (" + std::to_string(mesh_->elements.size()) + " tets)");

    const uint32_t max_steps = std::min<uint32_t>(256u, static_cast<uint32_t>(mesh_->elements.size()));
    uint32_t cur = hint_element;
    uint32_t prev = MeshProperties::no_neighbour;
    for (uint32_t step = 0; step < max_steps; ++step) {
        const auto lambda = barycentric(cur, x);
        int worst = 0;
        for (int i = 1; i < 4; ++i)
            if (lambda[i] < lambda[worst]) worst = i;
        if (lambda[worst] >= -bary_tol) return Location{cur, lambda};

        // Step across the face opposite the most negative barycentric coordinate.
        const uint32_t next = mesh_->neighbours[cur][worst];
        if (next == MeshProperties::no_neighbour || next == prev) break;   // hit boundary/hole or grazing 2-cycle: fall back to BVH
        prev = cur;
        cur = next;
    }
    return locate(x);
}

std::vector<uint32_t> MeshLocator::query_sphere(const Vec3& center, double radius) const {
    if (!finite_vec(center)) throw std::invalid_argument("MeshLocator::query_sphere: center is not finite");
    if (!(std::isfinite(radius) && radius >= 0.0))
        throw std::invalid_argument("MeshLocator::query_sphere: radius must be finite and non-negative, got " +
                                    std::to_string(radius));
    if (nodes_.empty()) return {};

    // Small outward tolerance so points lying directly on the sphere boundary are not missed.
    const double eff_r = radius + 1e-12 * std::max(1.0, radius);
    const double eff_r2 = eff_r * eff_r;
    std::vector<uint32_t> result;

    uint32_t stack[64];
    int sp = 0;
    if (dist2_point_box(center, nodes_[0].box.lo, nodes_[0].box.hi) <= eff_r2) stack[sp++] = 0;

    while (sp > 0) {
        const Node& node = nodes_[stack[--sp]];
        if (node.count > 0) {
            for (uint32_t i = node.left; i < node.left + node.count; ++i) {
                const uint32_t k = order_[i];
                if (dist2_point_box(center, tet_boxes_[k].lo, tet_boxes_[k].hi) > eff_r2) continue;
                const auto lambda = barycentric(k, center);
                if (std::min({lambda[0], lambda[1], lambda[2], lambda[3]}) >= -bary_tol) {
                    result.push_back(k);
                    continue;
                }
                const auto& n = mesh_->elements[k].node;
                const Vec3 &p0 = mesh_->nodes[n[0]], &p1 = mesh_->nodes[n[1]], &p2 = mesh_->nodes[n[2]], &p3 = mesh_->nodes[n[3]];
                const double d2 = std::min({dist2_point_triangle(center, p0, p1, p2),
                                            dist2_point_triangle(center, p0, p1, p3),
                                            dist2_point_triangle(center, p0, p2, p3),
                                            dist2_point_triangle(center, p1, p2, p3)});
                if (d2 <= eff_r2) result.push_back(k);
            }
            continue;
        }
        if (dist2_point_box(center, nodes_[node.left].box.lo, nodes_[node.left].box.hi) <= eff_r2) stack[sp++] = node.left;
        if (dist2_point_box(center, nodes_[node.right].box.lo, nodes_[node.right].box.hi) <= eff_r2) stack[sp++] = node.right;
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<uint32_t> MeshLocator::query_box(const Vec3& lo, const Vec3& hi) const {
    if (!finite_vec(lo) || !finite_vec(hi))
        throw std::invalid_argument("MeshLocator::query_box: box bounds must be finite");
    for (int d = 0; d < 3; ++d)
        if (lo[d] > hi[d])
            throw std::invalid_argument("MeshLocator::query_box: lo must not exceed hi");
    if (nodes_.empty()) return {};

    std::vector<uint32_t> result;
    uint32_t stack[64];
    int sp = 0;
    if (boxes_overlap(nodes_[0].box.lo, nodes_[0].box.hi, lo, hi)) stack[sp++] = 0;

    while (sp > 0) {
        const Node& node = nodes_[stack[--sp]];
        if (node.count > 0) {
            for (uint32_t i = node.left; i < node.left + node.count; ++i) {
                const uint32_t k = order_[i];
                if (boxes_overlap(tet_boxes_[k].lo, tet_boxes_[k].hi, lo, hi)) result.push_back(k);
            }
            continue;
        }
        if (boxes_overlap(nodes_[node.left].box.lo, nodes_[node.left].box.hi, lo, hi)) stack[sp++] = node.left;
        if (boxes_overlap(nodes_[node.right].box.lo, nodes_[node.right].box.hi, lo, hi)) stack[sp++] = node.right;
    }

    std::sort(result.begin(), result.end());
    return result;
}
