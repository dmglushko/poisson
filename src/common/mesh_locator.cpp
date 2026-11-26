#include "common/mesh_locator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

using Vec3 = std::array<double, 3>;

constexpr uint32_t leaf_size = 8;
constexpr double   bary_tol = 1e-9;
constexpr uint32_t steepest_descent_steps = 16;   // walk_to_point: steepest descent for steps 0..15, then remembering walk
constexpr uint8_t  no_face = 4;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
bool finite_vec(const Vec3& v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

double dist2(const Vec3& a, const Vec3& b) {
    const Vec3 d = sub(a, b);
    return dot(d, d);
}

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

// Squared distance from `p` to the farthest corner of the box [lo, hi].
double max_dist2_point_box(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    double d2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double diff = std::max(std::abs(p[d] - lo[d]), std::abs(hi[d] - p[d]));
        d2 += diff * diff;
    }
    return d2;
}

bool point_in_padded_box(const Vec3& p, const Vec3& lo, const Vec3& hi) {
    for (int d = 0; d < 3; ++d) {
        const double span = hi[d] - lo[d];
        const double pad = 4.0 * bary_tol * std::max(span, std::max(std::abs(lo[d]), std::abs(hi[d])));
        if (p[d] < lo[d] - pad || p[d] > hi[d] + pad) return false;
    }
    return true;
}

bool boxes_overlap(const Vec3& a_lo, const Vec3& a_hi, const Vec3& b_lo, const Vec3& b_hi) {
    for (int d = 0; d < 3; ++d)
        if (a_hi[d] < b_lo[d] || b_hi[d] < a_lo[d]) return false;
    return true;
}

// Whether the box [in_lo, in_hi] lies inside the box [lo, hi].
bool box_inside(const Vec3& in_lo, const Vec3& in_hi, const Vec3& lo, const Vec3& hi) {
    for (int d = 0; d < 3; ++d)
        if (in_lo[d] < lo[d] || in_hi[d] > hi[d]) return false;
    return true;
}

struct Corners {
    std::array<const Vec3*, 4> p;
};

Corners corners(const MeshProperties& mesh, const std::array<uint32_t, 4>& ids) {
    return {{&mesh.nodes[ids[0]], &mesh.nodes[ids[1]], &mesh.nodes[ids[2]], &mesh.nodes[ids[3]]}};
}

void tet_box(const Corners& c, Vec3& lo, Vec3& hi) {
    lo = *c.p[0];
    hi = lo;
    for (int i = 1; i < 4; ++i)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], (*c.p[i])[d]);
            hi[d] = std::max(hi[d], (*c.p[i])[d]);
        }
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

// The four faces of a tet as planes: V_i(x) = dot(normal[i], x - anchor[i]) is 6x the
// signed volume of the tet with local node i replaced by x, so V_0..V_3 are all positive
// inside a positively oriented tet and V_i / volume6 is the barycentric coordinate of
// node i (where volume6 = V_0(p_0) is 6x the tet's signed volume, independent of x). Each
// face's vertices are taken in ascending node-index order (anchor = smallest) and the
// permutation sign is folded into `normal`, so two tets sharing a face evaluate
// bit-identical values of opposite sign for it.
struct TetFaces {
    std::array<Vec3, 4> normal;
    std::array<Vec3, 4> anchor;
    double              volume6 = 0.0;
};

TetFaces tet_faces(const std::vector<Vec3>& nodes, const std::array<uint32_t, 4>& ids) {
    // Load all four vertices before any data-dependent work so their cache misses overlap.
    const std::array<Vec3, 4> p{nodes[ids[0]], nodes[ids[1]], nodes[ids[2]], nodes[ids[3]]};

    // Sort the local slots by node index (branchless 5-comparator network), tracking parity.
    std::array<uint32_t, 4> key = ids;
    std::array<int, 4> slot{0, 1, 2, 3};
    bool odd = false;
    const auto exchange = [&](int a, int b) {
        const bool swap = key[a] > key[b];
        odd = odd != swap;
        const uint32_t ka = key[a], kb = key[b];
        const int sa = slot[a], sb = slot[b];
        key[a] = swap ? kb : ka;
        key[b] = swap ? ka : kb;
        slot[a] = swap ? sb : sa;
        slot[b] = swap ? sa : sb;
    };
    exchange(0, 1);
    exchange(2, 3);
    exchange(0, 2);
    exchange(1, 3);
    exchange(1, 2);

    TetFaces f;
    for (int r = 0; r < 4; ++r) {
        // The face opposite the vertex of rank r: the other three in ascending node order.
        const int a = slot[r == 0 ? 1 : 0], b = slot[r <= 1 ? 2 : 1], c = slot[r <= 2 ? 3 : 2];
        Vec3 n = cross(sub(p[b], p[a]), sub(p[c], p[a]));
        // Sorting the 4-tuple (x in slot i) has parity `odd`; moving x from rank r to the end
        // takes 3 - r more transpositions.
        if (odd != ((3 - r) % 2 == 1)) n = {-n[0], -n[1], -n[2]};
        f.normal[slot[r]] = n;
        f.anchor[slot[r]] = p[a];
    }
    f.volume6 = dot(f.normal[0], sub(p[0], f.anchor[0]));
    return f;
}

std::array<double, 4> face_values(const TetFaces& f, const Vec3& x) {
    std::array<double, 4> v{};
    for (int i = 0; i < 4; ++i) v[i] = dot(f.normal[i], sub(x, f.anchor[i]));
    return v;
}

std::array<double, 4> normalized(const std::array<double, 4>& v, double total) {
    return {v[0] / total, v[1] / total, v[2] / total, v[3] / total};
}

// Barycentric coordinates of `x`, or nothing if the tet's volume is non-positive or any
// resulting coordinate is non-finite.
std::optional<std::array<double, 4>> barycentric_at(const TetFaces& f, const Vec3& x) {
    if (!(std::isfinite(f.volume6) && f.volume6 > 0.0)) return std::nullopt;
    const auto l = normalized(face_values(f, x), f.volume6);
    if (!(std::isfinite(l[0]) && std::isfinite(l[1]) && std::isfinite(l[2]) && std::isfinite(l[3])))
        return std::nullopt;
    return l;
}

uint64_t splitmix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

std::string tet_name(uint32_t k) { return "tet " + std::to_string(k); }

// Validation of one visited tet; only the free walk functions check (MeshLocator validated
// the whole mesh when it was built).
template <bool Checked>
void check_tet(const MeshProperties& mesh, uint32_t k, const char* who) {
    if constexpr (Checked) {
        const std::size_t n_nodes = mesh.nodes.size(), n_tets = mesh.elements.size();
        if (k >= n_tets) throw std::invalid_argument(std::string(who) + ": " + tet_name(k) + " is out of range");
        for (const uint32_t node : mesh.elements[k].node) {
            if (node >= n_nodes)
                throw std::invalid_argument(std::string(who) + ": " + tet_name(k) + " has node " +
                                            std::to_string(node) + ", but the mesh has " + std::to_string(n_nodes) +
                                            " nodes");
            if (!finite_vec(mesh.nodes[node]))
                throw std::invalid_argument(std::string(who) + ": node " + std::to_string(node) +
                                            " has non-finite coordinates");
        }
        for (const uint32_t nb : mesh.neighbours[k])
            if (nb != MeshProperties::no_neighbour && nb >= n_tets)
                throw std::invalid_argument(std::string(who) + ": " + tet_name(k) + " names neighbour " +
                                            std::to_string(nb) + " out of range");
    } else {
        (void)mesh;
        (void)k;
        (void)who;
    }
}

void check_volume(double total, uint32_t k, const char* who) {
    if (!(std::isfinite(total) && total > 0.0))
        throw std::invalid_argument(std::string(who) + ": " + tet_name(k) + " has volume " +
                                    std::to_string(total / 6.0) + "; a positive volume is needed");
}

void check_walk_args(const MeshProperties& mesh, uint32_t start_element, const char* who) {
    if (start_element >= mesh.elements.size())
        throw std::invalid_argument(std::string(who) + ": start element " + std::to_string(start_element) +
                                    " is out of range (" + std::to_string(mesh.elements.size()) + " tets)");
}

void check_mesh_shape(const MeshProperties& mesh, uint32_t max_steps, const char* who) {
    if (mesh.neighbours.size() != mesh.elements.size())
        throw std::invalid_argument(std::string(who) + ": neighbours has " + std::to_string(mesh.neighbours.size()) +
                                    " entries for " + std::to_string(mesh.elements.size()) + " tets");
    if (max_steps < 1) throw std::invalid_argument(std::string(who) + ": max_steps must be at least 1");
}

// The local face of `next` that `cur` is on the other side of, or no_face.
uint8_t entry_face(const MeshProperties& mesh, uint32_t next, uint32_t cur) {
    const auto& nb = mesh.neighbours[next];
    for (uint8_t i = 0; i < 4; ++i)
        if (nb[i] == cur) return i;
    return no_face;
}

template <bool Checked>
PointWalkResult walk_to_point_impl(const MeshProperties& mesh, const Vec3& x, uint32_t start, uint32_t max_steps,
                                   const char* who) {
    PointWalkResult result;
    uint32_t cur = start;
    uint8_t entry = no_face;
    for (uint32_t step = 0;; ++step) {
        check_tet<Checked>(mesh, cur, who);
        const std::array<uint32_t, 4> nb = mesh.neighbours[cur];   // load early: overlaps the vertex loads
        const TetFaces faces = tet_faces(mesh.nodes, mesh.elements[cur].node);
        const double total = faces.volume6;
        check_volume(total, cur, who);
        const auto v = face_values(faces, x);

        // Classify the faces x lies beyond: interior ones can be crossed, boundary ones cannot.
        // Using `v[i] >= threshold` ensures NaN from extreme coordinates counts as beyond.
        const double threshold = -bary_tol * total;
        int worst_interior = -1, worst_boundary = -1;
        unsigned interior = 0;
        for (int i = 0; i < 4; ++i) {
            if (v[i] >= threshold) continue;
            if (nb[i] == MeshProperties::no_neighbour) {
                if (worst_boundary < 0 || v[i] < v[worst_boundary]) worst_boundary = i;
            } else {
                interior |= 1u << i;
                if (worst_interior < 0 || v[i] < v[worst_interior]) worst_interior = i;
            }
        }

        if (worst_interior < 0 && worst_boundary < 0) {
            result.status = PointWalkResult::Status::found;
            result.element = cur;
            result.steps = step;
            result.barycentric = normalized(v, total);
            return result;
        }
        if (worst_interior < 0) {
            result.status = PointWalkResult::Status::exited;
            result.exit_face = static_cast<uint8_t>(worst_boundary);
            result.element = cur;
            result.steps = step;
            result.barycentric = normalized(v, total);
            return result;
        }
        if (step == max_steps) {
            result.status = PointWalkResult::Status::step_limit;
            result.element = cur;
            result.steps = step;
            result.barycentric = normalized(v, total);
            return result;
        }

        int face = worst_interior;
        if (step >= steepest_descent_steps) {
            // Remembering walk: never step straight back, and start the face scan at a
            // pseudo-random face so the walk cannot cycle forever on degenerate input.
            unsigned candidates = interior;
            if (entry != no_face && (candidates & ~(1u << entry)) != 0) candidates &= ~(1u << entry);
            const unsigned first = static_cast<unsigned>(splitmix64((uint64_t{cur} << 32) | step) & 3u);
            for (unsigned k = 0; k < 4; ++k) {
                const unsigned i = (first + k) & 3u;
                if ((candidates >> i) & 1u) {
                    face = static_cast<int>(i);
                    break;
                }
            }
        }
        const uint32_t next = nb[face];
        entry = entry_face(mesh, next, cur);
        cur = next;
    }
}

template <bool Checked>
SegmentWalkResult walk_along_segment_impl(const MeshProperties& mesh, const Vec3& x0, const Vec3& x1, uint32_t start,
                                          uint32_t max_steps, const char* who) {
    SegmentWalkResult result;
    const Vec3 dx = sub(x1, x0);

    uint32_t cur = start;
    double t_cur = 0.0;
    for (uint32_t step = 0;; ++step) {
        check_tet<Checked>(mesh, cur, who);
        const std::array<uint32_t, 4> nb = mesh.neighbours[cur];   // load early: overlaps the vertex loads
        const TetFaces faces = tet_faces(mesh.nodes, mesh.elements[cur].node);
        const double total = faces.volume6;
        check_volume(total, cur, who);
        const auto v0 = face_values(faces, x0);
        const auto v1 = face_values(faces, x1);

        // No entry-face skip: on a consistent mesh the shared face has v1 > 0 in the new tet,
        // so it is never a candidate; checking all four faces also prevents a malformed mesh
        // with disjoint neighbours from falsely reporting `found`.
        const double threshold = -bary_tol * total;
        int face = -1;
        double best_t = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (v1[i] >= threshold) continue;
            const double ti = v0[i] <= 0.0 ? 0.0 : v0[i] / (v0[i] - v1[i]);
            if (face < 0 || ti < best_t || (ti == best_t && v1[i] < v1[face])) {
                face = i;
                best_t = ti;
            }
        }

        if (face < 0) {
            result.status = SegmentWalkResult::Status::found;
            result.element = cur;
            result.steps = step;
            result.t = 1.0;
            result.barycentric = normalized(v1, total);
            return result;
        }

        const double t_face = best_t <= t_cur ? t_cur : (best_t >= 1.0 ? 1.0 : best_t);
        uint32_t next = nb[face];
        if (next == MeshProperties::no_neighbour) {
            // Before exiting through a boundary face, check whether p(t_face) grazes an
            // interior face (within bary_tol) that x1 lies strictly beyond. Crossing into
            // that neighbour avoids false boundary exits at reflex edges (including rotated
            // meshes with ULP near-ties) and handles drifted/stationary x0 interior-first.
            const Vec3 p{x0[0] + t_face * dx[0], x0[1] + t_face * dx[1], x0[2] + t_face * dx[2]};
            const auto vp = face_values(faces, p);
            const double graze_tol = bary_tol * total;
            int redirect = -1;
            for (int g = 0; g < 4; ++g) {
                if (nb[g] == MeshProperties::no_neighbour) continue;
                if (v1[g] < 0.0 && vp[g] <= graze_tol) {
                    if (redirect < 0 || v1[g] < v1[redirect]) redirect = g;
                }
            }
            if (redirect < 0) {
                result.status = SegmentWalkResult::Status::exited;
                result.exit_face = static_cast<uint8_t>(face);
                result.element = cur;
                result.steps = step;
                result.t = t_face;
                result.barycentric = normalized(vp, total);
                return result;
            }
            next = nb[redirect];
        }
        if (step == max_steps) {
            const Vec3 p{x0[0] + t_cur * dx[0], x0[1] + t_cur * dx[1], x0[2] + t_cur * dx[2]};
            result.status = SegmentWalkResult::Status::step_limit;
            result.element = cur;
            result.steps = step;
            result.t = t_cur;
            result.barycentric = normalized(face_values(faces, p), total);
            return result;
        }

        t_cur = t_face;
        cur = next;
    }
}

} // namespace

PointWalkResult walk_to_point(const MeshProperties& mesh, const Vec3& x, uint32_t start_element, uint32_t max_steps) {
    constexpr const char* who = "walk_to_point";
    check_mesh_shape(mesh, max_steps, who);
    if (!finite_vec(x)) throw std::invalid_argument("walk_to_point: point is not finite");
    check_walk_args(mesh, start_element, who);
    return walk_to_point_impl<true>(mesh, x, start_element, max_steps, who);
}

SegmentWalkResult walk_along_segment(const MeshProperties& mesh, const Vec3& x0, const Vec3& x1,
                                     uint32_t start_element, uint32_t max_steps) {
    constexpr const char* who = "walk_along_segment";
    check_mesh_shape(mesh, max_steps, who);
    if (!finite_vec(x0) || !finite_vec(x1))
        throw std::invalid_argument("walk_along_segment: segment end point is not finite");
    check_walk_args(mesh, start_element, who);
    return walk_along_segment_impl<true>(mesh, x0, x1, start_element, max_steps, who);
}

MeshLocator::MeshLocator(const MeshProperties& mesh) : MeshLocator(mesh, Settings{}) {}

MeshLocator::MeshLocator(const MeshProperties& mesh, Settings settings) : mesh_(&mesh), settings_(settings) {
    if (settings_.max_walk_steps < 1) throw std::invalid_argument("MeshLocator: max_walk_steps must be at least 1");

    const std::size_t n_nodes = mesh.nodes.size();
    const std::size_t n_tets = mesh.elements.size();
    if (n_tets >= std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("MeshLocator: too many tets (" + std::to_string(n_tets) + ")");

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

    // Temporary float split keys (tet box midpoints shifted by node 0 of tet 0 so small
    // meshes far from the origin keep full float precision); exact double boxes are rebuilt
    // bottom-up from mesh.nodes after partitioning, so no per-tet box is kept.
    std::vector<std::array<float, 3>> split_keys(n_tets);
    constexpr double float_max = std::numeric_limits<float>::max();
    Vec3 split_origin{0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < n_tets; ++k) {
        const auto& n = mesh.elements[k].node;
        for (uint32_t node : n)
            if (node >= n_nodes)
                throw std::invalid_argument("MeshLocator: tet " + std::to_string(k) + " has node " +
                                            std::to_string(node) + ", but the mesh has " + std::to_string(n_nodes) +
                                            " nodes");
        if (k == 0) split_origin = mesh.nodes[n[0]];

        const TetFaces faces = tet_faces(mesh.nodes, n);
        if (!(std::isfinite(faces.volume6) && faces.volume6 > 0.0))
            throw std::invalid_argument("MeshLocator: tet " + std::to_string(k) + " has volume " +
                                        std::to_string(faces.volume6 / 6.0) + "; a positive volume is needed");

        Vec3 lo, hi;
        tet_box(corners(mesh, n), lo, hi);
        for (int d = 0; d < 3; ++d) {
            const double mid = 0.5 * (lo[d] - split_origin[d]) + 0.5 * (hi[d] - split_origin[d]);
            split_keys[k][d] = static_cast<float>(std::clamp(mid, -float_max, float_max));
        }
    }

    order_.resize(n_tets);
    std::iota(order_.begin(), order_.end(), 0u);
    // Median splits stop at <= leaf_size, so every leaf holds > leaf_size / 2 tets:
    // at most n_tets / 4 leaves and n_tets / 2 - 1 nodes.
    nodes_.reserve(n_tets / 2 + 1);
    build(0, static_cast<uint32_t>(n_tets), split_keys);

    // Exact double boxes, bottom-up: children always have larger indices than their parent.
    for (std::size_t idx = nodes_.size(); idx-- > 0;) {
        Node& node = nodes_[idx];
        if (node.right == 0) {
            tet_box(corners(mesh, mesh.elements[order_[node.first]].node), node.box.lo, node.box.hi);
            for (uint32_t i = node.first + 1; i < node.first + node.count; ++i) {
                Vec3 lo, hi;
                tet_box(corners(mesh, mesh.elements[order_[i]].node), lo, hi);
                for (int d = 0; d < 3; ++d) {
                    node.box.lo[d] = std::min(node.box.lo[d], lo[d]);
                    node.box.hi[d] = std::max(node.box.hi[d], hi[d]);
                }
            }
        } else {
            const Aabb& l = nodes_[idx + 1].box;
            const Aabb& r = nodes_[node.right].box;
            for (int d = 0; d < 3; ++d) {
                node.box.lo[d] = std::min(l.lo[d], r.lo[d]);
                node.box.hi[d] = std::max(l.hi[d], r.hi[d]);
            }
        }
    }
}

void MeshLocator::build(uint32_t begin, uint32_t end, const std::vector<std::array<float, 3>>& split_keys) {
    const uint32_t idx = static_cast<uint32_t>(nodes_.size());
    const uint32_t count = end - begin;
    nodes_.push_back(Node{{}, begin, count, 0});
    if (count <= leaf_size) return;

    // Split along the axis with the largest key spread at the median for guaranteed O(log N) depth.
    std::array<float, 3> klo = split_keys[order_[begin]], khi = klo;
    for (uint32_t i = begin + 1; i < end; ++i) {
        const auto& k = split_keys[order_[i]];
        for (int d = 0; d < 3; ++d) {
            klo[d] = std::min(klo[d], k[d]);
            khi[d] = std::max(khi[d], k[d]);
        }
    }
    int axis = 0;
    double best_spread = double{khi[0]} - double{klo[0]};
    for (int d = 1; d < 3; ++d) {
        const double spread = double{khi[d]} - double{klo[d]};
        if (spread > best_spread) {
            best_spread = spread;
            axis = d;
        }
    }

    const uint32_t mid = begin + count / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end, [&](uint32_t a, uint32_t b) {
        const float ka = split_keys[a][axis], kb = split_keys[b][axis];
        return ka < kb || (ka == kb && a < b);
    });

    build(begin, mid, split_keys);
    const uint32_t right = static_cast<uint32_t>(nodes_.size());
    assert(nodes_[idx + 1].first == begin && nodes_[idx + 1].count == mid - begin);
    build(mid, end, split_keys);
    nodes_[idx].right = right;
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
        const uint32_t idx = stack[--sp];
        const Node& node = nodes_[idx];
        if (node.right == 0) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t k = order_[i];
                const auto& ids = mesh_->elements[k].node;
                Vec3 lo, hi;
                tet_box(corners(*mesh_, ids), lo, hi);
                if (!point_in_padded_box(x, lo, hi)) continue;
                const auto lambda = barycentric_at(tet_faces(mesh_->nodes, ids), x);
                if (!lambda) continue;
                const double smallest = std::min({(*lambda)[0], (*lambda)[1], (*lambda)[2], (*lambda)[3]});
                if (smallest >= -bary_tol &&
                    (smallest > best_min || (smallest == best_min && (!best || k < best->element)))) {
                    best_min = smallest;
                    best = Location{k, *lambda};
                }
            }
            continue;
        }
        if (point_in_padded_box(x, nodes_[idx + 1].box.lo, nodes_[idx + 1].box.hi)) stack[sp++] = idx + 1;
        if (point_in_padded_box(x, nodes_[node.right].box.lo, nodes_[node.right].box.hi)) stack[sp++] = node.right;
    }
    return best;
}

PointWalkResult MeshLocator::walk_to_point(const Vec3& x, uint32_t start_element) const {
    constexpr const char* who = "MeshLocator::walk_to_point";
    if (!finite_vec(x)) throw std::invalid_argument("MeshLocator::walk_to_point: point is not finite");
    check_walk_args(*mesh_, start_element, who);
    return walk_to_point_impl<false>(*mesh_, x, start_element, settings_.max_walk_steps, who);
}

SegmentWalkResult MeshLocator::walk_along_segment(const Vec3& x0, const Vec3& x1, uint32_t start_element) const {
    constexpr const char* who = "MeshLocator::walk_along_segment";
    if (!finite_vec(x0) || !finite_vec(x1))
        throw std::invalid_argument("MeshLocator::walk_along_segment: segment end point is not finite");
    check_walk_args(*mesh_, start_element, who);
    return walk_along_segment_impl<false>(*mesh_, x0, x1, start_element, settings_.max_walk_steps, who);
}

std::optional<MeshLocator::Location> MeshLocator::locate_near(const Vec3& x, uint32_t hint_element) const {
    if (!finite_vec(x)) throw std::invalid_argument("MeshLocator::locate_near: point is not finite");
    if (hint_element >= mesh_->elements.size())
        throw std::invalid_argument("MeshLocator::locate_near: hint element " + std::to_string(hint_element) +
                                    " is out of range (" + std::to_string(mesh_->elements.size()) + " tets)");
    if (!point_in_padded_box(x, nodes_[0].box.lo, nodes_[0].box.hi)) return std::nullopt;

    const PointWalkResult walk =
        walk_to_point_impl<false>(*mesh_, x, hint_element, settings_.max_walk_steps, "MeshLocator::locate_near");
    if (walk.status == PointWalkResult::Status::found) return Location{walk.element, walk.barycentric};
    return locate(x);
}

std::vector<uint32_t> MeshLocator::query_sphere(const Vec3& center, double radius) const {
    std::vector<uint32_t> result;
    query_sphere(center, radius, result);
    std::sort(result.begin(), result.end());
    return result;
}

void MeshLocator::query_sphere(const Vec3& center, double radius, std::vector<uint32_t>& out) const {
    if (!finite_vec(center)) throw std::invalid_argument("MeshLocator::query_sphere: center is not finite");
    if (!(std::isfinite(radius) && radius >= 0.0))
        throw std::invalid_argument("MeshLocator::query_sphere: radius must be finite and non-negative, got " +
                                    std::to_string(radius));
    out.clear();
    if (nodes_.empty()) {
        if (!std::isfinite(radius * radius))
            throw std::invalid_argument("MeshLocator::query_sphere: radius is too large for squared-distance queries");
        return;
    }

    // Small outward tolerance relative to the query radius and mesh coordinate scale so
    // points lying directly on the sphere boundary are not missed.
    const Aabb& root = nodes_[0].box;
    double root_scale = 0.0;
    for (int d = 0; d < 3; ++d)
        root_scale = std::max(root_scale, std::max(root.hi[d] - root.lo[d],
                                                   std::max(std::abs(root.lo[d]), std::abs(root.hi[d]))));
    const double eff_r = radius + 1e-12 * std::max(radius, root_scale);
    const double eff_r2 = eff_r * eff_r;
    if (!std::isfinite(eff_r2))
        throw std::invalid_argument("MeshLocator::query_sphere: radius is too large for squared-distance queries");

    uint32_t stack[64];
    int sp = 0;
    if (dist2_point_box(center, nodes_[0].box.lo, nodes_[0].box.hi) <= eff_r2) stack[sp++] = 0;

    while (sp > 0) {
        const uint32_t idx = stack[--sp];
        const Node& node = nodes_[idx];
        // Whole subtree inside the ball: every tet in it intersects.
        if (max_dist2_point_box(center, node.box.lo, node.box.hi) <= eff_r2) {
            out.insert(out.end(), order_.begin() + node.first, order_.begin() + node.first + node.count);
            continue;
        }
        if (node.right == 0) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t k = order_[i];
                const auto& ids = mesh_->elements[k].node;
                const Corners c = corners(*mesh_, ids);
                bool hit = dist2(*c.p[0], center) <= eff_r2 || dist2(*c.p[1], center) <= eff_r2 ||
                           dist2(*c.p[2], center) <= eff_r2 || dist2(*c.p[3], center) <= eff_r2;
                if (!hit) {
                    Vec3 lo, hi;
                    tet_box(c, lo, hi);
                    if (dist2_point_box(center, lo, hi) > eff_r2) continue;
                    const auto v = face_values(tet_faces(mesh_->nodes, ids), center);
                    hit = v[0] >= 0.0 && v[1] >= 0.0 && v[2] >= 0.0 && v[3] >= 0.0;
                }
                if (!hit) {
                    const Vec3 &p0 = *c.p[0], &p1 = *c.p[1], &p2 = *c.p[2], &p3 = *c.p[3];
                    const double d2 = std::min({dist2_point_triangle(center, p0, p1, p2),
                                                dist2_point_triangle(center, p0, p1, p3),
                                                dist2_point_triangle(center, p0, p2, p3),
                                                dist2_point_triangle(center, p1, p2, p3)});
                    hit = d2 <= eff_r2;
                }
                if (hit) out.push_back(k);
            }
            continue;
        }
        if (dist2_point_box(center, nodes_[idx + 1].box.lo, nodes_[idx + 1].box.hi) <= eff_r2) stack[sp++] = idx + 1;
        if (dist2_point_box(center, nodes_[node.right].box.lo, nodes_[node.right].box.hi) <= eff_r2)
            stack[sp++] = node.right;
    }
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
        const uint32_t idx = stack[--sp];
        const Node& node = nodes_[idx];
        // Whole subtree inside the query box: every tet box in it overlaps.
        if (box_inside(node.box.lo, node.box.hi, lo, hi)) {
            result.insert(result.end(), order_.begin() + node.first, order_.begin() + node.first + node.count);
            continue;
        }
        if (node.right == 0) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t k = order_[i];
                Vec3 tlo, thi;
                tet_box(corners(*mesh_, mesh_->elements[k].node), tlo, thi);
                if (boxes_overlap(tlo, thi, lo, hi)) result.push_back(k);
            }
            continue;
        }
        if (boxes_overlap(nodes_[idx + 1].box.lo, nodes_[idx + 1].box.hi, lo, hi)) stack[sp++] = idx + 1;
        if (boxes_overlap(nodes_[node.right].box.lo, nodes_[node.right].box.hi, lo, hi)) stack[sp++] = node.right;
    }

    std::sort(result.begin(), result.end());
    return result;
}
