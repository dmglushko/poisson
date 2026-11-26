#pragma once

#include "mesh_properties.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// Spatial lookup over a tetrahedral MeshProperties: an AABB BVH for cold point and range
// queries, plus BVH-free walks through `mesh.neighbours` for warm point location and
// segment tracing.
//
// Shared across modules that need fast 3D point or range queries:
//   - AMR (locating unhinted point sources),
//   - AnalyticalMaker / P3M (locating trapped charges and cutoff-sphere neighborhoods),
//   - Ray Tracing (initial electron tet lookup, per-step segment walks, and non-convex
//     cavity fallback).
//
// Barycentric coordinates are computed from four signed face volumes evaluated with each
// face's vertices in ascending node-index order, so two tets sharing a face see exactly
// opposite values for that face: a point is never "outside" both of them, and walks
// cannot flip-flop across a shared face because of rounding.

// Result of walking from `start_element` toward a point `x` (see walk_to_point).
struct PointWalkResult {
    enum class Status : uint8_t {
        found,        // `element` contains `x` (within a small barycentric tolerance)
        exited,       // `x` lies beyond a boundary face of `element` (outside the mesh, or across a cavity)
        step_limit,   // the step budget ran out before reaching `x`
    };

    Status                status = Status::step_limit;
    uint8_t               exit_face = 0;    // exited: local face 0..3 (opposite that node) with no_neighbour; the most negative one
    uint32_t              element = 0;      // found: containing tet; exited / step_limit: last visited tet
    uint32_t              steps = 0;        // hops taken; 0 if `start_element` already contains `x` or exits at once
    std::array<double, 4> barycentric{};    // barycentric coordinates of `x` in `element` (sum to 1 up to rounding)
};

// Result of walking along the segment p(t) = x0 + t * (x1 - x0), t in [0, 1], starting
// from the tet `start_element` that contains `x0` (see walk_along_segment).
struct SegmentWalkResult {
    enum class Status : uint8_t {
        found,        // `element` contains `x1`
        exited,       // the segment leaves the mesh through boundary face `exit_face` of `element` at p(t)
        step_limit,   // the step budget ran out before reaching `x1`
    };

    Status                status = Status::step_limit;
    uint8_t               exit_face = 0;    // exited: local face 0..3 of `element` pierced on the boundary
    uint32_t              element = 0;      // found: containing tet; exited / step_limit: last visited tet
    uint32_t              steps = 0;        // hops taken
    double                t = 0.0;          // found: 1; exited: exit crossing fraction; step_limit: last crossing fraction reached
    std::array<double, 4> barycentric{};    // in `element`: at x1 when found, else at p(t)
};

// Spatial index over a tetrahedral mesh. Holds a non-owning reference to `MeshProperties`:
// the mesh must outlive the locator and remain unchanged while it is queried.
class MeshLocator {
public:
    struct Location {
        uint32_t              element = 0;      // index into MeshProperties::elements
        std::array<double, 4> barycentric{};    // barycentric coordinates in `element`
    };

    static constexpr uint32_t default_max_walk_steps = 64;

    struct Settings {
        uint32_t max_walk_steps = default_max_walk_steps;   // hop budget of walk_to_point / walk_along_segment / locate_near; >= 1
    };

    // Builds an O(N log N) AABB BVH over `mesh.elements`. Throws std::invalid_argument if
    // any node coordinate is non-finite, any tet/neighbour index is out of range, any tet
    // has a non-positive volume, or `settings.max_walk_steps` is 0. An empty mesh is allowed.
    explicit MeshLocator(const MeshProperties& mesh);
    MeshLocator(const MeshProperties& mesh, Settings settings);

    // Cold O(log N) point location via the BVH. Returns the tet containing `x` (within a
    // small barycentric tolerance) and its barycentric coordinates, or empty if `x` lies
    // outside the mesh. Ties on shared faces/edges break deterministically (largest minimum
    // barycentric coordinate, then smallest tet index). Throws std::invalid_argument if `x`
    // is not finite.
    std::optional<Location> locate(const std::array<double, 3>& x) const;

    // Walks through `mesh.neighbours` from `start_element` toward `x`, at most
    // `settings.max_walk_steps` hops; never uses the BVH. See the free walk_to_point.
    // Throws std::invalid_argument if `x` is not finite or `start_element` is out of range.
    PointWalkResult walk_to_point(const std::array<double, 3>& x, uint32_t start_element) const;

    // Walks along the segment x0 -> x1 from `start_element`, at most
    // `settings.max_walk_steps` hops; never uses the BVH. `x0` should lie inside
    // `start_element` (within tolerance). See the free walk_along_segment. Throws
    // std::invalid_argument if `x0` or `x1` is not finite or `start_element` is out of range.
    SegmentWalkResult walk_along_segment(const std::array<double, 3>& x0, const std::array<double, 3>& x1,
                                         uint32_t start_element) const;

    // Warm point location: walk_to_point from `hint_element`, falling back to locate(x) if
    // the walk does not end `found` (it exited through a boundary face, e.g. across a
    // non-convex cavity/hole, or hit the step limit). Throws std::invalid_argument if `x`
    // is not finite or `hint_element` is out of range.
    std::optional<Location> locate_near(const std::array<double, 3>& x, uint32_t hint_element) const;

    // All tets whose closed tetrahedron intersects the closed ball |y - center| <= radius,
    // returned sorted and unique. Squared distances within the mesh must be representable in
    // `double`. Throws std::invalid_argument if `center` is not finite, or `radius` is
    // negative, not finite, or too large for its padded square to be finite.
    std::vector<uint32_t> query_sphere(const std::array<double, 3>& center, double radius) const;

    // Same set as above, but clears `out` and fills it with the unique tet indices in
    // unspecified (unsorted) order, reusing `out`'s capacity across calls.
    void query_sphere(const std::array<double, 3>& center, double radius, std::vector<uint32_t>& out) const;

    // All tets whose axis-aligned bounding box (not the tet itself) intersects the closed box
    // [lo, hi]: a conservative superset of the tets that exactly intersect the box. Returned
    // sorted and unique. Throws std::invalid_argument if `lo` or `hi` is not finite or
    // `lo[d] > hi[d]` for any axis d.
    std::vector<uint32_t> query_box(const std::array<double, 3>& lo, const std::array<double, 3>& hi) const;

private:
    struct Aabb {
        std::array<double, 3> lo{};
        std::array<double, 3> hi{};
    };

    // Depth-first (pre-order) layout: the left child of node `idx` is `idx + 1`. Every node
    // covers the contiguous range order_[first, first + count).
    struct alignas(64) Node {
        Aabb     box;
        uint32_t first = 0;   // first index into order_
        uint32_t count = 0;   // number of tets in the subtree
        uint32_t right = 0;   // right child index; 0 = leaf
    };
    static_assert(sizeof(Node) == 64);

    void build(uint32_t begin, uint32_t end, const std::vector<std::array<float, 3>>& split_keys);

    const MeshProperties* mesh_ = nullptr;
    Settings              settings_;
    std::vector<uint32_t> order_;   // tet indices, grouped by BVH leaf
    std::vector<Node>     nodes_;
};

// Walks from `start_element` toward `x` across faces in `mesh.neighbours`, taking at most
// `max_steps` hops and never using a BVH. Each step crosses a face whose barycentric
// coordinate is negative and that has a neighbour (steepest descent for the first 16 steps,
// then a deterministic randomized "remembering" walk that cannot cycle forever). Returns
// `exited` only when every face that `x` lies beyond is a boundary face (`no_neighbour`),
// so an exit means `x` is outside the mesh or across a cavity/hole of a non-convex mesh.
// Throws std::invalid_argument on a malformed mesh (neighbours size, out-of-range node or
// neighbour index, non-finite coordinate, or non-positive volume of a visited tet), a
// non-finite `x`, an out-of-range `start_element`, or `max_steps == 0`.
PointWalkResult walk_to_point(const MeshProperties& mesh, const std::array<double, 3>& x, uint32_t start_element,
                              uint32_t max_steps = MeshLocator::default_max_walk_steps);

// Walks along the segment p(t) = x0 + t * (x1 - x0) from `start_element` (which should
// contain `x0` within tolerance), always crossing the face the segment pierces first, and
// taking at most `max_steps` hops without a BVH. Returns `found` with t = 1 when a tet
// contains `x1`, or `exited` at the first boundary face pierced, even when `x1` lies back
// inside the mesh beyond a cavity (callers wanting the point itself should use
// MeshLocator::locate_near). When an exit point grazes an interior face (within tolerance)
// that `x1` lies strictly beyond, the walk crosses into that neighbour instead of exiting.
// A slightly-outside `x0` (e.g. integration drift) crosses or exits at t = 0; `x0 == x1`
// needs no special case and walks interior-first at t = 0, returning `found` (t = 1) or
// `exited` at t = 0 only when `x0` lies beyond boundary faces alone. Same validation as
// walk_to_point.
SegmentWalkResult walk_along_segment(const MeshProperties& mesh, const std::array<double, 3>& x0,
                                     const std::array<double, 3>& x1, uint32_t start_element,
                                     uint32_t max_steps = MeshLocator::default_max_walk_steps);
