#pragma once

#include "mesh_properties.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// Spatial lookup index (AABB BVH + neighbour walking) over a tetrahedral MeshProperties.
//
// Shared across modules that need fast 3D point or range queries:
//   - AMR (locating unhinted point sources),
//   - AnalyticalMaker / P3M (locating trapped charges and cutoff-sphere neighborhoods),
//   - Ray Tracing (initial electron tet lookup and non-convex cavity fallback).
//
// `MeshLocator` holds a non-owning reference to `MeshProperties`: the mesh must outlive
// the locator and remain unchanged while it is queried.
class MeshLocator {
public:
    struct Location {
        uint32_t              element = 0;      // index into MeshProperties::elements
        std::array<double, 4> barycentric{};    // barycentric coordinates in `element`
    };

    // Builds an O(N log N) AABB BVH over `mesh.elements`. Throws std::invalid_argument if
    // any node coordinate is non-finite, any tet/neighbour index is out of range, or any
    // tet has a non-positive volume. An empty mesh is allowed.
    explicit MeshLocator(const MeshProperties& mesh);

    // Cold O(log N) point location via the BVH. Returns the tet containing `x` (within a
    // small barycentric tolerance) and its barycentric coordinates, or empty if `x` lies
    // outside the mesh. Ties on shared faces/edges break deterministically (largest minimum
    // barycentric coordinate, then smallest tet index). Throws std::invalid_argument if `x`
    // is not finite.
    std::optional<Location> locate(const std::array<double, 3>& x) const;

    // Warm point location starting from `hint_element`: walks through `mesh.neighbours`
    // toward `x` first, falling back to the BVH if `mesh.neighbours` is empty, the walk
    // hits a boundary (e.g. a non-convex cavity/hole), or the walk step limit is reached.
    // Throws std::invalid_argument if `x` is not finite or `hint_element` is out of range.
    std::optional<Location> locate_near(const std::array<double, 3>& x, uint32_t hint_element) const;

    // All tets whose closed tetrahedron intersects the closed ball |y - center| <= radius,
    // returned sorted and unique. Throws std::invalid_argument if `center` is not finite or
    // `radius` is negative or not finite.
    std::vector<uint32_t> query_sphere(const std::array<double, 3>& center, double radius) const;

    // All tets whose axis-aligned bounding box intersects the closed box [lo, hi], returned
    // sorted and unique. Throws std::invalid_argument if `lo` or `hi` is not finite or
    // `lo[d] > hi[d]` for any axis d.
    std::vector<uint32_t> query_box(const std::array<double, 3>& lo, const std::array<double, 3>& hi) const;

private:
    struct Aabb {
        std::array<double, 3> lo{};
        std::array<double, 3> hi{};
    };

    struct Node {
        Aabb     box;
        uint32_t left = 0;    // left child index when count == 0; first index in order_ when leaf
        uint32_t right = 0;   // right child index when count == 0
        uint32_t count = 0;   // 0 = internal node; > 0 = leaf with `count` tets
    };

    uint32_t build_node(uint32_t begin, uint32_t end, const std::vector<std::array<double, 3>>& centroids);
    std::array<double, 4> barycentric(uint32_t element, const std::array<double, 3>& x) const;

    const MeshProperties*                            mesh_ = nullptr;
    std::vector<std::array<std::array<double, 3>, 4>> grad_lambda_;
    std::vector<Aabb>                                tet_boxes_;
    std::vector<uint32_t>                            order_;
    std::vector<Node>                                nodes_;
};
