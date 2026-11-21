#pragma once

#include "amr/refiner.h"

#include <cstdint>

// Longest-edge bisection in 3D (Rivara). A tet is bisected at its longest edge: the
// midpoint m of that edge a-b is joined to the other two nodes, giving two children. Ties
// between equally long edges are broken by node indices, the same way in every tet, so
// neighbours agree on which edge is longest.
//
// refine(): the longest edge of every marked tet is marked for splitting. Then, in passes
// until no tet contains a marked edge:
//   1. closure: every tet that contains a marked edge gets its longest edge marked too
//      (repeated until nothing changes; the longest-edge propagation path);
//   2. every tet that contains a marked edge is bisected at its longest edge.
// A marked edge stays marked until every tet around it has split it, so the mesh is
// conforming at the end. A marked tet is bisected at least once (2 or more children).
//
// No state is kept between cycles. Shape: in practice the angles stay bounded; unlike in
// 2D, a bound is not proven for 3D.
//
// Details (in Info::details): passes, bisections, marked_edges.
class LongestEdgeBisection final : public Refiner {
public:
    struct Settings {
        uint64_t max_new_nodes = 10'000'000;      // refine() throws std::runtime_error beyond this
    };

    LongestEdgeBisection();   // default Settings
    explicit LongestEdgeBisection(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "longest_edge_bisection"; }

private:
    RefinementResult refine_marked(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                   const RefinementHistory& history) const override;
    Settings settings_;
};
