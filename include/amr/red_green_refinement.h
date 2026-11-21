#pragma once

#include "amr/refiner.h"

#include <cstdint>

// Red-green refinement in 3D (after Bey).
//
// Red: a tet is split into 8 by the midpoints of its 6 edges: 4 corner tets, and the inner
// octahedron cut into 4 along its shortest diagonal (ties by node indices). Red children
// are similar to their parent at the corners; the shortest diagonal keeps the inner ones
// well shaped.
//
// Green: a tet that is not red but has split edges is closed without new nodes, in one of
// three patterns:
//   - one split edge: bisected into 2;
//   - two split edges on one face: 3 tets; the quadrilateral left on that face is cut along
//     its shorter diagonal (ties by node indices), so the tets on both sides agree;
//   - the three edges of one face split: 4 tets (the face in 4, each joined to the node
//     opposite).
// Any other pattern (two opposite edges, three edges not on one face, four or more) makes
// the tet red, which splits more edges; this repeats until every tet is red, green or
// untouched.
//
// Green tets are never subdivided further. The history (refiner.h) records each green
// family. When a member of a family is marked, or one of its edges gets split, the whole
// family is removed and its parent is put back, red if a member was marked, and then
// treated like any other tet. A family nothing touches is kept as it is. So a cycle only
// ever subdivides red children and reopened parents, and shapes do not degrade from cycle
// to cycle. With an empty history (an imported mesh) every tet counts as regular.
//
// Inside a reopened family the old u_h is piecewise linear on the members, not linear on
// the parent, so Parents::nodes interpolates only approximately there.
//
// Details (in Info::details): red (tets split into 8), green_families (made by this
// refinement), kept_green_families, reopened_green_families.
class RedGreenRefinement final : public Refiner {
public:
    struct Settings {
        uint64_t max_new_nodes = 10'000'000;      // refine() throws std::runtime_error beyond this
    };

    RedGreenRefinement();   // default Settings
    explicit RedGreenRefinement(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "red_green"; }

private:
    RefinementResult refine_marked(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                   const RefinementHistory& history) const override;
    Settings settings_;
};
