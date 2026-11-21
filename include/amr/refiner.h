#pragma once

#include "mesh_properties.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Refinement: the third of the three parts of adaptive refinement (estimate -> mark ->
// refine). A refiner subdivides the marked tets, and as many others as it needs to keep the
// mesh conforming (no hanging nodes). See src/amr/architecture.md.

// What a refined mesh came from. Indices on the left are those of the refined mesh.
struct Parents {
    // Per tet: the tet of the input mesh that contains its centroid.
    std::vector<uint32_t> elements;
    // Per node: {v, v} for input node v (which keeps its index); {a, b} with a < b for a new
    // node, the midpoint of the nodes a and b of the refined mesh. a and b may be new nodes
    // themselves, but always have smaller indices, so going through the nodes in order
    // gives a P1 function by u[new] = (u[a] + u[b]) / 2. That is exactly the old u_h
    // except inside a green family that red-green refinement reopened (see there).
    std::vector<std::array<uint32_t, 2>> nodes;
};

// What a refiner remembers between cycles. Only red-green refinement uses it: it records
// the green families (tets made only to close the mesh), so that the next refinement can
// undo them instead of subdividing them further. The other refiners ignore it and return
// it empty. Pass the history of the previous result with its mesh; for a mesh that no
// refiner made (imported), pass an empty one.
struct RefinementHistory {
    struct GreenFamily {
        std::array<uint32_t, 4> parent;                       // the tet the family subdivides
        // The midpoint node of each parent edge that is split, edges in the order
        // 01, 02, 03, 12, 13, 23 of the parent's nodes.
        std::array<std::optional<uint32_t>, 6> midpoint;
    };

    std::vector<GreenFamily> families;
    // Per tet: the green family it belongs to, or empty. Either empty (no history) or one
    // entry per tet of the mesh it belongs to.
    std::vector<std::optional<uint32_t>> family;
};

struct RefinementResult {
    struct Info {
        std::string                   refiner;                // the refiner's name()
        double                        time = 0;               // total, in seconds
        uint32_t                      marked = 0;
        uint32_t                      refined_elements = 0;   // input tets that were subdivided
        uint32_t                      elements_before = 0;
        uint32_t                      elements_after = 0;
        uint32_t                      nodes_before = 0;
        uint32_t                      nodes_after = 0;
        // Method-specific diagnostic numbers (see each refiner's header).
        std::map<std::string, double> details;
        std::vector<std::string>      warnings;
    };

    // The refined mesh: nodes (the input nodes first, unchanged), elements (a child keeps
    // its parent's material), boundary_elements (split with the tets, keeping their
    // boundary condition and orientation), neighbours (rebuilt). boundary_conditions,
    // materials and charges are copied.
    MeshProperties    mesh;
    Parents           parents;
    RefinementHistory history;
    Info              info;
};

// The common interface of the refiners. Each refiner has its own Settings, given to its
// constructor and checked there (std::invalid_argument).
//
// refine() is not virtual: it checks the input, times the work, handles "nothing marked",
// and fills the shared metrics; refine_marked() does the refiner's own work.
class Refiner {
public:
    virtual ~Refiner() = default;
    virtual std::string name() const = 0;

    // Refines the marked tets (indices into mesh.elements, any order, repetitions allowed)
    // once, and others as needed for conformity; what "refined" means is the refiner's
    // rule. Nothing marked returns the mesh unchanged. Throws std::invalid_argument for a
    // marked index out of range, a tet with a node index out of range, or a history that
    // does not fit the mesh, and std::runtime_error when the refinement would make more
    // new nodes than the refiner's limit.
    RefinementResult refine(const MeshProperties& mesh, std::span<const uint32_t> marked,
                            const RefinementHistory& history = {}) const;

private:
    // Called by refine() with a checked input and at least one marked tet (sorted, unique).
    // Returns the mesh, parents and history, and may add metrics and warnings.
    virtual RefinementResult refine_marked(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                           const RefinementHistory& history) const = 0;
};
