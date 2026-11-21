// The shared part of every refiner: Refiner::refine.

#include "amr/refiner.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

RefinementResult Refiner::refine(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                 const RefinementHistory& history) const {
    const auto start = std::chrono::steady_clock::now();
    const std::string who = name() + " refiner";

    for (std::size_t k = 0; k < mesh.elements.size(); ++k) {
        for (uint32_t node : mesh.elements[k].node)
            if (node >= mesh.nodes.size())
                throw std::invalid_argument(who + ": tet " + std::to_string(k) + " has node " + std::to_string(node) +
                                            ", but the mesh has " + std::to_string(mesh.nodes.size()) + " nodes");
        if (mesh.elements[k].material >= mesh.materials.size())
            throw std::invalid_argument(who + ": tet " + std::to_string(k) + " has material " +
                                        std::to_string(mesh.elements[k].material) + ", but the mesh has " +
                                        std::to_string(mesh.materials.size()) + " materials");
    }
    for (std::size_t b = 0; b < mesh.boundary_elements.size(); ++b) {
        for (uint32_t node : mesh.boundary_elements[b].node)
            if (node >= mesh.nodes.size())
                throw std::invalid_argument(who + ": boundary element " + std::to_string(b) + " has node " +
                                            std::to_string(node) + ", but the mesh has " +
                                            std::to_string(mesh.nodes.size()) + " nodes");
        if (mesh.boundary_elements[b].boundary_condition >= mesh.boundary_conditions.size())
            throw std::invalid_argument(who + ": boundary element " + std::to_string(b) + " has boundary condition " +
                                        std::to_string(mesh.boundary_elements[b].boundary_condition) +
                                        ", but the mesh has " + std::to_string(mesh.boundary_conditions.size()));
    }
    for (uint32_t k : marked)
        if (k >= mesh.elements.size())
            throw std::invalid_argument(who + ": marked tet " + std::to_string(k) + " is out of range (" +
                                        std::to_string(mesh.elements.size()) + " tets)");
    if (history.family.empty()) {
        if (!history.families.empty())
            throw std::invalid_argument(who + ": the history has " + std::to_string(history.families.size()) +
                                        " green families, but no per-tet family assignments");
    } else {
        if (history.family.size() != mesh.elements.size())
            throw std::invalid_argument(who + ": the history has " + std::to_string(history.family.size()) +
                                        " entries for " + std::to_string(mesh.elements.size()) + " tets");
        for (const auto& family : history.family)
            if (family && *family >= history.families.size())
                throw std::invalid_argument(who + ": the history names green family " + std::to_string(*family) +
                                            " of " + std::to_string(history.families.size()));
        for (const auto& family : history.families) {
            for (uint32_t node : family.parent)
                if (node >= mesh.nodes.size()) throw std::invalid_argument(who + ": the history names a node out of range");
            for (const auto& m : family.midpoint)
                if (m && *m >= mesh.nodes.size()) throw std::invalid_argument(who + ": the history names a node out of range");
        }
    }

    std::vector<uint32_t> sorted(marked.begin(), marked.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    RefinementResult result;
    if (sorted.empty()) {
        result.mesh = mesh;
        result.parents.elements.resize(mesh.elements.size());
        for (uint32_t k = 0; k < mesh.elements.size(); ++k) result.parents.elements[k] = k;
        result.parents.nodes.resize(mesh.nodes.size());
        for (uint32_t i = 0; i < mesh.nodes.size(); ++i) result.parents.nodes[i] = {i, i};
        result.history = history;
    } else {
        result = refine_marked(mesh, sorted, history);
    }

    // Input tets that were subdivided: those that are not the only descendant of themselves.
    std::vector<uint32_t> descendants(mesh.elements.size(), 0);
    for (uint32_t parent : result.parents.elements) ++descendants[parent];
    uint32_t refined = 0;
    for (uint32_t d : descendants)
        if (d != 1) ++refined;

    result.info.refiner = name();
    result.info.marked = static_cast<uint32_t>(sorted.size());
    result.info.refined_elements = refined;
    result.info.elements_before = static_cast<uint32_t>(mesh.elements.size());
    result.info.elements_after = static_cast<uint32_t>(result.mesh.elements.size());
    result.info.nodes_before = static_cast<uint32_t>(mesh.nodes.size());
    result.info.nodes_after = static_cast<uint32_t>(result.mesh.nodes.size());
    result.info.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
