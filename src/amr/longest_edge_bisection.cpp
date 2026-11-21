#include "amr/longest_edge_bisection.h"

#include "refinement_common.h"

#include <stdexcept>
#include <unordered_set>

using namespace amr_detail;

LongestEdgeBisection::LongestEdgeBisection() : LongestEdgeBisection(Settings{}) {}

LongestEdgeBisection::LongestEdgeBisection(Settings settings) : settings_(settings) {
    if (settings_.max_new_nodes == 0)
        throw std::invalid_argument("LongestEdgeBisection::Settings::max_new_nodes must be positive");
}

RefinementResult LongestEdgeBisection::refine_marked(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                                     const RefinementHistory& history) const {
    WorkMesh work(mesh, settings_.max_new_nodes, name() + " refiner");
    std::vector<WorkTet> tets = initial_tets(mesh);

    // The longest edge of a tet as local nodes: longest first, then the smaller node pair,
    // so every tet around an edge ranks it the same.
    const auto longest = [&](const WorkTet& t) {
        int best = 0;
        double best_length = -1;
        uint64_t best_key = 0;
        for (int e = 0; e < 6; ++e) {
            const uint32_t a = t.node[tet_edges[e].first], b = t.node[tet_edges[e].second];
            const double length = work.length2(a, b);
            const uint64_t key = edge_key(a, b);
            if (length > best_length || (length == best_length && key < best_key)) {
                best = e;
                best_length = length;
                best_key = key;
            }
        }
        return tet_edges[best];
    };
    const auto key_of = [](const WorkTet& t, std::pair<int, int> e) { return edge_key(t.node[e.first], t.node[e.second]); };

    std::unordered_set<uint64_t> marked_edges;
    const auto has_marked_edge = [&](const WorkTet& t) {
        for (const auto& e : tet_edges)
            if (marked_edges.contains(key_of(t, e))) return true;
        return false;
    };

    for (uint32_t k : marked) marked_edges.insert(key_of(tets[k], longest(tets[k])));

    uint32_t passes = 0, bisections = 0;
    while (true) {
        // Closure: a tet with a marked edge is bisected at its longest edge, so mark that too.
        for (bool changed = true; changed;) {
            changed = false;
            for (const WorkTet& t : tets)
                if (has_marked_edge(t) && marked_edges.insert(key_of(t, longest(t))).second) changed = true;
        }
        std::vector<WorkTet> next;
        next.reserve(tets.size() + tets.size() / 2);
        bool any = false;
        for (const WorkTet& t : tets) {
            if (!has_marked_edge(t)) {
                next.push_back(t);
                continue;
            }
            const auto [a, b] = longest(t);
            ChildTemplate first{vertex(0), vertex(1), vertex(2), vertex(3)};
            ChildTemplate second = first;
            first[b] = edge(a, b);    // the half at node a
            second[a] = edge(a, b);   // the half at node b
            const ChildTemplate children[] = {first, second};
            work.subdivide(t, children, next);
            any = true;
            ++bisections;
        }
        if (!any) break;
        tets.swap(next);
        ++passes;
    }

    RefinementResult result = output(work, tets);
    if (!history.family.empty())
        result.info.warnings.push_back("the history (green families of red-green refinement) was ignored");
    result.info.details["passes"] = passes;
    result.info.details["bisections"] = bisections;
    result.info.details["marked_edges"] = static_cast<double>(marked_edges.size());
    return result;
}
