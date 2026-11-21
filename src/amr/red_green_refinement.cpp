#include "amr/red_green_refinement.h"

#include "refinement_common.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <stdexcept>

using namespace amr_detail;

namespace {

using Vec3 = std::array<double, 3>;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The smallest barycentric coordinate of x in input tet k: >= 0 inside.
double inside_measure(const MeshProperties& mesh, uint32_t k, const Vec3& x) {
    const auto& n = mesh.elements[k].node;
    const double whole = volume6(mesh.nodes, n);
    double smallest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        std::array<Vec3, 4> p{mesh.nodes[n[0]], mesh.nodes[n[1]], mesh.nodes[n[2]], mesh.nodes[n[3]]};
        p[i] = x;
        const double part = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])));
        smallest = std::min(smallest, part / whole);
    }
    return smallest;
}

// Split edges of a tet as a 6-bit mask, bit e for tet_edges[e].
using Pattern = unsigned;

// The edge index of local nodes i and j.
int edge_index(int i, int j) {
    for (int e = 0; e < 6; ++e)
        if ((tet_edges[e].first == i && tet_edges[e].second == j) || (tet_edges[e].first == j && tet_edges[e].second == i))
            return e;
    throw std::logic_error("no such edge");
}

// The edges of the face opposite node d.
Pattern face_edges(int d) {
    Pattern p = 0;
    for (int e = 0; e < 6; ++e)
        if (tet_edges[e].first != d && tet_edges[e].second != d) p |= 1u << e;
    return p;
}

// The face (opposite node) whose edges all of `p` lie on, if one exists.
std::optional<int> common_face(Pattern p) {
    for (int d = 0; d < 4; ++d)
        if ((p & ~face_edges(d)) == 0) return d;
    return std::nullopt;
}

// Green closure exists for: one edge; two edges of one face; the three edges of one face.
bool green_allowed(Pattern p) {
    const int count = std::popcount(p);
    if (count == 0 || count == 1) return true;
    if (count == 2 || count == 3) return common_face(p).has_value();
    return false;
}

} // namespace

RedGreenRefinement::RedGreenRefinement() : RedGreenRefinement(Settings{}) {}

RedGreenRefinement::RedGreenRefinement(Settings settings) : settings_(settings) {
    if (settings_.max_new_nodes == 0)
        throw std::invalid_argument("RedGreenRefinement::Settings::max_new_nodes must be positive");
}

RefinementResult RedGreenRefinement::refine_marked(const MeshProperties& mesh, std::span<const uint32_t> marked,
                                                   const RefinementHistory& history) const {
    const std::string who = name() + " refiner";
    WorkMesh work(mesh, settings_.max_new_nodes, who);
    std::vector<WorkTet> tets = initial_tets(mesh);

    // The history: green members (private to red-green), and the midpoints of their parents' split edges.
    std::vector<int32_t> tet_green(tets.size(), -1);
    std::vector<int32_t> tet_reopened(tets.size(), -1);
    std::vector<std::vector<uint32_t>> reopened_members;
    std::vector<std::vector<uint32_t>> members_of(history.families.size());
    if (!history.family.empty()) {
        for (uint32_t k = 0; k < tets.size(); ++k)
            if (history.family[k]) {
                tet_green[k] = static_cast<int32_t>(*history.family[k]);
                members_of[*history.family[k]].push_back(k);
            }
        for (const auto& family : history.families)
            for (int e = 0; e < 6; ++e)
                if (family.midpoint[e])
                    work.set_midpoint(family.parent[tet_edges[e].first], family.parent[tet_edges[e].second], *family.midpoint[e]);
    }

    std::vector<char> alive(tets.size(), 1), must_red(tets.size(), 0);
    std::vector<char> reopened(history.families.size(), 0);
    uint32_t reopened_count = 0, red_count = 0;

    const auto split_pattern = [&](const WorkTet& t) {
        Pattern p = 0;
        for (int e = 0; e < 6; ++e)
            if (work.find_midpoint(t.node[tet_edges[e].first], t.node[tet_edges[e].second])) p |= 1u << e;
        return p;
    };
    const auto is_split = [&](uint32_t a, uint32_t b) { return work.find_midpoint(a, b).has_value(); };

    // A tet must be red if its pattern has no green closure, or if something finer than its
    // own edges is split on its boundary: half of a split edge, or a line inside a face from
    // a midpoint (to the opposite node or to another midpoint). Green closure would leave
    // those nodes hanging.
    const auto needs_red = [&](const WorkTet& t, Pattern p) {
        if (!green_allowed(p)) return true;
        std::array<std::optional<uint32_t>, 6> m;
        for (int e = 0; e < 6; ++e)
            if (p & (1u << e)) {
                const uint32_t a = t.node[tet_edges[e].first], b = t.node[tet_edges[e].second];
                m[e] = work.find_midpoint(a, b);
                if (is_split(a, *m[e]) || is_split(*m[e], b)) return true;
            }
        for (int d = 0; d < 4; ++d) {
            int corners[3], c = 0;
            for (int i = 0; i < 4; ++i)
                if (i != d) corners[c++] = i;
            for (int x = 0; x < 3; ++x) {
                const int i = corners[x], j = corners[(x + 1) % 3], opposite = corners[(x + 2) % 3];
                const int e = edge_index(i, j);
                if (!m[e]) continue;
                if (is_split(*m[e], t.node[opposite])) return true;
                const int f = edge_index(j, opposite);
                if (m[f] && is_split(*m[e], *m[f])) return true;
            }
        }
        return false;
    };

    // Replaces green family g by its parent, red if `red`.
    const auto reopen = [&](uint32_t g, bool red) {
        if (reopened[g]) return;
        reopened[g] = 1;
        ++reopened_count;
        const auto& family = history.families[g];
        WorkTet parent;
        parent.node = family.parent;
        parent.material = tets[members_of[g].front()].material;
        parent.ancestor = members_of[g].front();
        parent.face_tag = {no_tag, no_tag, no_tag, no_tag};
        const int32_t reopened_id = static_cast<int32_t>(reopened_members.size());
        reopened_members.push_back(members_of[g]);
        // The parent's face tags, from the member faces that lie in them.
        for (uint32_t k : members_of[g]) {
            alive[k] = 0;
            const WorkTet& member = tets[k];
            std::array<unsigned, 4> mask{};
            for (int q = 0; q < 4; ++q) {
                for (int i = 0; i < 4; ++i)
                    if (member.node[q] == family.parent[i]) mask[q] = 1u << i;
                for (int e = 0; e < 6 && mask[q] == 0; ++e)
                    if (family.midpoint[e] == member.node[q])
                        mask[q] = (1u << tet_edges[e].first) | (1u << tet_edges[e].second);
                if (mask[q] == 0)
                    throw std::invalid_argument(who + ": the history does not fit the mesh (tet " + std::to_string(k) +
                                                " is not part of the parent of its green family)");
            }
            for (int q = 0; q < 4; ++q) {
                unsigned used = 0;
                for (int r = 0; r < 4; ++r)
                    if (r != q) used |= mask[r];
                if (std::popcount(used) == 3 && member.face_tag[q] != no_tag)
                    parent.face_tag[std::countr_zero(~used & 0xFu)] = member.face_tag[q];
            }
        }
        tets.push_back(parent);
        tet_green.push_back(-1);
        tet_reopened.push_back(reopened_id);
        alive.push_back(1);
        must_red.push_back(red ? 1 : 0);
    };

    for (uint32_t k : marked) {
        if (tet_green[k] >= 0)
            reopen(static_cast<uint32_t>(tet_green[k]), true);
        else
            must_red[k] = 1;
    }

    // Red: the octahedron is cut along its shortest diagonal.
    const auto red_children = [&](const WorkTet& t) {
        std::array<uint32_t, 6> m;
        for (int e = 0; e < 6; ++e) m[e] = work.midpoint(t.node[tet_edges[e].first], t.node[tet_edges[e].second]);
        // The diagonals join midpoints of opposite edges: 01-23, 02-13, 03-12.
        const std::array<std::pair<int, int>, 3> diagonals{{{0, 5}, {1, 4}, {2, 3}}};
        int best = 0;
        for (int d = 1; d < 3; ++d) {
            const double l = work.length2(m[diagonals[d].first], m[diagonals[d].second]);
            const double lb = work.length2(m[diagonals[best].first], m[diagonals[best].second]);
            if (l < lb || (l == lb && edge_key(m[diagonals[d].first], m[diagonals[d].second]) <
                                          edge_key(m[diagonals[best].first], m[diagonals[best].second])))
                best = d;
        }
        const auto mid = [](int e) { return edge(tet_edges[e].first, tet_edges[e].second); };
        std::vector<ChildTemplate> children{
            {vertex(0), edge(0, 1), edge(0, 2), edge(0, 3)},
            {edge(0, 1), vertex(1), edge(1, 2), edge(1, 3)},
            {edge(0, 2), edge(1, 2), vertex(2), edge(2, 3)},
            {edge(0, 3), edge(1, 3), edge(2, 3), vertex(3)},
        };
        // The 4 other midpoints around the diagonal, in a cycle (consecutive ones are not opposite).
        const std::array<std::array<int, 4>, 3> rings{{{1, 2, 4, 3}, {0, 2, 5, 3}, {0, 1, 5, 4}}};
        const auto [p, q] = diagonals[best];
        for (int r = 0; r < 4; ++r)
            children.push_back({mid(p), mid(q), mid(rings[best][r]), mid(rings[best][(r + 1) % 4])});
        return children;
    };

    while (true) {
        bool changed = false;
        for (std::size_t i = 0, n = tets.size(); i < n; ++i) {
            if (!alive[i] || !must_red[i]) continue;
            alive[i] = 0;
            const WorkTet t = tets[i];
            const int32_t reopened_id = tet_reopened[i];
            std::vector<WorkTet> children;
            work.subdivide(t, red_children(t), children);
            for (const WorkTet& child : children) {
                tets.push_back(child);
                tet_green.push_back(-1);
                tet_reopened.push_back(reopened_id);
                alive.push_back(1);
                must_red.push_back(0);
            }
            ++red_count;
            changed = true;
        }
        for (std::size_t i = 0; i < tets.size(); ++i) {
            if (!alive[i] || must_red[i]) continue;
            const WorkTet t = tets[i];
            const Pattern p = split_pattern(t);
            if (p == 0) continue;
            if (tet_green[i] >= 0) {
                // A green tet is never subdivided: put its parent back instead.
                reopen(static_cast<uint32_t>(tet_green[i]), false);
                changed = true;
            } else if (needs_red(t, p)) {
                must_red[i] = 1;
                changed = true;
            }
        }
        if (!changed) break;
    }

    // Emit: untouched tets and kept green families as they are, the rest closed green.
    std::vector<WorkTet> out;
    std::vector<int32_t> out_reopened;
    RefinementHistory new_history;
    std::map<int32_t, uint32_t> kept;   // input family -> its index in new_history
    uint32_t green_count = 0;
    for (std::size_t i = 0; i < tets.size(); ++i) {
        if (!alive[i]) continue;
        const WorkTet t = tets[i];
        if (tet_green[i] >= 0) {
            auto [it, inserted] = kept.try_emplace(tet_green[i], static_cast<uint32_t>(new_history.families.size()));
            if (inserted) new_history.families.push_back(history.families[tet_green[i]]);
            out.push_back(t);
            out_reopened.push_back(-1);
            new_history.family.push_back(it->second);
            continue;
        }
        const Pattern p = split_pattern(t);
        if (p == 0) {
            out.push_back(t);
            out_reopened.push_back(tet_reopened[i]);
            new_history.family.emplace_back();
            continue;
        }
        if (needs_red(t, p)) throw std::logic_error(who + ": a tet left for green closure needs red refinement");

        std::vector<ChildTemplate> children;
        const int count = std::popcount(p);
        if (count == 1) {
            const auto [a, b] = tet_edges[std::countr_zero(p)];
            ChildTemplate first{vertex(0), vertex(1), vertex(2), vertex(3)};
            ChildTemplate second = first;
            first[b] = edge(a, b);
            second[a] = edge(a, b);
            children = {first, second};
        } else {
            const int d = *common_face(p);
            if (count == 3) {
                int c[3], n = 0;
                for (int i2 = 0; i2 < 4; ++i2)
                    if (i2 != d) c[n++] = i2;
                const int a = c[0], b = c[1], e = c[2];
                children = {{vertex(a), edge(a, b), edge(a, e), vertex(d)},
                            {edge(a, b), vertex(b), edge(b, e), vertex(d)},
                            {edge(a, e), edge(b, e), vertex(e), vertex(d)},
                            {edge(a, b), edge(b, e), edge(a, e), vertex(d)}};
            } else {
                // Two split edges a-b and a-c on the face (a, b, c): the corner (a, ab, ac),
                // and the quadrilateral ab, b, c, ac cut along its shorter diagonal.
                const auto [e1, e2] = std::pair{std::countr_zero(p), 31 - std::countl_zero(p)};
                const auto [p1, q1] = tet_edges[e1];
                const auto [p2, q2] = tet_edges[e2];
                const int a = (p1 == p2 || p1 == q2) ? p1 : q1;
                const int b = a == p1 ? q1 : p1;
                const int c = a == p2 ? q2 : p2;
                const uint32_t m_ab = *work.find_midpoint(t.node[a], t.node[b]);
                const uint32_t m_ac = *work.find_midpoint(t.node[a], t.node[c]);
                const double l1 = work.length2(m_ab, t.node[c]), l2 = work.length2(m_ac, t.node[b]);
                const bool from_ab = l1 < l2 || (l1 == l2 && edge_key(m_ab, t.node[c]) < edge_key(m_ac, t.node[b]));
                children.push_back({vertex(a), edge(a, b), edge(a, c), vertex(d)});
                if (from_ab) {
                    children.push_back({edge(a, b), vertex(b), vertex(c), vertex(d)});
                    children.push_back({edge(a, b), vertex(c), edge(a, c), vertex(d)});
                } else {
                    children.push_back({edge(a, b), vertex(b), edge(a, c), vertex(d)});
                    children.push_back({vertex(b), vertex(c), edge(a, c), vertex(d)});
                }
            }
        }
        RefinementHistory::GreenFamily family;
        family.parent = t.node;
        for (int e = 0; e < 6; ++e) family.midpoint[e] = work.find_midpoint(t.node[tet_edges[e].first], t.node[tet_edges[e].second]);
        const uint32_t index = static_cast<uint32_t>(new_history.families.size());
        new_history.families.push_back(family);
        const std::size_t before = out.size();
        work.subdivide(t, children, out);
        for (std::size_t j = before; j < out.size(); ++j) {
            out_reopened.push_back(tet_reopened[i]);
            new_history.family.push_back(index);
        }
        ++green_count;
    }

    // Resolve the input parent for every final descendant of a reopened green family.
    for (std::size_t j = 0; j < out.size(); ++j) {
        if (out_reopened[j] < 0) continue;
        Vec3 centroid{0, 0, 0};
        for (uint32_t node : out[j].node)
            for (int d = 0; d < 3; ++d) centroid[d] += 0.25 * work.nodes[node][d];
        double best = -std::numeric_limits<double>::infinity();
        for (uint32_t member : reopened_members[out_reopened[j]]) {
            const double measure = inside_measure(mesh, member, centroid);
            if (measure > best) {
                best = measure;
                out[j].ancestor = member;
            }
        }
    }

    RefinementResult result = output(work, out);
    result.history = std::move(new_history);
    auto& d = result.info.details;
    d["red"] = red_count;
    d["green_families"] = green_count;
    d["kept_green_families"] = static_cast<double>(kept.size());
    d["reopened_green_families"] = reopened_count;
    return result;
}
