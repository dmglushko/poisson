#include "verification/verification_case.h"

#include "case_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>

using namespace verification_detail;

namespace {

void check_point(const std::array<double, 3>& x, std::size_t material, std::size_t materials) {
    if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2]))
        throw std::invalid_argument("VerificationCase: the point must be finite");
    if (material >= materials)
        throw std::invalid_argument("VerificationCase: material index " + std::to_string(material) + " out of range");
}

std::string describe(const MeshProperties::BoundaryCondition& condition) {
    return "'" + surface_name(condition) + "'";
}

}   // namespace

double VerificationCase::u(const std::array<double, 3>& x, std::size_t material) const {
    check_point(x, material, problem_.materials.size());
    return exact_u(x, material);
}

std::array<double, 3> VerificationCase::grad_u(const std::array<double, 3>& x, std::size_t material) const {
    check_point(x, material, problem_.materials.size());
    return exact_grad(x, material);
}

std::optional<MeshProperties::BoundaryCondition> VerificationCase::expected_condition(const std::array<std::array<double, 3>, 3>& triangle) const {
    for (const auto& p : triangle) check_point(p, 0, 1);
    return condition_on(triangle);
}

bool VerificationCase::fits(const std::array<std::array<double, 3>, 4>& tet, std::size_t material) const {
    for (const auto& p : tet) check_point(p, material, problem_.materials.size());
    return fits_region(tet, material);
}

std::size_t VerificationCase::material_index(std::string_view name) const {
    for (std::size_t m = 0; m < problem_.materials.size(); ++m)
        if (problem_.materials[m].name == name) return m;
    throw std::invalid_argument(std::string(this->name()) + ": no material named '" + std::string(name) + "'");
}

std::vector<std::string> mesh_mismatches(const MeshProperties& mesh, const VerificationCase& c) {
    const VerificationProblem& problem = c.problem();
    std::vector<std::string> out;

    if (mesh.screened != problem.screened)
        out.push_back(std::string("the mesh is ") + (mesh.screened ? "screened" : "unscreened") + " but the case is " +
                      (problem.screened ? "screened" : "unscreened"));

    for (const auto& expected : problem.materials) {
        const auto it = std::find_if(mesh.materials.begin(), mesh.materials.end(),
                                     [&](const MeshProperties::Material& m) { return m.name == expected.name; });
        if (it == mesh.materials.end()) {
            out.push_back("material '" + expected.name + "' is missing from the mesh");
            continue;
        }
        if (it->is_metal != expected.is_metal)
            out.push_back("material '" + expected.name + "' is " + (it->is_metal ? "" : "not ") + "a metal in the mesh");
        if (it->voltage != expected.voltage)
            out.push_back("material '" + expected.name + "' has voltage " + (it->voltage ? number(*it->voltage) : "none") +
                          " in the mesh, " + (expected.voltage ? number(*expected.voltage) : "none") + " in the case");
        if (!expected.is_metal && it->eps_r != expected.eps_r)
            out.push_back("material '" + expected.name + "' has eps_r " + number(it->eps_r) + " in the mesh, " +
                          number(expected.eps_r) + " in the case");
        if (!expected.is_metal && it->screening_length != expected.screening_length)
            out.push_back("material '" + expected.name + "' has screening_length " + number(it->screening_length) +
                          " in the mesh, " + number(expected.screening_length) + " in the case");
    }
    for (const auto& material : mesh.materials)
        if (std::none_of(problem.materials.begin(), problem.materials.end(),
                         [&](const MeshProperties::Material& m) { return m.name == material.name; }))
            out.push_back("the mesh has material '" + material.name + "', which the case does not have");

    for (const auto& condition : problem.boundary_conditions)
        if (std::none_of(mesh.boundary_conditions.begin(), mesh.boundary_conditions.end(),
                         [&](const auto& b) { return same_condition(b, condition); }))
            out.push_back("boundary condition " + describe(condition) + " is missing from the mesh");
    for (const auto& condition : mesh.boundary_conditions)
        if (std::none_of(problem.boundary_conditions.begin(), problem.boundary_conditions.end(),
                         [&](const auto& b) { return same_condition(b, condition); }))
            out.push_back("the mesh has boundary condition " + describe(condition) + ", which the case does not have");

    // Where things are: every tet inside its material's region, every boundary triangle with
    // the condition the case has there, and no outer face without a condition.
    {
        std::vector<std::optional<std::size_t>> material(mesh.materials.size());
        for (std::size_t m = 0; m < mesh.materials.size(); ++m)
            for (std::size_t k = 0; k < problem.materials.size(); ++k)
                if (problem.materials[k].name == mesh.materials[m].name) material[m] = k;
        uint64_t misplaced = 0;
        std::size_t first = 0;
        for (std::size_t e = 0; e < mesh.elements.size(); ++e) {
            const auto& element = mesh.elements[e];
            if (element.material >= material.size() || !material[element.material]) continue;
            std::array<std::array<double, 3>, 4> tet;
            for (int i = 0; i < 4; ++i) tet[i] = mesh.nodes[element.node[i]];
            if (!c.fits(tet, *material[element.material]) && misplaced++ == 0) first = e;
        }
        if (misplaced > 0)
            out.push_back(std::to_string(misplaced) + " tets lie outside the region of their material (first: tet " +
                          std::to_string(first) + ", material '" + mesh.materials[mesh.elements[first].material].name + "')");

        uint64_t off_boundary = 0, wrong = 0;
        std::string first_off, first_wrong;
        for (std::size_t b = 0; b < mesh.boundary_elements.size(); ++b) {
            const auto& element = mesh.boundary_elements[b];
            std::array<std::array<double, 3>, 3> triangle;
            for (int i = 0; i < 3; ++i) triangle[i] = mesh.nodes[element.node[i]];
            const auto expected = c.expected_condition(triangle);
            if (!expected) {
                if (off_boundary++ == 0) first_off = std::to_string(b);
                continue;
            }
            if (element.boundary_condition >= mesh.boundary_conditions.size()) continue;
            const auto& actual = mesh.boundary_conditions[element.boundary_condition];
            if (!same_condition(actual, *expected) && wrong++ == 0)
                first_wrong = "triangle " + std::to_string(b) + " has " + describe(actual) + ", the case " + describe(*expected);
        }
        if (off_boundary > 0)
            out.push_back(std::to_string(off_boundary) + " boundary triangles do not lie on the boundary of the case (first: triangle " +
                          first_off + ")");
        if (wrong > 0) out.push_back(std::to_string(wrong) + " boundary triangles have the wrong condition (first: " + first_wrong + ")");

        if (mesh.neighbours.size() == mesh.elements.size()) {
            std::set<std::array<uint32_t, 3>> tagged;
            for (const auto& element : mesh.boundary_elements) {
                auto face = element.node;
                std::sort(face.begin(), face.end());
                tagged.insert(face);
            }
            uint64_t untagged = 0;
            for (std::size_t e = 0; e < mesh.elements.size(); ++e)
                for (int i = 0; i < 4; ++i) {
                    if (mesh.neighbours[e][i] != MeshProperties::no_neighbour) continue;
                    std::array<uint32_t, 3> face;
                    for (int j = 0, k = 0; j < 4; ++j)
                        if (j != i) face[k++] = mesh.elements[e].node[j];
                    std::sort(face.begin(), face.end());
                    if (!tagged.contains(face)) ++untagged;
                }
            if (untagged > 0) out.push_back(std::to_string(untagged) + " outer tet faces have no boundary condition");
        }
    }

    if (mesh.charges.size() != problem.charges.size()) {
        out.push_back("the mesh has " + std::to_string(mesh.charges.size()) + " trapped charges, the case " +
                      std::to_string(problem.charges.size()));
    } else {
        for (std::size_t i = 0; i < problem.charges.size(); ++i)
            if (mesh.charges[i].position != problem.charges[i].position || mesh.charges[i].value != problem.charges[i].value)
                out.push_back("trapped charge " + std::to_string(i) + " differs between the mesh and the case");
    }
    return out;
}
