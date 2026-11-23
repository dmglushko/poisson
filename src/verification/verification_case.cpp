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

    // Every mesh material against the case material of its name; the importer keeps one
    // entry per volume name, so a name can occur twice (e.g. "gate 5V" and "gate 7V").
    std::vector<std::optional<std::size_t>> material(mesh.materials.size());
    for (std::size_t m = 0; m < mesh.materials.size(); ++m) {
        const auto& actual = mesh.materials[m];
        for (std::size_t k = 0; k < problem.materials.size(); ++k)
            if (problem.materials[k].name == actual.name) material[m] = k;
        if (!material[m]) {
            out.push_back("the mesh has material '" + actual.name + "', which the case does not have");
            continue;
        }
        const auto& expected = problem.materials[*material[m]];
        if (actual.is_metal != expected.is_metal)
            out.push_back("material '" + expected.name + "' is " + (actual.is_metal ? "" : "not ") + "a metal in the mesh");
        if (actual.voltage != expected.voltage)
            out.push_back("material '" + expected.name + "' has voltage " + (actual.voltage ? number(*actual.voltage) : "none") +
                          " in the mesh, " + (expected.voltage ? number(*expected.voltage) : "none") + " in the case");
        if (!expected.is_metal && actual.eps_r != expected.eps_r)
            out.push_back("material '" + expected.name + "' has eps_r " + number(actual.eps_r) + " in the mesh, " +
                          number(expected.eps_r) + " in the case");
        if (!expected.is_metal && actual.screening_length != expected.screening_length)
            out.push_back("material '" + expected.name + "' has screening_length " + number(actual.screening_length) +
                          " in the mesh, " + number(expected.screening_length) + " in the case");
    }
    for (const auto& expected : problem.materials)
        if (std::none_of(mesh.materials.begin(), mesh.materials.end(),
                         [&](const MeshProperties::Material& m) { return m.name == expected.name; }))
            out.push_back("material '" + expected.name + "' is missing from the mesh");

    for (const auto& condition : problem.boundary_conditions)
        if (std::none_of(mesh.boundary_conditions.begin(), mesh.boundary_conditions.end(),
                         [&](const auto& b) { return same_condition(b, condition); }))
            out.push_back("boundary condition " + describe(condition) + " is missing from the mesh");
    for (const auto& condition : mesh.boundary_conditions)
        if (std::none_of(problem.boundary_conditions.begin(), problem.boundary_conditions.end(),
                         [&](const auto& b) { return same_condition(b, condition); }))
            out.push_back("the mesh has boundary condition " + describe(condition) + ", which the case does not have");

    // Where things are (the case's check), and every boundary triangle an outer face and
    // every outer face a boundary triangle.
    for (auto& m : c.geometry_mismatches(mesh, material)) out.push_back(std::move(m));
    if (mesh.neighbours.size() == mesh.elements.size()) {
        std::set<std::array<uint32_t, 3>> outer;
        for (std::size_t e = 0; e < mesh.elements.size(); ++e)
            for (int i = 0; i < 4; ++i) {
                if (mesh.neighbours[e][i] != MeshProperties::no_neighbour) continue;
                std::array<uint32_t, 3> face;
                for (int j = 0, k = 0; j < 4; ++j)
                    if (j != i) face[k++] = mesh.elements[e].node[j];
                std::sort(face.begin(), face.end());
                outer.insert(face);
            }
        uint64_t inner = 0;
        std::set<std::array<uint32_t, 3>> tagged;
        for (const auto& element : mesh.boundary_elements) {
            auto face = element.node;
            std::sort(face.begin(), face.end());
            tagged.insert(face);
            if (!outer.contains(face)) ++inner;
        }
        uint64_t untagged = 0;
        for (const auto& face : outer)
            if (!tagged.contains(face)) ++untagged;
        if (inner > 0) out.push_back(std::to_string(inner) + " boundary triangles are inner faces of the mesh");
        if (untagged > 0) out.push_back(std::to_string(untagged) + " outer tet faces have no boundary condition");
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
