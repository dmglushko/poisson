#pragma once

// Private helpers for evaluating a case on a mesh.

#include "mesh_properties.h"
#include "verification/verification_case.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace verification_detail {

// For every mesh material, the index of the case material with the same name. Throws
// std::invalid_argument if the case has no such material.
inline std::vector<std::size_t> case_materials(const MeshProperties& mesh, const VerificationCase& c) {
    std::vector<std::size_t> out;
    out.reserve(mesh.materials.size());
    for (const auto& m : mesh.materials) {
        try {
            out.push_back(c.material_index(m.name));
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("the mesh has material '" + m.name + "', which case '" + std::string(c.name()) +
                                        "' does not have");
        }
    }
    return out;
}

// Gradients of the four P1 basis functions of a tet, and its volume.
inline void p1_gradients(const MeshProperties& mesh, const MeshProperties::Element& element,
                         std::array<std::array<double, 3>, 4>& grad, double& volume) {
    const auto& p0 = mesh.nodes[element.node[0]];
    double e[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) e[i][k] = mesh.nodes[element.node[i + 1]][k] - p0[k];
    // With the edges e_i = p_{i+1} - p_0 as rows of E, the barycentric coordinates 1..3 are
    // E^-T (x - p_0), so their gradients are the columns of E^-1.
    const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                       e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    if (det == 0) throw std::invalid_argument("the mesh has a tet of zero volume");
    const double inv = 1 / det;
    const std::array<std::array<double, 3>, 3> cofactor{{
        {(e[1][1] * e[2][2] - e[1][2] * e[2][1]) * inv, (e[1][2] * e[2][0] - e[1][0] * e[2][2]) * inv,
         (e[1][0] * e[2][1] - e[1][1] * e[2][0]) * inv},
        {(e[0][2] * e[2][1] - e[0][1] * e[2][2]) * inv, (e[0][0] * e[2][2] - e[0][2] * e[2][0]) * inv,
         (e[0][1] * e[2][0] - e[0][0] * e[2][1]) * inv},
        {(e[0][1] * e[1][2] - e[0][2] * e[1][1]) * inv, (e[0][2] * e[1][0] - e[0][0] * e[1][2]) * inv,
         (e[0][0] * e[1][1] - e[0][1] * e[1][0]) * inv},
    }};
    // grad lambda_{i+1} is column i of the inverse, which is cofactor row i (e.g. e1 x e2 / det).
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) grad[i + 1][k] = cofactor[i][k];
    for (int k = 0; k < 3; ++k) grad[0][k] = -(grad[1][k] + grad[2][k] + grad[3][k]);
    volume = std::abs(det) / 6;
}

}   // namespace verification_detail
