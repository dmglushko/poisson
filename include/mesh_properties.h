#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// The mesh and everything attached to it. Six vectors linked by index: an element stores
// only node indices and a material index; coordinates, materials and boundary conditions
// are looked up in the other vectors.
struct MeshProperties {
    struct Element {                              // volume element (tet)
        std::array<uint32_t, 4> node;             // indices into nodes
        uint16_t material;                        // index into materials
    };

    struct BoundaryElement {                      // surface element (triangle)
        std::array<uint32_t, 3> node;             // indices into nodes
        uint16_t boundary_condition;              // index into boundary_conditions
    };

    enum class BcType { Dirichlet, Neumann, Robin };
    struct BoundaryCondition {
        BcType type;
        double value;                             // Dirichlet: u [V]; Neumann and Robin: g [V/nm]
        std::optional<double> alpha;              // Robin only [1/nm]: eps_r * du/dn + alpha * u = g
    };

    struct Material {
        std::string name;                         // e.g. "silicondioxide"
        double eps_r = 1;                         // relative permittivity
        double screening_length = 0;              // nm; 0 = no screening, else c = eps_r / screening_length^2
        bool is_metal = false;                    // conductor: one potential everywhere inside
        std::optional<double> voltage;            // prescribed potential [V]; empty if not prescribed
    };

    struct Charge {                               // trapped charge
        std::array<double, 3> position;           // x, y, z in nm
        int32_t               value;              // in elementary charges: +1 hole, -1 electron
    };

    std::vector<std::array<double, 3>> nodes;               // nodes[id] = {x, y, z} in nm
    std::vector<Element>               elements;
    std::vector<BoundaryElement>       boundary_elements;
    std::vector<BoundaryCondition>     boundary_conditions;
    std::vector<Material>              materials;
    std::vector<Charge>                charges;             // all trapped charges; grows after every batch

    // PROPOSED, not fully decided: the tets around each tet. neighbours[e][i] is the tet on
    // the other side of the face opposite node i of element e, or empty if that face is on
    // the outside of the mesh. Same index as elements.
    // std::optional doubles the memory here (32 instead of 16 bytes per tet). If that turns
    // out to matter, a plain uint32_t with a reserved "no neighbour" value may be better.
    std::vector<std::array<std::optional<uint32_t>, 4>> neighbours;
};
