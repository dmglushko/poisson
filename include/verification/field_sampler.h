#pragma once

#include "mesh_properties.h"
#include "verification/verification_case.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Values of a P1 nodal field at chosen points, so that branches with different meshes can be
// compared at the same places. With a case, the exact potential is evaluated at the same
// points with the material of the same located tet, so the two can be plotted together.
struct PointSamples {
    static constexpr uint16_t no_material = std::numeric_limits<uint16_t>::max();

    std::vector<std::array<double, 3>> points;
    std::vector<uint8_t>               found;        // 1 if the point is inside the mesh
    std::vector<uint32_t>              element;      // containing tet, or MeshProperties::no_neighbour
    std::vector<uint16_t>              material;     // index into mesh.materials, or no_material
    std::vector<double>                u;            // NaN outside the mesh
    std::vector<std::array<double, 3>> grad;         // gradient of the P1 field in the tet; NaN outside
    std::vector<double>                u_exact;      // empty without a case; NaN outside the mesh
    std::vector<std::array<double, 3>> grad_exact;   // empty without a case
};

struct SamplingResult {
    struct Info {
        double                   time = 0;
        std::vector<std::string> warnings;           // e.g. points outside the mesh
        uint64_t                 outside = 0;        // number of points outside the mesh
    };

    PointSamples samples;
    Info         info;
};

class FieldSampler {
public:
    // u: one finite value per node of `mesh`. Throws std::invalid_argument if the sizes
    // differ or a value is not finite.
    // With `exact`, every mesh material must be a material of the case (by name).
    SamplingResult sample(const MeshProperties& mesh, const std::vector<double>& u,
                          const std::vector<std::array<double, 3>>& points,
                          const VerificationCase* exact = nullptr) const;

    // n >= 2 equally spaced points from `from` to `to`, both included.
    static std::vector<std::array<double, 3>> line(const std::array<double, 3>& from, const std::array<double, 3>& to,
                                                   uint32_t n);
};
