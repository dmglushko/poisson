#pragma once

#include "amr/refinement_input.h"
#include "mesh_properties.h"

#include <map>
#include <string>
#include <vector>

// Error estimation: the first of the three parts of adaptive refinement
// (estimate -> mark -> refine). An estimator turns a solution into one error indicator per
// tet. See src/amr/architecture.md.

// What an estimator does with the tets around a point source. Near a delta, u_h grows
// without bound under refinement, so every estimator would keep refining there forever.
enum class PointSourceRule {
    // eta_K = 0 for every tet that shares a node with a tet containing a point source.
    exclude_patch,
    // Nothing: the point sources are not looked at.
    ignore,
};

// What an estimator returns.
struct ErrorEstimate {
    struct Info {
        std::string                   estimator;              // the estimator's name()
        double                        time = 0;               // total, in seconds
        uint32_t                      elements = 0;
        double                        max_indicator = 0;
        uint32_t                      point_sources = 0;
        uint32_t                      excluded_elements = 0;
        // Method-specific diagnostic numbers (see each estimator's header).
        std::map<std::string, double> details;
        std::vector<std::string>      warnings;               // e.g. a point source outside the mesh
    };

    std::vector<double> eta;                      // eta_K per tet, same index as MeshProperties::elements
    double              total = 0;                // sqrt(sum of eta_K^2)
    Info                info;
};

// The common interface of the error estimators. Each estimator has its own Settings, given
// to its constructor and checked there (std::invalid_argument).
//
// estimate() is not virtual: it checks the input, times the work, applies the point source
// rule, adds the total and the shared metrics, and asks the estimator for the indicators
// (indicators()). So the checks are written once and no estimator can skip them.
class ErrorEstimator {
public:
    virtual ~ErrorEstimator() = default;
    virtual std::string name() const = 0;

    // The error indicators of the solution input.u of the problem on `mesh`. Throws
    // std::invalid_argument for a bad input, naming it: u not one finite value per node; no
    // source for an estimator that needs one; a point source that is not finite or names
    // an element out of range; a tet with a volume that is not positive; a source value
    // that is not finite. A metal tet (Material::is_metal) has eta_K = 0: u is one constant
    // there.
    ErrorEstimate estimate(const MeshProperties& mesh, const RefinementInput& input) const;

protected:
    explicit ErrorEstimator(PointSourceRule point_sources) : point_sources_(point_sources) {}

private:
    // Called by estimate() with a checked input and the point-source exclusion mask
    // (excluded[k] != 0 means tet k's indicator will be forced to 0). Returns one finite,
    // non-negative value per tet, and may add details and warnings to `info`.
    virtual std::vector<double> indicators(const MeshProperties& mesh, const RefinementInput& input,
                                           std::span<const uint8_t> excluded, ErrorEstimate::Info& info) const = 0;
    // Whether indicators() calls input.source.
    virtual bool needs_source() const = 0;

    PointSourceRule point_sources_;
};
