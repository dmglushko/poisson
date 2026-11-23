#pragma once

#include "amr/refinement_input.h"
#include "amr/refiner.h"
#include "mesh_properties.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Top-level mesh adaptation interface. A MeshAdapter represents a complete mesh adaptation
// method (e.g. error-based estimate -> mark -> refine, or a future metric-based remesher)
// and owns its own stopping rules and per-run state.
//
// A configured MeshAdapter is immutable and reusable. Call start() to open a Session for
// one adaptive sequence; the Session holds any cross-cycle state (such as red-green
// RefinementHistory or the first cycle's error for relative tolerances) and releases it
// when destroyed. Adapters whose state must survive a process boundary (one CLI call per
// cycle) expose it as a value; see ErrorBasedAdapter::State.

struct StageInfo {
    std::string                   stage;    // e.g. "estimate", "mark", "refine"
    std::string                   method;   // e.g. "residual", "dorfler", "red_green"
    double                        time = 0; // seconds
    std::map<std::string, double> metrics;  // stage numbers for logging / diagnostics
    std::vector<std::string>      warnings;
};

enum class AdaptationStatus {
    adapted,       // produced a new mesh
    converged,     // the adapter's own stopping criterion was met (e.g. error tolerance)
    limit_reached, // the adapter's own size budget was reached (e.g. max_elements)
    unchanged,     // nothing to change (e.g. no elements marked)
};

// Adapter-specific per-cycle data carried in AdaptationResult::extras. Each adapter fills its
// own alternative (or leaves std::monostate when it produces none), keeping the top-level
// AdaptationResult fields strictly common across all adapters.
struct ErrorBasedExtras {
    // Per tet of the INPUT mesh (indexed like mesh.elements): the local error indicator eta_K
    // the decision was based on, estimated before any refinement. Info::error_estimate is its norm.
    std::vector<double> eta;
};

using AdaptationExtras = std::variant<std::monostate, ErrorBasedExtras>;

struct AdaptationResult {
    struct Info {
        std::string              adapter;          // the adapter's name()
        double                   time = 0;         // total, in seconds
        uint32_t                 elements_before = 0;
        uint32_t                 elements_after = 0;
        std::optional<double>    error_estimate;   // filled when the adapter estimates error
        std::vector<std::string> warnings;         // adapter-level warnings
        std::vector<StageInfo>   stages;           // per-stage breakdown, in execution order
    };

    AdaptationStatus              status = AdaptationStatus::unchanged;
    std::optional<MeshProperties> mesh;            // populated iff status == adapted
    std::optional<Parents>        parents;         // populated when the method tracks parents
    AdaptationExtras              extras;          // adapter-specific per-cycle outputs, if any
    Info                          info;
};

class MeshAdapter {
public:
    // Per-run adaptation state. adapt() is not virtual: it times the call, fills the shared
    // Info fields (adapter, elements_before, elements_after), checks postconditions on the
    // status/mesh pair, and delegates to adapt_step().
    //
    // Sequencing: each call in a session after an `adapted` result must pass the mesh
    // returned by the previous call. Call MeshAdapter::start() to begin a fresh session on
    // a different mesh.
    class Session {
    public:
        virtual ~Session() = default;
        AdaptationResult adapt(const MeshProperties& mesh, const RefinementInput& input);

    protected:
        explicit Session(std::string adapter_name) : adapter_name_(std::move(adapter_name)) {}

    private:
        virtual AdaptationResult adapt_step(const MeshProperties& mesh, const RefinementInput& input) = 0;
        std::string adapter_name_;
    };

    virtual ~MeshAdapter() = default;
    virtual std::string              name() const = 0;
    virtual std::unique_ptr<Session> start() const = 0;
};
