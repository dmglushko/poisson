#pragma once

// PLACEHOLDER: top-level solver driver.
//
// Cannot be implemented yet because the contracts for AnalyticalMaker / AnalyticalSolution,
// Assembly, Solver, and SolutionBuilder / Solution are not defined yet (see architecture.md).
// This header anchors how those steps and the optional MeshAdapter will fit together.

#include "amr/mesh_adapter.h"
#include "assembled_problem.h"
#include "mesh_properties.h"
#include "solver_result.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class DriverStopReason {
    single_solve,      // no MeshAdapter was configured
    converged,         // adapter returned AdaptationStatus::converged
    adapter_limit,     // adapter returned AdaptationStatus::limit_reached
    unchanged,         // adapter returned AdaptationStatus::unchanged
    max_cycles,        // driver cycle budget reached
    max_nodes,         // next adapted mesh would exceed driver node budget (not solved)
};

// Planned top-level driver that orchestrates the full Poisson solve pipeline:
//
//   Dependencies (to be injected via constructor once their interfaces exist):
//     - AnalyticalMaker  (optional, e.g. P3M singularity subtraction around trapped charges)
//     - Assembler        (MeshProperties + AnalyticalSolution -> AssembledProblem)
//     - Solver           (AssembledProblem -> SolverResult)
//     - SolutionBuilder  (AssembledProblem + SolverResult + AnalyticalSolution -> Solution)
//     - MeshAdapter      (optional; nullptr => single solve with no adaptive loop)
//
//   Planned settings / budgets (driver-level, independent of the adaptation method):
//     - max_cycles       (at least 1)
//     - max_nodes        (optional cap; stop before solving a mesh beyond this)
//
//   Planned cycle loop:
//     auto session = adapter ? adapter->start() : nullptr;
//     for (uint32_t cycle = 0; cycle < max_cycles; ++cycle) {
//         // 1. Build AnalyticalSolution on current mesh (if enabled)
//         // 2. Assemble A x = f        -> AssembledProblem (with AssembledProblem::Info)
//         // 3. Solve A x = f           -> SolverResult     (with SolverResult::Info)
//         // 4. Build Solution          -> Solution + nodal u + RefinementInput
//         // 5. Record per-cycle metrics (assembly info, solver info, mesh size, ...)
//         // 6. If !session: stop (DriverStopReason::single_solve)
//         // 7. If cycle + 1 == max_cycles: stop (DriverStopReason::max_cycles)
//         // 8. AdaptationResult adapted = session->adapt(mesh, refinement_input);
//         //    Record adapted.info (stages, error_estimate, time, warnings).
//         // 9. If adapted.status != AdaptationStatus::adapted: stop accordingly.
//         // 10. If max_nodes && adapted.mesh->nodes.size() > *max_nodes:
//         //       discard adapted.mesh, stop (DriverStopReason::max_nodes).
//         // 11. mesh = std::move(*adapted.mesh);  // optionally prolongate u via adapted.parents
//     }
//
//   Planned result:
//     - final MeshProperties & Solution
//     - DriverStopReason stop_reason
//     - per-cycle info (assembly, solver, adaptation stages) and total Info {time, warnings}
class PoissonDriver {
public:
    struct Settings {
        uint32_t                max_cycles = 1;
        std::optional<uint64_t> max_nodes = std::nullopt;
    };

    // Placeholder constructors taking only the already-defined optional adapter and settings.
    // AnalyticalMaker, Assembler, Solver, and SolutionBuilder will be added once defined.
    explicit PoissonDriver(std::shared_ptr<const MeshAdapter> adapter = nullptr)
        : adapter_(std::move(adapter)), settings_{} {}
    PoissonDriver(std::shared_ptr<const MeshAdapter> adapter, Settings settings)
        : adapter_(std::move(adapter)), settings_(settings) {}

    const Settings&                   settings() const { return settings_; }
    const std::shared_ptr<const MeshAdapter>& adapter() const { return adapter_; }

private:
    std::shared_ptr<const MeshAdapter> adapter_;
    Settings                           settings_;
};
