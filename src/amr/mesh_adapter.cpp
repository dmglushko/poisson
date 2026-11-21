// Shared part of every MeshAdapter::Session: timing, common Info fields, and
// status/mesh postcondition checks.

#include "amr/mesh_adapter.h"

#include <chrono>
#include <stdexcept>

AdaptationResult MeshAdapter::Session::adapt(const MeshProperties& mesh, const RefinementInput& input) {
    const auto start = std::chrono::steady_clock::now();
    AdaptationResult result = adapt_step(mesh, input);

    const bool adapted = result.status == AdaptationStatus::adapted;
    if (adapted != result.mesh.has_value())
        throw std::logic_error(adapter_name_ + " adapter: status and mesh presence disagree");

    result.info.adapter = adapter_name_;
    result.info.elements_before = static_cast<uint32_t>(mesh.elements.size());
    result.info.elements_after =
        static_cast<uint32_t>(adapted ? result.mesh->elements.size() : mesh.elements.size());
    result.info.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
