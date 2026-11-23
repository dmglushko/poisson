#include "verification/case_inputs.h"

#include "case_support.h"
#include "verification_mesh.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

using namespace verification_detail;

namespace {

using Clock = std::chrono::steady_clock;

void put_u32(std::string& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

}   // namespace

CaseInputsResult make_case_inputs(const VerificationCase& c) {
    const auto start = Clock::now();
    const VerificationProblem& problem = c.problem();
    CaseInputsResult result;
    result.geometry = problem.geometry;

    std::ostringstream yaml;
    for (const auto& m : problem.materials) {
        yaml << "---\nname: " << m.name << "\n";
        if (m.is_metal) {
            yaml << "band_structure:\n    model: metal\n";
        } else {
            yaml << "dielectric:\n    epsilon_r: " << number(m.eps_r) << "\n";
            if (m.screening_length > 0) yaml << "    screening_length: " << number(m.screening_length) << " nm\n";
        }
    }
    result.materials = yaml.str();

    for (const auto& charge : problem.charges) {
        for (const double x : charge.position) put_u32(result.charges, std::bit_cast<uint32_t>(static_cast<float>(x)));
        put_u32(result.charges, static_cast<uint32_t>(charge.value));
    }

    result.info.case_name = std::string(c.name());
    result.info.materials = static_cast<uint32_t>(problem.materials.size());
    result.info.charges = static_cast<uint32_t>(problem.charges.size());
    result.info.default_exclusion_radius = problem.default_exclusion_radius;
    result.info.boundary_error_bound = problem.boundary_error_bound;
    if (problem.boundary_error_bound > 0)
        result.info.warnings.push_back("some boundary conditions only approximate the exact solution, by at most " +
                                       number(problem.boundary_error_bound) + " V");
    result.info.time = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

ExactInterpolationResult interpolate_exact(const MeshProperties& mesh, const VerificationCase& c) {
    const auto start = Clock::now();
    const std::vector<std::size_t> material = case_materials(mesh, c);
    ExactInterpolationResult result;
    result.u.assign(mesh.nodes.size(), 0);

    // Each node takes the material of one of its tets, a metal one if any.
    constexpr std::size_t none = static_cast<std::size_t>(-1);
    std::vector<std::size_t> node_material(mesh.nodes.size(), none);
    for (const auto& element : mesh.elements) {
        const std::size_t m = material[element.material];
        for (const uint32_t n : element.node)
            if (node_material[n] == none || c.problem().materials[m].is_metal) node_material[n] = m;
    }

    // A node closer to a charge than a round-off distance counts as lying on it.
    std::array<double, 3> low{0, 0, 0}, high{0, 0, 0};
    if (!mesh.nodes.empty()) low = high = mesh.nodes.front();
    for (const auto& p : mesh.nodes)
        for (int k = 0; k < 3; ++k) {
            low[k] = std::min(low[k], p[k]);
            high[k] = std::max(high[k], p[k]);
        }
    const double on_charge = 1e-9 * distance(low, high);

    uint64_t singular = 0, unused = 0;
    for (std::size_t n = 0; n < mesh.nodes.size(); ++n) {
        if (node_material[n] == none) {
            ++unused;
            continue;
        }
        bool at_charge = false;
        for (const auto& charge : c.problem().charges) at_charge = at_charge || distance(mesh.nodes[n], charge.position) <= on_charge;
        const double u = at_charge ? 0 : c.u(mesh.nodes[n], node_material[n]);
        if (std::isfinite(u) && !at_charge) result.u[n] = u;
        else ++singular;
    }
    if (singular > 0)
        result.info.warnings.push_back(std::to_string(singular) + " nodes lie on a trapped charge, where u is infinite; set to 0");
    if (unused > 0) result.info.warnings.push_back(std::to_string(unused) + " nodes belong to no tet; set to 0");
    result.info.time = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}
