#include "verification/case_inputs.h"

#include "case_support.h"

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
