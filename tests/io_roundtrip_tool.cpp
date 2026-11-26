// Test-only helper verifying C++ poisson::io readers/writers interoperate with Python
// HDF5 files produced by poisson_test_solver, and exposing C++ physical constants for parity checks.

#include "common/constants.h"
#include "io/amr_io.h"
#include "io/assembled_problem_io.h"
#include "io/mesh_io.h"
#include "io/solver_result_io.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: io_roundtrip_tool <mesh|assembled|solver|nodal> <in.h5> <out.h5>\n"
                      << "       io_roundtrip_tool constants\n";
            return EXIT_FAILURE;
        }
        const std::string_view cmd(argv[1]);
        if (cmd == "constants") {
            std::cout << std::setprecision(17)
                      << poisson::constants::e_over_eps0_nm << " "
                      << poisson::constants::coulomb_constant_nm << "\n";
            return EXIT_SUCCESS;
        }
        if (argc != 4) {
            std::cerr << "error: expected <in.h5> <out.h5>\n";
            return EXIT_FAILURE;
        }
        const std::filesystem::path in_path(argv[2]);
        const std::filesystem::path out_path(argv[3]);
        if (cmd == "mesh") {
            const MeshProperties m = poisson::io::read_mesh_properties(in_path);
            poisson::io::write_mesh_properties(out_path, m);
        } else if (cmd == "assembled") {
            const AssembledProblem p = poisson::io::read_assembled_problem(in_path);
            poisson::io::write_assembled_problem(out_path, p);
        } else if (cmd == "solver") {
            const SolverResult s = poisson::io::read_solver_result(in_path);
            poisson::io::write_solver_result(out_path, s);
        } else if (cmd == "nodal") {
            const poisson::io::NodalField f = poisson::io::read_nodal_field(in_path);
            poisson::io::write_nodal_field(out_path, f);
        } else {
            std::cerr << "error: unknown command '" << cmd << "'\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
