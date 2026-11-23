#pragma once

#include <Eigen/SparseCore>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Holds the matrix A and the right-hand side vector f of A x = f. Pure problem data: how
// the run that built it went is in AssemblyResult::Info, not here.
//
// A has one row per unknown, not one per node. `row` says which row belongs to each node;
// several nodes may share a row. Nodes whose value is known in advance (Dirichlet) have no
// row and are listed in `known`. The potential at every node is then
//     u[i] = x[*row[i]]             for a node with a row
//     u[node] = value               for every (node, value) in known
struct AssembledProblem {
    Eigen::SparseMatrix<double>              A;
    std::vector<double>                      f;       // right-hand side, one entry per row of A
    std::vector<std::optional<uint32_t>>     row;     // per node: its row in A, or empty if its value is known
    std::vector<std::pair<uint32_t, double>> known;   // (node, value) for every node whose value is known
};

// What Assembly returns: the problem it built and how the run went.
struct AssemblyResult {
    struct Info {
        double   time = 0;                      // total, in seconds
        double   time_assemble_A = 0;
        double   time_assemble_f = 0;
        uint64_t nnz = 0;                       // nonzeros in A
        uint32_t n_dirichlet = 0;               // eliminated nodes
        std::vector<std::string> warnings;
    };

    AssembledProblem problem;
    Info             info;
};
