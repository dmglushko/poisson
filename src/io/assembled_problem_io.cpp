#include "h5_common.h"
#include "io/assembled_problem_io.h"

#include <limits>
#include <optional>
#include <sstream>

namespace poisson::io {

using namespace detail;

void write_assembled_problem(const fs::path& file, const AssembledProblem& problem) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "AssembledProblem");

    {
        const H5Handle g_a = create_group(file, h5.id, "A");
        std::optional<Eigen::SparseMatrix<double>> owned_compressed;
        const Eigen::SparseMatrix<double>* mat = &problem.A;
        if (!problem.A.isCompressed()) {
            owned_compressed.emplace(problem.A);
            owned_compressed->makeCompressed();
            mat = &*owned_compressed;
        }

        const uint64_t rows = static_cast<uint64_t>(mat->rows());
        const uint64_t cols = static_cast<uint64_t>(mat->cols());
        const std::size_t nnz = static_cast<std::size_t>(mat->nonZeros());

        write_string_attr(file, g_a.id, "storage_order", "col_major");
        write_u64_attr(file, g_a.id, "rows", rows);
        write_u64_attr(file, g_a.id, "cols", cols);

        std::vector<int32_t> outer_index(cols + 1, 0);
        if (cols > 0) {
            const auto* ptr = mat->outerIndexPtr();
            for (uint64_t j = 0; j <= cols; ++j) outer_index[j] = static_cast<int32_t>(ptr[j]);
        }
        std::vector<int32_t> inner_index(nnz);
        std::vector<double> values(nnz);
        if (nnz > 0) {
            const auto* iptr = mat->innerIndexPtr();
            const double* vptr = mat->valuePtr();
            for (std::size_t k = 0; k < nnz; ++k) {
                inner_index[k] = static_cast<int32_t>(iptr[k]);
                values[k] = vptr[k];
            }
        }
        write_dataset_1d(file, g_a.id, "outer_index", outer_index);
        write_dataset_1d(file, g_a.id, "inner_index", inner_index);
        write_dataset_1d(file, g_a.id, "values", values);
    }

    write_dataset_1d(file, h5.id, "f", problem.f);

    std::vector<uint32_t> row_raw;
    row_raw.reserve(problem.row.size());
    for (const auto& r : problem.row) {
        if (r && *r == kNoneU32) fail(file, "row index collides with sentinel UINT32_MAX");
        row_raw.push_back(r.value_or(kNoneU32));
    }
    write_dataset_1d(file, h5.id, "row", row_raw);

    {
        const H5Handle g_known = create_group(file, h5.id, "known");
        std::vector<uint32_t> known_node;
        std::vector<double> known_value;
        known_node.reserve(problem.known.size());
        known_value.reserve(problem.known.size());
        for (const auto& [node, value] : problem.known) {
            known_node.push_back(node);
            known_value.push_back(value);
        }
        write_dataset_1d(file, g_known.id, "node", known_node);
        write_dataset_1d(file, g_known.id, "value", known_value);
    }

    h5.close_file_checked(file);
}

AssembledProblem read_assembled_problem(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "AssembledProblem");

    AssembledProblem problem;
    {
        const H5Handle g_a = open_group(file, h5.id, "A");
        const std::string order = read_string_attr(file, g_a.id, "storage_order");
        if (order != "col_major")
            fail(file, "unsupported A/storage_order '" + order + "', expected 'col_major'");
        const uint64_t rows = read_u64_attr(file, g_a.id, "rows");
        const uint64_t cols = read_u64_attr(file, g_a.id, "cols");
        if (rows > static_cast<uint64_t>(std::numeric_limits<Eigen::SparseMatrix<double>::StorageIndex>::max()) ||
            cols > static_cast<uint64_t>(std::numeric_limits<Eigen::SparseMatrix<double>::StorageIndex>::max()))
            fail(file, "A dimensions exceed Eigen StorageIndex range");

        const auto outer_index = read_dataset_1d<int32_t>(file, g_a.id, "outer_index");
        const auto inner_index = read_dataset_1d<int32_t>(file, g_a.id, "inner_index");
        const auto values = read_dataset_1d<double>(file, g_a.id, "values");
        if (outer_index.size() != cols + 1)
            fail(file, "A/outer_index length must equal cols + 1");
        if (inner_index.size() != values.size())
            fail(file, "A/inner_index and A/values length mismatch");
        if (outer_index.front() != 0 || static_cast<uint64_t>(outer_index.back()) != inner_index.size())
            fail(file, "A/outer_index must start at 0 and end at nnz");

        for (uint64_t j = 0; j < cols; ++j) {
            if (outer_index[j] < 0 || outer_index[j] > outer_index[j + 1])
                fail(file, "A/outer_index is not non-decreasing");
            int32_t prev_row = -1;
            for (int32_t k = outer_index[j]; k < outer_index[j + 1]; ++k) {
                const int32_t r = inner_index[static_cast<std::size_t>(k)];
                if (r < 0 || static_cast<uint64_t>(r) >= rows)
                    fail(file, "A/inner_index references out-of-range row " + std::to_string(r));
                if (r <= prev_row)
                    fail(file, "A/inner_index must be strictly increasing within each column");
                prev_row = r;
            }
        }

        if (rows > 0 || cols > 0) {
            if (inner_index.empty()) {
                problem.A.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
                problem.A.makeCompressed();
            } else {
                const Eigen::Map<const Eigen::SparseMatrix<double>> mapped(
                    static_cast<Eigen::Index>(rows),
                    static_cast<Eigen::Index>(cols),
                    static_cast<Eigen::Index>(inner_index.size()),
                    outer_index.data(),
                    inner_index.data(),
                    values.data());
                problem.A = mapped;
            }
        }
    }

    problem.f = read_dataset_1d<double>(file, h5.id, "f");
    if (static_cast<std::size_t>(problem.A.rows()) != problem.f.size())
        fail(file, "A.rows() does not match f.size()");

    const auto row_raw = read_dataset_1d<uint32_t>(file, h5.id, "row");
    problem.row.resize(row_raw.size());
    for (std::size_t i = 0; i < row_raw.size(); ++i) {
        if (row_raw[i] == kNoneU32) {
            problem.row[i] = std::nullopt;
        } else if (row_raw[i] >= problem.f.size()) {
            fail(file, "row references out-of-range row index " + std::to_string(row_raw[i]));
        } else {
            problem.row[i] = row_raw[i];
        }
    }

    const H5Handle g_known = open_group(file, h5.id, "known");
    const auto known_node = read_dataset_1d<uint32_t>(file, g_known.id, "node");
    const auto known_value = read_dataset_1d<double>(file, g_known.id, "value");
    if (known_node.size() != known_value.size()) fail(file, "known dataset size mismatch");
    problem.known.resize(known_node.size());
    for (std::size_t i = 0; i < known_node.size(); ++i) {
        if (!problem.row.empty() && known_node[i] >= problem.row.size())
            fail(file, "known/node references out-of-range node index " + std::to_string(known_node[i]));
        problem.known[i] = {known_node[i], known_value[i]};
    }

    return problem;
}

void write_info_json(const fs::path& file, const AssemblyResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"AssemblyResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"time_assemble_A\": " << format_double(info.time_assemble_A) << ",\n"
        << "  \"time_assemble_f\": " << format_double(info.time_assemble_f) << ",\n"
        << "  \"nnz\": " << info.nnz << ",\n"
        << "  \"n_dirichlet\": " << info.n_dirichlet << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
