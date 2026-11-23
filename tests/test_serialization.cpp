#include "assembled_problem.h"
#include "blake3/blake3_hash.h"
#include "free_charges.h"
#include "importer/importer.h"
#include "io/amr_io.h"
#include "io/assembled_problem_io.h"
#include "io/free_charges_io.h"
#include "io/mesh_io.h"
#include "io/solver_result_io.h"
#include "mesh_properties.h"
#include "solver_result.h"

#include <hdf5.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAILED line " << __LINE__ << ": " #condition "\n"; \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

void check_throws(int line, const std::function<void()>& call, const std::string& expected) {
    try {
        call();
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            std::cerr << "FAILED line " << line << ": expected '" << expected << "' in: " << error.what() << "\n";
            ++failures;
        }
        return;
    }
    std::cerr << "FAILED line " << line << ": no exception, expected '" << expected << "'\n";
    ++failures;
}
#define CHECK_THROWS(call, expected) check_throws(__LINE__, [&] { call; }, expected)

std::string read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> blake3_pattern(std::size_t len) {
    std::vector<uint8_t> data(len);
    for (std::size_t i = 0; i < len; ++i) data[i] = static_cast<uint8_t>(i % 251);
    return data;
}

void test_blake3(const fs::path& dir) {
    // Official BLAKE3 v1.8.7 test vectors (input[i] = i % 251).
    struct Vector {
        std::size_t len;
        const char* hex;
    };
    const Vector vectors[] = {
        {0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
        {1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
        {64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
        {1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
        {1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"},
        {102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"},
    };

    for (const auto& v : vectors) {
        const auto data = blake3_pattern(v.len);
        CHECK(poisson::blake3_bytes_hex(data.data(), data.size()) == v.hex);

        const fs::path file = dir / ("b3_" + std::to_string(v.len) + ".bin");
        std::ofstream out(file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        out.close();
        CHECK(poisson::blake3_file_hex(file) == v.hex);
    }

    CHECK_THROWS(poisson::blake3_file_hex(dir / "does_not_exist.bin"), "failed to hash file");
}

MeshProperties make_sample_mesh() {
    MeshProperties mesh;
    mesh.nodes = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
    mesh.elements = {
        {{0, 1, 2, 3}, 0},
        {{1, 0, 2, 4}, 1},
    };
    mesh.boundary_elements = {
        {{0, 1, 3}, 0},
        {{0, 2, 4}, 1},
        {{1, 2, 3}, 2},
    };
    mesh.boundary_conditions = {
        {MeshProperties::BcType::Dirichlet, 0.0, std::nullopt},
        {MeshProperties::BcType::Neumann, 1.5, std::nullopt},
        {MeshProperties::BcType::Robin, -2.25, 0.75},
    };
    mesh.materials = {
        {"siliconnitride", 7.5, 2.0, false, std::nullopt},
        {"detector", 1.0, 0.0, true, 10.0},
    };
    mesh.charges = {
        {{0.1, 0.2, 0.3}, 1},
        {{0.1, 0.2, -0.5}, -3},
    };
    mesh.screened = true;
    mesh.neighbours = {
        {std::nullopt, std::nullopt, std::nullopt, 1u},
        {std::nullopt, std::nullopt, std::nullopt, 0u},
    };
    return mesh;
}

void test_mesh_roundtrip_and_determinism(const fs::path& dir) {
    const MeshProperties mesh = make_sample_mesh();
    const fs::path file_a = dir / "mesh_a.h5";
    const fs::path file_b = dir / "mesh_b.h5";

    poisson::io::write_mesh_properties(file_a, mesh);
    // Cross a full wall-clock second boundary to prove HDF5 timestamps are disabled.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    poisson::io::write_mesh_properties(file_b, mesh);

    const std::string hash_a = poisson::blake3_file_hex(file_a);
    const std::string hash_b = poisson::blake3_file_hex(file_b);
    CHECK(hash_a == hash_b);
    CHECK(read_bytes(file_a) == read_bytes(file_b));

    const MeshProperties loaded = poisson::io::read_mesh_properties(file_a);
    CHECK(loaded.nodes == mesh.nodes);
    CHECK(loaded.elements.size() == mesh.elements.size());
    CHECK(loaded.elements[0].node == mesh.elements[0].node && loaded.elements[0].material == 0);
    CHECK(loaded.elements[1].node == mesh.elements[1].node && loaded.elements[1].material == 1);
    CHECK(loaded.boundary_elements.size() == 3);
    CHECK(loaded.boundary_elements[2].node == mesh.boundary_elements[2].node);
    CHECK(loaded.boundary_elements[2].boundary_condition == 2);
    CHECK(loaded.boundary_conditions.size() == 3);
    CHECK(loaded.boundary_conditions[0].type == MeshProperties::BcType::Dirichlet &&
          !loaded.boundary_conditions[0].alpha);
    CHECK(loaded.boundary_conditions[2].type == MeshProperties::BcType::Robin &&
          loaded.boundary_conditions[2].value == -2.25 && loaded.boundary_conditions[2].alpha == 0.75);
    CHECK(loaded.materials.size() == 2);
    CHECK(loaded.materials[0].name == "siliconnitride" && loaded.materials[0].eps_r == 7.5 &&
          loaded.materials[0].screening_length == 2.0 && !loaded.materials[0].is_metal &&
          !loaded.materials[0].voltage);
    CHECK(loaded.materials[1].name == "detector" && loaded.materials[1].is_metal &&
          loaded.materials[1].voltage == 10.0);
    CHECK(loaded.charges.size() == 2);
    CHECK(loaded.charges[1].position == mesh.charges[1].position && loaded.charges[1].value == -3);
    CHECK(loaded.screened == mesh.screened);
    CHECK(loaded.neighbours == mesh.neighbours);

    // Negative zero (-0.0) normalizes to +0.0 so hashes match byte-for-byte.
    MeshProperties neg_zero = mesh;
    neg_zero.nodes[0] = {-0.0, -0.0, -0.0};
    neg_zero.boundary_conditions[0].value = -0.0;
    const fs::path file_nz = dir / "mesh_nz.h5";
    poisson::io::write_mesh_properties(file_nz, neg_zero);
    CHECK(poisson::blake3_file_hex(file_a) == poisson::blake3_file_hex(file_nz));

    // Empty vectors edge case.
    MeshProperties empty;
    const fs::path empty_file = dir / "mesh_empty.h5";
    poisson::io::write_mesh_properties(empty_file, empty);
    const MeshProperties empty_loaded = poisson::io::read_mesh_properties(empty_file);
    CHECK(empty_loaded.nodes.empty() && empty_loaded.elements.empty() && empty_loaded.materials.empty() &&
          empty_loaded.charges.empty() && !empty_loaded.screened && empty_loaded.neighbours.empty());
}

void test_other_data_classes(const fs::path& dir) {
    // FreeCharges
    FreeCharges fc;
    fc.electrons = {
        {{1.0, 2.0, 35.0}, {0.0, 0.0, -1.0}, 500.0, {3, 4}},
        {{1.5, 2.5, 36.0}, {0.1, -0.2, -2.0}, 300.0, {5, 6}},
    };
    const fs::path fc_file = dir / "fc.h5";
    poisson::io::write_free_charges(fc_file, fc);
    const FreeCharges fc_loaded = poisson::io::read_free_charges(fc_file);
    CHECK(fc_loaded.electrons.size() == 2);
    CHECK(fc_loaded.electrons[0].position == fc.electrons[0].position);
    CHECK(fc_loaded.electrons[1].direction == fc.electrons[1].direction);
    CHECK(fc_loaded.electrons[1].energy == 300.0);
    CHECK(fc_loaded.electrons[1].pixel == fc.electrons[1].pixel);

    // Schema mismatch check.
    CHECK_THROWS(poisson::io::read_mesh_properties(fc_file), "expected schema 'MeshProperties'");

    // AssembledProblem with real Eigen::SparseMatrix<double>
    AssembledProblem ap;
    ap.A.resize(3, 3);
    std::vector<Eigen::Triplet<double>> triplets = {
        {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.5}, {2, 2, 4.0},
    };
    ap.A.setFromTriplets(triplets.begin(), triplets.end());
    ap.f = {1.0, -2.5, 3.25};
    ap.row = {std::nullopt, 0u, 1u, 1u, 2u, std::nullopt};
    ap.known = {{0u, 0.0}, {5u, 5.0}};
    const fs::path ap_file = dir / "ap.h5";
    poisson::io::write_assembled_problem(ap_file, ap);
    const AssembledProblem ap_loaded = poisson::io::read_assembled_problem(ap_file);
    CHECK(ap_loaded.A.rows() == 3 && ap_loaded.A.cols() == 3 && ap_loaded.A.nonZeros() == 5);
    CHECK(ap_loaded.A.coeff(0, 0) == 2.0 && ap_loaded.A.coeff(0, 1) == -1.0 && ap_loaded.A.coeff(1, 1) == 2.5 &&
          ap_loaded.A.coeff(2, 2) == 4.0);
    CHECK(ap_loaded.f == ap.f);
    CHECK(ap_loaded.row == ap.row);
    CHECK(ap_loaded.known == ap.known);

    // Uncompressed sparse matrix produces byte-identical HDF5 to compressed one.
    AssembledProblem ap_uncompressed = ap;
    ap_uncompressed.A.uncompress();
    const fs::path ap_file_b = dir / "ap_b.h5";
    poisson::io::write_assembled_problem(ap_file_b, ap_uncompressed);
    CHECK(poisson::blake3_file_hex(ap_file) == poisson::blake3_file_hex(ap_file_b));

    // SolverResult
    SolverResult sr;
    sr.x = {0.25, 1.5, -4.75};
    const fs::path sr_file = dir / "sr.h5";
    poisson::io::write_solver_result(sr_file, sr);
    const SolverResult sr_loaded = poisson::io::read_solver_result(sr_file);
    CHECK(sr_loaded.x == sr.x);

    // Empty FreeCharges / AssembledProblem / SolverResult
    poisson::io::write_free_charges(dir / "fc_empty.h5", {});
    CHECK(poisson::io::read_free_charges(dir / "fc_empty.h5").electrons.empty());
    poisson::io::write_assembled_problem(dir / "ap_empty.h5", {});
    const AssembledProblem ap_empty_loaded = poisson::io::read_assembled_problem(dir / "ap_empty.h5");
    CHECK(ap_empty_loaded.f.empty() && ap_empty_loaded.A.rows() == 0 && ap_empty_loaded.A.nonZeros() == 0);
    poisson::io::write_solver_result(dir / "sr_empty.h5", {});
    CHECK(poisson::io::read_solver_result(dir / "sr_empty.h5").x.empty());

    // NodalField (u + point_sources)
    poisson::io::NodalField nf;
    nf.u = {0.0, 1.25, -2.5, 3.0};
    nf.point_sources = {{{0.1, 0.2, 0.3}, -1.0, 0u}, {{0.4, 0.5, 0.6}, 2.0, std::nullopt}};
    const fs::path nf_file = dir / "nf.h5";
    poisson::io::write_nodal_field(nf_file, nf);
    const auto nf_loaded = poisson::io::read_nodal_field(nf_file);
    CHECK(nf_loaded.u == nf.u && nf_loaded.point_sources.size() == 2);
    CHECK(nf_loaded.point_sources[0].element == 0u && !nf_loaded.point_sources[1].element);

    // Parents
    Parents parents;
    parents.elements = {0u, 0u, 1u};
    parents.nodes = {{0u, 0u}, {1u, 1u}, {0u, 1u}};
    const fs::path par_file = dir / "parents.h5";
    poisson::io::write_parents(par_file, parents);
    const auto par_loaded = poisson::io::read_parents(par_file);
    CHECK(par_loaded.elements == parents.elements && par_loaded.nodes == parents.nodes);

    // RefinementHistory
    RefinementHistory hist;
    hist.families = {{{0, 1, 2, 3}, {4u, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 5u}}};
    hist.family = {0u, std::nullopt, 0u};
    const fs::path hist_file = dir / "hist.h5";
    poisson::io::write_refinement_history(hist_file, hist);
    const auto hist_loaded = poisson::io::read_refinement_history(hist_file);
    CHECK(hist_loaded.families.size() == 1 && hist_loaded.families[0].parent == hist.families[0].parent);
    CHECK(hist_loaded.families[0].midpoint == hist.families[0].midpoint && hist_loaded.family == hist.family);

    // ErrorBasedAdapter::State: history + optional initial_error
    const ErrorBasedAdapter::State state{hist, 0.125};
    const fs::path state_file = dir / "state.h5";
    poisson::io::write_adaptation_state(state_file, state);
    const auto state_loaded = poisson::io::read_adaptation_state(state_file);
    CHECK(state_loaded.initial_error == 0.125);
    CHECK(state_loaded.history.families.size() == 1 && state_loaded.history.family == hist.family);
    CHECK(state_loaded.history.families[0].midpoint == hist.families[0].midpoint);
    const fs::path state_again = dir / "state_again.h5";
    poisson::io::write_adaptation_state(state_again, state_loaded);
    CHECK(read_bytes(state_file) == read_bytes(state_again));   // deterministic bytes

    const fs::path empty_state_file = dir / "empty_state.h5";
    poisson::io::write_adaptation_state(empty_state_file, ErrorBasedAdapter::State{});
    const auto empty_state = poisson::io::read_adaptation_state(empty_state_file);
    CHECK(!empty_state.initial_error && empty_state.history.families.empty() && empty_state.history.family.empty());
    CHECK_THROWS(poisson::io::read_refinement_history(state_file), "expected schema 'RefinementHistory'");
    CHECK_THROWS(poisson::io::write_adaptation_state(dir / "bad_state.h5", ErrorBasedAdapter::State{{}, -1.0}),
                 "initial_error");

    // Error indicators (eta), including the empty case
    const std::vector<double> eta = {0.5, 0.0, 1.25};
    const fs::path eta_file = dir / "eta.h5";
    poisson::io::write_error_indicators(eta_file, eta);
    CHECK(poisson::io::read_error_indicators(eta_file) == eta);
    poisson::io::write_error_indicators(dir / "eta_empty.h5", {});
    CHECK(poisson::io::read_error_indicators(dir / "eta_empty.h5").empty());
}

void test_info_json(const fs::path& dir) {
    const MeshImportResult::Info info{0.42, 0.15, 0.20, {"warning with \"quotes\" and \\backslash\\"}};
    const fs::path mesh_info = dir / "mesh_info.json";
    poisson::io::write_info_json(mesh_info, info);
    const std::string text = read_bytes(mesh_info);
    CHECK(text.find("\"MeshImportResult::Info\"") != std::string::npos);
    CHECK(text.find("\\\"quotes\\\"") != std::string::npos);
    CHECK(text.find("\\\\backslash\\\\") != std::string::npos);

    poisson::io::write_info_json(dir / "fc_info.json", FreeChargesImportResult::Info{0.1, {}});
    poisson::io::write_info_json(dir / "ap_info.json", AssemblyResult::Info{1.2, 0.8, 0.3, 4200, 12, {}});
    CHECK(read_bytes(dir / "ap_info.json").find("\"AssemblyResult::Info\"") != std::string::npos);
    poisson::io::write_info_json(dir / "sr_info.json", SolverResult::Info{2.5, 42, 1e-10, {"slow"}});
    CHECK(read_bytes(dir / "sr_info.json").find("\"iterations\": 42") != std::string::npos);
}

void test_malformed_h5_rejected(const fs::path& dir) {
    // 1. Non-scalar array-valued schema attribute must be rejected (no buffer overrun).
    {
        const fs::path bad = dir / "array_schema.h5";
        const hid_t h5 = H5Fcreate(bad.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        const hsize_t dim = 3;
        const hid_t space = H5Screate_simple(1, &dim, nullptr);
        const hid_t stype = H5Tcopy(H5T_C_S1);
        H5Tset_size(stype, 16);
        const hid_t attr = H5Acreate2(h5, "schema", stype, space, H5P_DEFAULT, H5P_DEFAULT);
        const char data[3][16] = {"SolverResult", "SolverResult", "SolverResult"};
        H5Awrite(attr, stype, data);
        H5Aclose(attr);
        H5Tclose(stype);
        H5Sclose(space);
        H5Fclose(h5);
        CHECK_THROWS(poisson::io::read_solver_result(bad), "must be a scalar");
    }

    // 2. Non-scalar or wrong-typed schema_version attribute must be rejected.
    {
        const fs::path bad = dir / "bad_version.h5";
        poisson::io::write_solver_result(bad, SolverResult{{1.0}, {}});
        const hid_t h5 = H5Fopen(bad.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        H5Adelete(h5, "schema_version");
        const hsize_t dim = 2;
        const hid_t space = H5Screate_simple(1, &dim, nullptr);
        const hid_t attr = H5Acreate2(h5, "schema_version", H5T_STD_U32LE, space, H5P_DEFAULT, H5P_DEFAULT);
        const uint32_t vals[2] = {1, 2};
        H5Awrite(attr, H5T_NATIVE_UINT32, vals);
        H5Aclose(attr);
        H5Sclose(space);
        H5Fclose(h5);
        CHECK_THROWS(poisson::io::read_solver_result(bad), "must be a scalar");
    }

    // 3. Wrong numeric dataset type (float32 instead of float64) must be rejected.
    {
        const fs::path bad = dir / "bad_dtype.h5";
        poisson::io::write_solver_result(bad, SolverResult{{1.0}, {}});
        const hid_t h5 = H5Fopen(bad.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        H5Ldelete(h5, "x", H5P_DEFAULT);
        const hsize_t dim = 1;
        const hid_t space = H5Screate_simple(1, &dim, nullptr);
        const hid_t dset = H5Dcreate2(h5, "x", H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const float v = 1.0f;
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(dset);
        H5Sclose(space);
        H5Fclose(h5);
        CHECK_THROWS(poisson::io::read_solver_result(bad), "unexpected datatype");
    }

    // 4. Out-of-range element node index in MeshProperties must be rejected.
    {
        MeshProperties broken = make_sample_mesh();
        broken.elements[0].node[0] = 999;
        const fs::path bad = dir / "dangling_node.h5";
        poisson::io::write_mesh_properties(bad, broken);
        CHECK_THROWS(poisson::io::read_mesh_properties(bad), "out-of-range node index");
    }

    // 5. Out-of-range row index in AssembledProblem must be rejected.
    {
        AssembledProblem broken;
        broken.A.resize(1, 1);
        broken.f = {1.0};
        broken.row = {5u};
        const fs::path bad = dir / "dangling_row.h5";
        poisson::io::write_assembled_problem(bad, broken);
        CHECK_THROWS(poisson::io::read_assembled_problem(bad), "out-of-range row index");
    }

    // 6. Empty A with non-empty f must be rejected.
    {
        AssembledProblem broken;
        broken.f = {1.0};
        const fs::path bad = dir / "zero_rows_nonempty_f.h5";
        poisson::io::write_assembled_problem(bad, broken);
        CHECK_THROWS(poisson::io::read_assembled_problem(bad), "A.rows() does not match f.size()");
    }
}

}   // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "poisson_test_serialization";
    fs::remove_all(dir);
    fs::create_directories(dir);

    try {
        test_blake3(dir);
        test_mesh_roundtrip_and_determinism(dir);
        test_other_data_classes(dir);
        test_info_json(dir);
        test_malformed_h5_rejected(dir);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << "\n";
        ++failures;
    }

    fs::remove_all(dir);
    std::cout << (failures == 0 ? "all serialization tests passed" : "TESTS FAILED") << "\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
