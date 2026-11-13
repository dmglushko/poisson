// Tests for the importer.
//
//   test_importer                              small generated files only
//   test_importer <msh> <yaml> <cha> <pri>     also import the holes_3x2 sample files

#include "assembled_problem.h"
#include "importer.h"
#include "solver_result.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::cerr << "FAILED line " << __LINE__ << ": " #condition "\n";          \
            ++failures;                                                               \
        }                                                                             \
    } while (false)

// The call must throw std::runtime_error with `expected` somewhere in its message.
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

bool has_warning(const std::vector<std::string>& warnings, const std::string& text) {
    for (const auto& warning : warnings)
        if (warning.find(text) != std::string::npos) return true;
    return false;
}

fs::path write_file(const fs::path& directory, const std::string& name, const std::string& content) {
    const fs::path file = directory / name;
    std::ofstream(file, std::ios::binary) << content;
    return file;
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos == std::string::npos) throw std::logic_error("test bug: '" + from + "' not found");
    return text.replace(pos, from.size(), to);
}

// Little-endian binary records.
void put_float(std::string& out, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>(bits >> (8 * i) & 0xff));
}
void put_int(std::string& out, int32_t value) {
    const auto bits = static_cast<uint32_t>(value);
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>(bits >> (8 * i) & 0xff));
}
std::string charge_record(float x, float y, float z, int32_t q) {
    std::string out;
    put_float(out, x), put_float(out, y), put_float(out, z), put_int(out, q);
    return out;
}
std::string primary_record(float z, float dz, float energy) {
    std::string out;
    for (float v : {1.0f, 2.0f, z, 0.0f, 0.0f, dz, energy}) put_float(out, v);
    put_int(out, 3), put_int(out, 4);
    return out;
}

// Two tets sharing the face (10, 20, 30). Node ids have gaps and the second tet is
// written with a negative volume.
const std::string small_mesh = R"($MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 7 "0V"
2 9 "D0"
3 5 "siliconnitride"
3 2 "vacuum"
$EndPhysicalNames
$Nodes
5
10 0 0 0
20 1 0 0
30 0 1 0
40 0 0 1
50 0 0 -1
$EndNodes
$Elements
4
1 2 2 7 1 10 20 40
2 2 2 9 1 10 30 50
3 4 2 5 1 10 20 30 40
4 4 2 2 2 10 20 30 50
$EndElements
)";

const std::string small_material = R"(name:           siliconnitride
density:        3.17 g/cm^3

dielectric:
    epsilon_r:  7.5   # relative permittivity

phonon:
    lattice:    7.60 A
)";

void test_small_files(const fs::path& dir) {
    const Importer importer;
    const fs::path mesh_file = write_file(dir, "small.msh", small_mesh);
    const fs::path material_file = write_file(dir, "siliconnitride.yaml", small_material);
    const fs::path charge_file =
        write_file(dir, "small.cha", charge_record(0.1f, 0.1f, 0.1f, 1) + charge_record(0.1f, 0.1f, -0.5f, -1));

    // ---- a valid mesh
    const MeshProperties mesh = importer.import_mesh(mesh_file, {material_file}, charge_file);
    CHECK(mesh.nodes.size() == 5);
    CHECK((mesh.nodes[3] == std::array<double, 3>{0, 0, 1}));
    CHECK(mesh.elements.size() == 2);
    CHECK((mesh.elements[0].node == std::array<uint32_t, 4>{0, 1, 2, 3}));
    CHECK((mesh.elements[1].node == std::array<uint32_t, 4>{1, 0, 2, 4}));   // reoriented
    CHECK(mesh.elements[0].material == 0 && mesh.elements[1].material == 1);
    const std::optional<uint32_t> none;
    CHECK(mesh.neighbours.size() == 2);
    CHECK((mesh.neighbours[0] == std::array<std::optional<uint32_t>, 4>{none, none, none, 1}));   // shared face is opposite node 3
    CHECK((mesh.neighbours[1] == std::array<std::optional<uint32_t>, 4>{none, none, none, 0}));
    CHECK(mesh.materials.size() == 2);
    CHECK(mesh.materials[0].name == "siliconnitride" && mesh.materials[0].eps_r == 7.5);
    CHECK(mesh.materials[1].name == "vacuum" && mesh.materials[1].eps_r == 1.0);
    CHECK(mesh.boundary_elements.size() == 2);
    CHECK((mesh.boundary_elements[0].node == std::array<uint32_t, 3>{0, 1, 3}));
    CHECK(mesh.boundary_elements[0].boundary_condition == 0 && mesh.boundary_elements[1].boundary_condition == 1);
    CHECK(mesh.boundary_conditions.size() == 2);
    CHECK(mesh.boundary_conditions[0].type == MeshProperties::BcType::Dirichlet);
    CHECK(mesh.boundary_conditions[0].value == 0.0 && !mesh.boundary_conditions[0].alpha);
    CHECK(mesh.boundary_conditions[1].type == MeshProperties::BcType::Neumann);
    CHECK(mesh.charges.size() == 2);
    CHECK(mesh.charges[0].value == 1 && mesh.charges[1].value == -1);
    CHECK(mesh.charges[1].position[2] == -0.5);
    CHECK(mesh.info.warnings.size() == 1);
    CHECK(has_warning(mesh.info.warnings, "1 tetrahedra had a negative volume"));
    CHECK(mesh.info.time > 0 && mesh.info.time_read > 0);

    // ---- warnings
    const MeshProperties no_material = importer.import_mesh(mesh_file);
    CHECK(has_warning(no_material.info.warnings, "no material file for volume 'siliconnitride'"));
    CHECK(!has_warning(no_material.info.warnings, "vacuum"));
    CHECK(no_material.charges.empty());

    const fs::path voltage_mesh =
        write_file(dir, "voltage.msh", replaced(small_mesh, "\"siliconnitride\"", "\"silicon -5V\""));
    const MeshProperties with_voltage = importer.import_mesh(voltage_mesh, {material_file});
    CHECK(with_voltage.materials[0].name == "silicon");
    CHECK(with_voltage.materials[0].voltage == -5.0 && !with_voltage.materials[0].is_metal);
    CHECK(!with_voltage.materials[1].voltage);
    CHECK(has_warning(with_voltage.info.warnings, "'silicon' has a voltage but its material is not a metal"));
    CHECK(has_warning(with_voltage.info.warnings, "'siliconnitride'"));   // material file not used

    // A metal: no epsilon_r needed, and a voltage on it is fine.
    const fs::path metal_file = write_file(dir, "silicon.yaml", "name: silicon\nband_structure:\n    model: metal\n    fermi: 7 eV\n");
    const MeshProperties metal = importer.import_mesh(voltage_mesh, {metal_file});
    CHECK(metal.materials[0].is_metal && metal.materials[0].voltage == -5.0 && metal.materials[0].eps_r == 1.0);
    CHECK(metal.info.warnings.size() == 1);   // only the reoriented tet
    CHECK(!mesh.materials[0].is_metal);       // siliconnitride.yaml has no band_structure

    // As in Nebula, a detector with a voltage is a metal.
    const fs::path detector_mesh =
        write_file(dir, "detector.msh", replaced(small_mesh, "\"siliconnitride\"", "\"detector 10V\""));
    const MeshProperties detector = importer.import_mesh(detector_mesh);
    CHECK(detector.materials[0].name == "detector" && detector.materials[0].is_metal && detector.materials[0].voltage == 10.0);

    const fs::path dirichlet_mesh = write_file(dir, "dirichlet.msh", replaced(small_mesh, "\"0V\"", "\"-2.5V\""));
    CHECK(importer.import_mesh(dirichlet_mesh).boundary_conditions[0].value == -2.5);

    const fs::path far_charge = write_file(dir, "far.cha", charge_record(100.0f, 0.0f, 0.0f, 1));
    CHECK(has_warning(importer.import_mesh(mesh_file, {}, far_charge).info.warnings, "outside the bounding box"));

    // ---- invalid meshes
    const auto broken = [&](const std::string& from, const std::string& to) {
        return write_file(dir, "broken.msh", replaced(small_mesh, from, to));
    };
    CHECK_THROWS(importer.import_mesh(dir / "missing.msh"), "file not found");
    CHECK_THROWS(importer.import_mesh(material_file), "is this a Gmsh file?");
    CHECK_THROWS(importer.import_mesh(broken("2.2 0 8", "4.1 0 8")), "unsupported Gmsh format version");
    CHECK_THROWS(importer.import_mesh(broken("2.2 0 8", "2.2 1 8")), "binary");
    CHECK_THROWS(importer.import_mesh(broken("\"0V\"", "\"ground\"")), "not a boundary condition");
    CHECK_THROWS(importer.import_mesh(broken("2 9 \"D0\"", "2 7 \"D0\"")), "named twice");
    CHECK_THROWS(importer.import_mesh(broken("20 1 0 0", "10 1 0 0")), "more than once");
    CHECK_THROWS(importer.import_mesh(broken("30 0 1 0", "30 0 x 0")), "invalid node line");
    CHECK_THROWS(importer.import_mesh(broken("40 0 0 1", "40 1 1 0")), "zero volume");
    CHECK_THROWS(importer.import_mesh(broken("10 20 30 50", "10 20 30 99")), "node id 99 does not exist");
    CHECK_THROWS(importer.import_mesh(broken("10 20 30 50", "10 20 30 30")), "twice");
    CHECK_THROWS(importer.import_mesh(broken("10 20 30 50", "10 20 30")), "too few nodes");
    CHECK_THROWS(importer.import_mesh(broken("4 4 2 2 2", "4 4 2 3 2")), "not a named volume");
    CHECK_THROWS(importer.import_mesh(broken("1 2 2 7 1 10 20 40", "1 2 2 7 1 20 40 50")), "not a face of any tetrahedron");
    CHECK_THROWS(importer.import_mesh(broken("4 4 2 2 2 10 20 30 50", "4 5 2 2 2 10 20 30 40 50 10 20 30")),
                 "unsupported element type 5");
    CHECK_THROWS(importer.import_mesh(broken("$Elements\n4", "$Elements\n3")), "expected $EndElements");
    CHECK_THROWS(importer.import_mesh(broken("$Elements\n4", "$Elements\n5")), "invalid element line");
    CHECK_THROWS(importer.import_mesh(broken("$EndElements\n", "")), "unexpected end of file");
    CHECK_THROWS(importer.import_mesh(broken("$Nodes", "$Knots")), "expected $EndKnots");
    // A section we do not know is skipped.
    CHECK(importer.import_mesh(broken("$Nodes", "$Comments\nanything\n$EndComments\n$Nodes")).nodes.size() == 5);
    // A third tet on the shared face.
    const std::string third_tet = replaced(replaced(small_mesh, "$Nodes\n5", "$Nodes\n6\n60 1 1 2"), "$Elements\n4",
                                           "$Elements\n5\n9 4 2 5 1 10 20 30 60");
    CHECK_THROWS(importer.import_mesh(write_file(dir, "third.msh", third_tet)), "more than two tetrahedra");

    // A node that no tet uses is only a warning.
    const MeshProperties unused_node = importer.import_mesh(broken("$Nodes\n5", "$Nodes\n6\n60 5 5 5"));
    CHECK(has_warning(unused_node.info.warnings, "1 nodes do not belong to any tetrahedron"));

    // ---- invalid material and charge files
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_file(dir, "a.yaml", "density: 1\n")}), "has no 'name'");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_file(dir, "b.yaml", replaced(small_material, "7.5", "high"))}),
                 "epsilon_r is not a number");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_file(dir, "c.yaml", replaced(small_material, "7.5", "-1"))}),
                 "must be positive");
    CHECK_THROWS(importer.import_mesh(mesh_file, {material_file, material_file}), "more than one file");
    CHECK_THROWS(importer.import_mesh(mesh_file, {}, write_file(dir, "short.cha", std::string(15, '\0'))),
                 "not a multiple of 16");
    // Any charge value is accepted; zero is only a warning.
    const MeshProperties general =
        importer.import_mesh(mesh_file, {}, write_file(dir, "q.cha", charge_record(0, 0, 0, -300) + charge_record(0, 0, 0, 0)));
    CHECK(general.charges[0].value == -300 && general.charges[1].value == 0);
    CHECK(has_warning(general.info.warnings, "1 trapped charges have a charge of zero"));

    // ---- primaries
    const fs::path primaries_file =
        write_file(dir, "small.pri", primary_record(35.0f, -1.0f, 500.0f) + primary_record(36.0f, -2.0f, 300.0f));
    const FreeCharges free_charges = importer.import_free_charges(primaries_file);
    CHECK(free_charges.electrons.size() == 2);
    CHECK((free_charges.electrons[0].position == std::array<double, 3>{1, 2, 35}));
    CHECK((free_charges.electrons[1].direction == std::array<double, 3>{0, 0, -2}));
    CHECK(free_charges.electrons[1].energy == 300.0);
    CHECK((free_charges.electrons[0].pixel == std::array<int32_t, 2>{3, 4}));
    CHECK(free_charges.info.warnings.empty());

    CHECK_THROWS(importer.import_free_charges(write_file(dir, "short.pri", std::string(35, '\0'))), "not a multiple of 36");
    CHECK_THROWS(importer.import_free_charges(write_file(dir, "e.pri", primary_record(35.0f, -1.0f, 0.0f))),
                 "energy must be positive");
    CHECK_THROWS(importer.import_free_charges(write_file(dir, "d.pri", primary_record(35.0f, 0.0f, 500.0f))),
                 "direction is zero");
}

void test_sample_files(const fs::path& msh, const fs::path& yaml, const fs::path& cha, const fs::path& pri) {
    const Importer importer;
    const MeshProperties mesh = importer.import_mesh(msh, {yaml}, cha);
    CHECK(mesh.nodes.size() == 7792);
    CHECK(mesh.elements.size() == 40795);
    CHECK(mesh.boundary_elements.size() == 1143);
    CHECK(mesh.materials.size() == 3);
    CHECK(mesh.materials[0].name == "siliconnitride" && mesh.materials[0].eps_r == 7.5);
    CHECK(mesh.materials[1].name == "vacuum" && mesh.materials[2].name == "detector");
    CHECK(!mesh.materials[0].is_metal && !mesh.materials[2].is_metal && !mesh.materials[2].voltage);
    CHECK(mesh.boundary_conditions.size() == 1);
    CHECK(mesh.boundary_conditions[0].type == MeshProperties::BcType::Dirichlet && mesh.boundary_conditions[0].value == 0);
    CHECK(mesh.charges.size() == 40);
    std::size_t outer_faces = 0, mismatched = 0;
    for (uint32_t e = 0; e < mesh.elements.size(); ++e)
        for (const auto& other : mesh.neighbours[e]) {
            if (!other) {
                ++outer_faces;
                continue;
            }
            const auto& back = mesh.neighbours[*other];
            if (std::find(back.begin(), back.end(), std::optional<uint32_t>(e)) == back.end()) ++mismatched;
        }
    CHECK(outer_faces == 4962);   // faces on the outside of the sample mesh
    CHECK(mismatched == 0);       // if a is a neighbour of b, b is a neighbour of a
    CHECK((mesh.charges[0].position == std::array<double, 3>{2, 0, -20}) && mesh.charges[0].value == 1);

    const FreeCharges free_charges = importer.import_free_charges(pri);
    CHECK(free_charges.electrons.size() == 494415);
    CHECK(free_charges.electrons[0].energy == 500.0);

    std::cout << "sample mesh: " << mesh.nodes.size() << " nodes, " << mesh.elements.size() << " tets, "
              << mesh.boundary_elements.size() << " boundary triangles, " << mesh.charges.size() << " charges\n"
              << "  time " << mesh.info.time << " s (read " << mesh.info.time_read << " s, geometry "
              << mesh.info.time_geometry << " s)\n";
    for (const auto& warning : mesh.info.warnings) std::cout << "  warning: " << warning << "\n";
    std::cout << "sample primaries: " << free_charges.electrons.size() << " electrons, time "
              << free_charges.info.time << " s\n";
    for (const auto& warning : free_charges.info.warnings) std::cout << "  warning: " << warning << "\n";
}

int main(int argc, char** argv) {
    const fs::path dir = fs::temp_directory_path() / "poisson_test_importer";
    fs::create_directories(dir);

    try {
        test_small_files(dir);
        if (argc == 5) test_sample_files(argv[1], argv[2], argv[3], argv[4]);
        else if (argc != 1) std::cerr << "usage: test_importer [<msh> <yaml> <cha> <pri>]\n", ++failures;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << "\n";
        ++failures;
    }

    fs::remove_all(dir);
    std::cout << (failures == 0 ? "all tests passed" : "TESTS FAILED") << "\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
