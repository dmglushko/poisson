// Tests for the importer.
//
//   test_importer                              small generated files only
//   test_importer <msh> <yaml> <cha> <pri>     also import the holes_3x2 sample files

#include "assembled_problem.h"
#include "importer.h"
#include "solver_result.h"

#include <hdf5.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <optional>
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

// Writes a material file the way Nebula's cstool does: text properties as strings and
// quantities as a compound {value, unit}, all as attributes of the root group.
struct Quantity { double value; const char* unit; };
fs::path write_mat(const fs::path& directory, const std::string& file_name, const std::string& name,
                   const std::string& conductor_type, std::optional<Quantity> eps_r = std::nullopt,
                   std::optional<Quantity> screening_length = std::nullopt) {
    const fs::path path = directory / file_name;
    const hid_t file = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const hid_t scalar = H5Screate(H5S_SCALAR);
    const hid_t text = H5Tcopy(H5T_C_S1);
    H5Tset_size(text, H5T_VARIABLE);
    const hid_t quantity = H5Tcreate(H5T_COMPOUND, sizeof(Quantity));
    H5Tinsert(quantity, "value", HOFFSET(Quantity, value), H5T_NATIVE_DOUBLE);
    H5Tinsert(quantity, "unit", HOFFSET(Quantity, unit), text);

    const auto write_text = [&](const char* key, const std::string& value) {
        const hid_t fixed = H5Tcopy(H5T_C_S1);
        H5Tset_size(fixed, value.size() + 1);
        const hid_t attribute = H5Acreate2(file, key, fixed, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, fixed, value.c_str());
        H5Aclose(attribute);
        H5Tclose(fixed);
    };
    const auto write_quantity = [&](const char* key, const Quantity& value) {
        const hid_t attribute = H5Acreate2(file, key, quantity, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attribute, quantity, &value);
        H5Aclose(attribute);
    };
    if (!name.empty()) write_text("name", name);
    write_text("conductor_type", conductor_type);
    if (eps_r) write_quantity("epsilon_r", *eps_r);
    if (screening_length) write_quantity("screening_length", *screening_length);

    H5Tclose(quantity);
    H5Tclose(text);
    H5Sclose(scalar);
    H5Fclose(file);
    return path;
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
    const MeshImportResult imported = importer.import_mesh(mesh_file, {material_file}, charge_file);
    const MeshProperties& mesh = imported.mesh;
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
    CHECK(imported.info.warnings.size() == 1);
    CHECK(has_warning(imported.info.warnings, "1 tetrahedra had a negative volume"));
    CHECK(imported.info.time > 0 && imported.info.time_read > 0);

    // ---- warnings
    const MeshImportResult no_material = importer.import_mesh(mesh_file);
    CHECK(has_warning(no_material.info.warnings, "no material file for volume 'siliconnitride'"));
    CHECK(!has_warning(no_material.info.warnings, "vacuum"));
    CHECK(no_material.mesh.charges.empty());

    const fs::path voltage_mesh =
        write_file(dir, "voltage.msh", replaced(small_mesh, "\"siliconnitride\"", "\"silicon -5V\""));
    const MeshImportResult with_voltage = importer.import_mesh(voltage_mesh, {material_file});
    CHECK(with_voltage.mesh.materials[0].name == "silicon");
    CHECK(with_voltage.mesh.materials[0].voltage == -5.0 && !with_voltage.mesh.materials[0].is_metal);
    CHECK(!with_voltage.mesh.materials[1].voltage);
    CHECK(has_warning(with_voltage.info.warnings, "'silicon' has a voltage but its material is not a metal"));
    CHECK(has_warning(with_voltage.info.warnings, "'siliconnitride'"));   // material file not used

    // A metal: no epsilon_r needed, and a voltage on it is fine.
    const fs::path metal_file = write_file(dir, "silicon.yaml", "name: silicon\nband_structure:\n    model: metal\n    fermi: 7 eV\n");
    const MeshImportResult metal = importer.import_mesh(voltage_mesh, {metal_file});
    CHECK(metal.mesh.materials[0].is_metal && metal.mesh.materials[0].voltage == -5.0 && metal.mesh.materials[0].eps_r == 1.0);
    CHECK(metal.info.warnings.size() == 1);   // only the reoriented tet
    CHECK(!mesh.materials[0].is_metal);       // siliconnitride.yaml has no band_structure

    // YAML edge cases: quoted model, and sibling keys containing "model" or "epsilon_r" as substrings.
    const fs::path quoted_metal = write_file(
        dir, "quoted_metal.yaml",
        "name: silicon\nband_structure:\n    model: \"metal\"\n    dos_model: parabolic\n");
    CHECK(importer.import_mesh(voltage_mesh, {quoted_metal}).mesh.materials[0].is_metal);
    const fs::path sibling_eps = write_file(
        dir, "sibling_eps.yaml",
        "name: siliconnitride\ndielectric:\n    static_epsilon_r: 99.0\n    epsilon_r_inf: 2.0\n    epsilon_r: 7.5\n");
    CHECK(importer.import_mesh(mesh_file, {sibling_eps}).mesh.materials[0].eps_r == 7.5);

    // Nebula .mat file with fixed-length (non-VLEN) string `unit` inside compound attributes.
    {
        struct FixedQuantity { double value; char unit[8]; };
        const fs::path fixed_mat = dir / "fixed_unit.mat";
        const hid_t file = H5Fcreate(fixed_mat.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        const hid_t scalar = H5Screate(H5S_SCALAR);
        const auto write_str = [&](const char* key, const std::string& val) {
            const hid_t t = H5Tcopy(H5T_C_S1);
            H5Tset_size(t, val.size() + 1);
            const hid_t a = H5Acreate2(file, key, t, scalar, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, t, val.c_str());
            H5Aclose(a);
            H5Tclose(t);
        };
        write_str("name", "siliconnitride");
        write_str("conductor_type", "insulator");
        const hid_t ustr = H5Tcopy(H5T_C_S1);
        H5Tset_size(ustr, sizeof(FixedQuantity::unit));
        const hid_t fq = H5Tcreate(H5T_COMPOUND, sizeof(FixedQuantity));
        H5Tinsert(fq, "value", HOFFSET(FixedQuantity, value), H5T_NATIVE_DOUBLE);
        H5Tinsert(fq, "unit", HOFFSET(FixedQuantity, unit), ustr);
        const FixedQuantity eps_val{8.25, ""};
        const FixedQuantity sl_val{4.0, "um"};
        const hid_t a1 = H5Acreate2(file, "epsilon_r", fq, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a1, fq, &eps_val);
        H5Aclose(a1);
        const hid_t a2 = H5Acreate2(file, "screening_length", fq, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a2, fq, &sl_val);
        H5Aclose(a2);
        H5Tclose(fq);
        H5Tclose(ustr);
        H5Sclose(scalar);
        H5Fclose(file);

        const MeshImportResult fixed_res = importer.import_mesh(mesh_file, {fixed_mat});
        CHECK(fixed_res.mesh.materials[0].eps_r == 8.25);
        CHECK(std::abs(fixed_res.mesh.materials[0].screening_length - 4000.0) < 1e-12);
    }

    // Nebula .mat files (HDF5) give the same as the .yaml, plus the screening length.
    const MeshImportResult from_mat = importer.import_mesh(
        mesh_file, {write_mat(dir, "siliconnitride.mat", "siliconnitride", "semiconductor", Quantity{7.5, ""}, Quantity{2.0, "nm"})});
    CHECK(from_mat.mesh.materials[0].eps_r == 7.5 && !from_mat.mesh.materials[0].is_metal);
    CHECK(from_mat.mesh.materials[0].screening_length == 2.0);
    CHECK(from_mat.info.warnings.size() == 1);   // only the reoriented tet
    const MeshImportResult in_metres = importer.import_mesh(
        mesh_file, {write_mat(dir, "m.mat", "siliconnitride", "insulator", Quantity{7.5, ""}, Quantity{3e-9, "m"})});
    CHECK(std::abs(in_metres.mesh.materials[0].screening_length - 3.0) < 1e-12);
    const MeshImportResult metal_mat = importer.import_mesh(voltage_mesh, {write_mat(dir, "silicon.mat", "silicon", "metal")});
    CHECK(metal_mat.mesh.materials[0].is_metal && metal_mat.mesh.materials[0].voltage == -5.0);
    CHECK(metal_mat.mesh.materials[0].screening_length == 0.0);
    CHECK_THROWS(importer.import_mesh(mesh_file, {dir / "missing.mat"}), "file not found");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_file(dir, "text.mat", small_material)}), "not a Nebula material file");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_mat(dir, "noname.mat", "", "metal")}), "has no 'name'");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_mat(dir, "neg.mat", "siliconnitride", "insulator", Quantity{-1, ""})}),
                 "epsilon_r must be positive");
    CHECK_THROWS(importer.import_mesh(mesh_file, {write_mat(dir, "unit.mat", "siliconnitride", "insulator", Quantity{7.5, ""},
                                                            Quantity{2.0, "eV"})}),
                 "unknown unit 'eV'");

    // As in Nebula, a detector with a voltage is a metal.
    const fs::path detector_mesh =
        write_file(dir, "detector.msh", replaced(small_mesh, "\"siliconnitride\"", "\"detector 10V\""));
    const MeshImportResult detector = importer.import_mesh(detector_mesh);
    CHECK(detector.mesh.materials[0].name == "detector" && detector.mesh.materials[0].is_metal && detector.mesh.materials[0].voltage == 10.0);

    // Signs: "+5V" and "-5V" glued to the name, an explicit "+" on a surface and on a coordinate.
    const auto voltage_of = [&](const std::string& name) {
        const fs::path file = write_file(dir, "sign.msh", replaced(small_mesh, "\"siliconnitride\"", "\"" + name + "\""));
        return importer.import_mesh(file).mesh.materials[0];
    };
    CHECK(voltage_of("detector+5V").voltage == 5.0 && voltage_of("detector+5V").is_metal);
    CHECK(voltage_of("detector-5V").voltage == -5.0);
    CHECK(voltage_of("silicon +2.5V").voltage == 2.5 && voltage_of("silicon +2.5V").name == "silicon");
    CHECK(voltage_of("silicon-3V").voltage == -3.0 && voltage_of("silicon-3V").name == "silicon");
    CHECK(!voltage_of("silicon5V").voltage);   // glued to the word: part of the name
    const fs::path plus_mesh = write_file(
        dir, "plus.msh", replaced(replaced(small_mesh, "\"0V\"", "\"+3V\""), "20 1 0 0", "20 +1 0 0"));
    CHECK(importer.import_mesh(plus_mesh).mesh.boundary_conditions[0].value == 3.0);
    CHECK(importer.import_mesh(plus_mesh).mesh.nodes[1][0] == 1.0);

    // Only a Neumann condition: nothing fixes the potential. A volume with a voltage does.
    const std::string neumann_only = replaced(small_mesh, "\"0V\"", "\"D1\"");
    CHECK(has_warning(importer.import_mesh(write_file(dir, "neumann.msh", neumann_only)).info.warnings,
                      "no potential is prescribed anywhere"));
    CHECK(!has_warning(importer.import_mesh(write_file(dir, "neumann.msh", replaced(neumann_only, "\"siliconnitride\"", "\"gold 1V\"")))
                            .info.warnings, "no potential is prescribed anywhere"));
    CHECK(!has_warning(imported.info.warnings, "no potential") && !has_warning(imported.info.warnings, "pieces"));

    // Two tets that share no face: the mesh is in two pieces.
    std::string two_pieces = replaced(small_mesh, "$Nodes\n5", "$Nodes\n8\n60 5 0 0\n70 6 0 0\n80 5 1 0");
    two_pieces = replaced(two_pieces, "50 0 0 -1", "50 5 0 1");
    two_pieces = replaced(two_pieces, "4 4 2 2 2 10 20 30 50", "4 4 2 2 2 60 70 80 50");
    two_pieces = replaced(two_pieces, "2 2 2 9 1 10 30 50", "2 2 2 9 1 60 80 50");
    CHECK(has_warning(importer.import_mesh(write_file(dir, "pieces.msh", two_pieces)).info.warnings, "consists of 2 pieces"));

    const fs::path dirichlet_mesh = write_file(dir, "dirichlet.msh", replaced(small_mesh, "\"0V\"", "\"-2.5V\""));
    CHECK(importer.import_mesh(dirichlet_mesh).mesh.boundary_conditions[0].value == -2.5);

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
    CHECK(importer.import_mesh(broken("$Nodes", "$Comments\nanything\n$EndComments\n$Nodes")).mesh.nodes.size() == 5);
    // A third tet on the shared face.
    const std::string third_tet = replaced(replaced(small_mesh, "$Nodes\n5", "$Nodes\n6\n60 1 1 2"), "$Elements\n4",
                                           "$Elements\n5\n9 4 2 5 1 10 20 30 60");
    CHECK_THROWS(importer.import_mesh(write_file(dir, "third.msh", third_tet)), "more than two tetrahedra");
    // The same tet twice, two tets on the same side of a face, the same triangle twice.
    CHECK_THROWS(importer.import_mesh(broken("$Elements\n4", "$Elements\n5\n9 4 2 2 1 20 10 40 30")), "listed more than once");
    CHECK_THROWS(importer.import_mesh(broken("50 0 0 -1", "50 0.2 0.2 0.5")), "two tetrahedra overlap");
    CHECK_THROWS(importer.import_mesh(broken("$Elements\n4", "$Elements\n5\n9 2 2 7 1 40 20 10")), "is already listed on line 21");
    CHECK_THROWS(importer.import_mesh(broken("$Elements\n4", "$Elements\n5\n9 2 2 9 1 40 20 10")),
                 "with another boundary condition ('D0' and '0V')");

    // A node that no tet uses is only a warning.
    const MeshImportResult unused_node = importer.import_mesh(broken("$Nodes\n5", "$Nodes\n6\n60 5 5 5"));
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
    const MeshImportResult general =
        importer.import_mesh(mesh_file, {}, write_file(dir, "q.cha", charge_record(0, 0, 0, -300) + charge_record(0, 0, 0, 0)));
    CHECK(general.mesh.charges[0].value == -300 && general.mesh.charges[1].value == 0);
    CHECK(has_warning(general.info.warnings, "1 trapped charges have a charge of zero"));

    // ---- primaries
    const fs::path primaries_file =
        write_file(dir, "small.pri", primary_record(35.0f, -1.0f, 500.0f) + primary_record(36.0f, -2.0f, 300.0f));
    const FreeChargesImportResult imported_charges = importer.import_free_charges(primaries_file);
    const FreeCharges& free_charges = imported_charges.charges;
    CHECK(free_charges.electrons.size() == 2);
    CHECK((free_charges.electrons[0].position == std::array<double, 3>{1, 2, 35}));
    CHECK((free_charges.electrons[1].direction == std::array<double, 3>{0, 0, -2}));
    CHECK(free_charges.electrons[1].energy == 300.0);
    CHECK((free_charges.electrons[0].pixel == std::array<int32_t, 2>{3, 4}));
    CHECK(imported_charges.info.warnings.empty());

    CHECK_THROWS(importer.import_free_charges(write_file(dir, "short.pri", std::string(35, '\0'))), "not a multiple of 36");
    CHECK_THROWS(importer.import_free_charges(write_file(dir, "e.pri", primary_record(35.0f, -1.0f, 0.0f))),
                 "energy must be positive");
    CHECK_THROWS(importer.import_free_charges(write_file(dir, "d.pri", primary_record(35.0f, 0.0f, 500.0f))),
                 "direction is zero");
}

void test_sample_files(const fs::path& msh, const fs::path& yaml, const fs::path& cha, const fs::path& pri) {
    const Importer importer;
    const MeshImportResult imported_mesh = importer.import_mesh(msh, {yaml}, cha);
    const MeshProperties& mesh = imported_mesh.mesh;
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

    const FreeChargesImportResult imported_charges = importer.import_free_charges(pri);
    const FreeCharges& free_charges = imported_charges.charges;
    CHECK(free_charges.electrons.size() == 494415);
    CHECK(free_charges.electrons[0].energy == 500.0);

    std::cout << "sample mesh: " << mesh.nodes.size() << " nodes, " << mesh.elements.size() << " tets, "
              << mesh.boundary_elements.size() << " boundary triangles, " << mesh.charges.size() << " charges\n"
              << "  time " << imported_mesh.info.time << " s (read " << imported_mesh.info.time_read << " s, geometry "
              << imported_mesh.info.time_geometry << " s)\n";
    for (const auto& warning : imported_mesh.info.warnings) std::cout << "  warning: " << warning << "\n";
    std::cout << "sample primaries: " << free_charges.electrons.size() << " electrons, time "
              << imported_charges.info.time << " s\n";
    for (const auto& warning : imported_charges.info.warnings) std::cout << "  warning: " << warning << "\n";
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
