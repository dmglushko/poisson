#include "serialization.h"

#include <hdf5.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace poisson::io {
namespace {

namespace fs = std::filesystem;

constexpr uint32_t kSchemaVersion = 1;
constexpr uint32_t kNoneU32 = std::numeric_limits<uint32_t>::max();

[[noreturn]] void fail(const fs::path& file, const std::string& message) {
    throw std::runtime_error(file.string() + ": " + message);
}

struct H5Handle {
    hid_t id;
    herr_t (*close)(hid_t);
    ~H5Handle() {
        if (id >= 0) close(id);
    }
    H5Handle(hid_t id, herr_t (*close)(hid_t)) : id(id), close(close) {}
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept : id(other.id), close(other.close) { other.id = -1; }
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            if (id >= 0) close(id);
            id = other.id;
            close = other.close;
            other.id = -1;
        }
        return *this;
    }

    // Explicitly flush (for files) and close while checking HDF5 error codes.
    void close_file_checked(const fs::path& file) {
        if (id < 0) return;
        const herr_t flush_err = H5Fflush(id, H5F_SCOPE_GLOBAL);
        const herr_t close_err = close(id);
        id = -1;
        if (flush_err < 0 || close_err < 0) fail(file, "failed to flush or close HDF5 file");
    }
};

// Creates an HDF5 file configured for deterministic byte output (no object timestamps,
// fixed library version bounds).
H5Handle create_deterministic_h5(const fs::path& file) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    const H5Handle fcpl(H5Pcreate(H5P_FILE_CREATE), H5Pclose);
    const H5Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (fcpl.id < 0 || fapl.id < 0) fail(file, "failed to create HDF5 property lists");
    if (H5Pset_obj_track_times(fcpl.id, false) < 0) fail(file, "failed to disable root timestamp tracking");
    if (H5Pset_libver_bounds(fapl.id, H5F_LIBVER_V18, H5F_LIBVER_V18) < 0)
        fail(file, "failed to set HDF5 library version bounds");

    H5Handle h5(H5Fcreate(file.string().c_str(), H5F_ACC_TRUNC, fcpl.id, fapl.id), H5Fclose);
    if (h5.id < 0) fail(file, "cannot create HDF5 file");
    return h5;
}

H5Handle open_readonly_h5(const fs::path& file) {
    if (!fs::is_regular_file(file)) fail(file, "file not found");
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    H5Handle h5(H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (h5.id < 0) fail(file, "cannot open HDF5 file");
    return h5;
}

H5Handle create_group(const fs::path& file, hid_t parent, const char* name) {
    const H5Handle gcpl(H5Pcreate(H5P_GROUP_CREATE), H5Pclose);
    if (gcpl.id < 0 || H5Pset_obj_track_times(gcpl.id, false) < 0)
        fail(file, "failed to create group property list for '" + std::string(name) + "'");
    H5Handle group(H5Gcreate2(parent, name, H5P_DEFAULT, gcpl.id, H5P_DEFAULT), H5Gclose);
    if (group.id < 0) fail(file, "failed to create group '" + std::string(name) + "'");
    return group;
}

H5Handle open_group(const fs::path& file, hid_t parent, const char* name) {
    if (H5Lexists(parent, name, H5P_DEFAULT) <= 0)
        fail(file, "missing group '" + std::string(name) + "'");
    H5Handle group(H5Gopen2(parent, name, H5P_DEFAULT), H5Gclose);
    if (group.id < 0) fail(file, "cannot open group '" + std::string(name) + "'");
    return group;
}

void write_string_attr(const fs::path& file, hid_t obj, const char* name, std::string_view value) {
    const H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    const H5Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    if (space.id < 0 || type.id < 0) fail(file, "failed to create string attribute '" + std::string(name) + "'");
    H5Tset_size(type.id, value.empty() ? 1 : value.size());
    H5Tset_strpad(type.id, H5T_STR_NULLPAD);
    H5Tset_cset(type.id, H5T_CSET_UTF8);
    const H5Handle attr(H5Acreate2(obj, name, type.id, space.id, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (attr.id < 0) fail(file, "failed to create attribute '" + std::string(name) + "'");
    const char empty = '\0';
    if (H5Awrite(attr.id, type.id, value.empty() ? &empty : value.data()) < 0)
        fail(file, "failed to write attribute '" + std::string(name) + "'");
}

std::string read_string_attr(const fs::path& file, hid_t obj, const char* name) {
    if (H5Aexists(obj, name) <= 0) fail(file, "missing attribute '" + std::string(name) + "'");
    const H5Handle attr(H5Aopen(obj, name, H5P_DEFAULT), H5Aclose);
    const H5Handle type(H5Aget_type(attr.id), H5Tclose);
    const H5Handle space(H5Aget_space(attr.id), H5Sclose);
    if (attr.id < 0 || type.id < 0 || space.id < 0)
        fail(file, "cannot inspect attribute '" + std::string(name) + "'");
    if (H5Sget_simple_extent_type(space.id) != H5S_SCALAR)
        fail(file, "attribute '" + std::string(name) + "' must be a scalar");
    if (H5Tget_class(type.id) != H5T_STRING)
        fail(file, "attribute '" + std::string(name) + "' is not a string");
    if (H5Tis_variable_str(type.id) > 0) {
        const H5Handle mem(H5Tcopy(H5T_C_S1), H5Tclose);
        H5Tset_size(mem.id, H5T_VARIABLE);
        H5Tset_cset(mem.id, H5Tget_cset(type.id));
        char* p = nullptr;
        if (H5Aread(attr.id, mem.id, &p) < 0) fail(file, "cannot read attribute '" + std::string(name) + "'");
        std::string out = p ? p : "";
        if (p) H5free_memory(p);
        return out;
    }
    const std::size_t size = H5Tget_size(type.id);
    std::string buffer(size, '\0');
    if (size > 0 && H5Aread(attr.id, type.id, buffer.data()) < 0)
        fail(file, "cannot read attribute '" + std::string(name) + "'");
    while (!buffer.empty() && buffer.back() == '\0') buffer.pop_back();
    return buffer;
}

void write_u32_attr(const fs::path& file, hid_t obj, const char* name, uint32_t value) {
    const H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    const H5Handle attr(H5Acreate2(obj, name, H5T_STD_U32LE, space.id, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (attr.id < 0 || H5Awrite(attr.id, H5T_NATIVE_UINT32, &value) < 0)
        fail(file, "failed to write attribute '" + std::string(name) + "'");
}

uint32_t read_u32_attr(const fs::path& file, hid_t obj, const char* name) {
    if (H5Aexists(obj, name) <= 0) fail(file, "missing attribute '" + std::string(name) + "'");
    const H5Handle attr(H5Aopen(obj, name, H5P_DEFAULT), H5Aclose);
    const H5Handle type(H5Aget_type(attr.id), H5Tclose);
    const H5Handle space(H5Aget_space(attr.id), H5Sclose);
    if (attr.id < 0 || type.id < 0 || space.id < 0)
        fail(file, "cannot inspect attribute '" + std::string(name) + "'");
    if (H5Sget_simple_extent_type(space.id) != H5S_SCALAR)
        fail(file, "attribute '" + std::string(name) + "' must be a scalar");
    if (H5Tequal(type.id, H5T_STD_U32LE) <= 0)
        fail(file, "attribute '" + std::string(name) + "' must have type uint32 (STD_U32LE)");
    uint32_t value = 0;
    if (H5Aread(attr.id, H5T_NATIVE_UINT32, &value) < 0)
        fail(file, "cannot read uint32 attribute '" + std::string(name) + "'");
    return value;
}

void write_u64_attr(const fs::path& file, hid_t obj, const char* name, uint64_t value) {
    const H5Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    const H5Handle attr(H5Acreate2(obj, name, H5T_STD_U64LE, space.id, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (attr.id < 0 || H5Awrite(attr.id, H5T_NATIVE_UINT64, &value) < 0)
        fail(file, "failed to write attribute '" + std::string(name) + "'");
}

uint64_t read_u64_attr(const fs::path& file, hid_t obj, const char* name) {
    if (H5Aexists(obj, name) <= 0) fail(file, "missing attribute '" + std::string(name) + "'");
    const H5Handle attr(H5Aopen(obj, name, H5P_DEFAULT), H5Aclose);
    const H5Handle type(H5Aget_type(attr.id), H5Tclose);
    const H5Handle space(H5Aget_space(attr.id), H5Sclose);
    if (attr.id < 0 || type.id < 0 || space.id < 0)
        fail(file, "cannot inspect attribute '" + std::string(name) + "'");
    if (H5Sget_simple_extent_type(space.id) != H5S_SCALAR)
        fail(file, "attribute '" + std::string(name) + "' must be a scalar");
    if (H5Tequal(type.id, H5T_STD_U64LE) <= 0)
        fail(file, "attribute '" + std::string(name) + "' must have type uint64 (STD_U64LE)");
    uint64_t value = 0;
    if (H5Aread(attr.id, H5T_NATIVE_UINT64, &value) < 0)
        fail(file, "cannot read uint64 attribute '" + std::string(name) + "'");
    return value;
}

void write_schema_header(const fs::path& file, hid_t h5, std::string_view schema) {
    write_string_attr(file, h5, "schema", schema);
    write_u32_attr(file, h5, "schema_version", kSchemaVersion);
}

void verify_schema_header(const fs::path& file, hid_t h5, std::string_view expected_schema) {
    const std::string actual = read_string_attr(file, h5, "schema");
    if (actual != expected_schema)
        fail(file, "expected schema '" + std::string(expected_schema) + "', found '" + actual + "'");
    const uint32_t version = read_u32_attr(file, h5, "schema_version");
    if (version != kSchemaVersion)
        fail(file, "unsupported schema_version " + std::to_string(version) + ", expected " +
                       std::to_string(kSchemaVersion));
}

template <typename T>
hid_t h5_file_type();
template <typename T>
hid_t h5_mem_type();
template <typename T>
const char* h5_type_name();

template <> hid_t h5_file_type<double>() { return H5T_IEEE_F64LE; }
template <> hid_t h5_mem_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> const char* h5_type_name<double>() { return "float64"; }

template <> hid_t h5_file_type<uint64_t>() { return H5T_STD_U64LE; }
template <> hid_t h5_mem_type<uint64_t>() { return H5T_NATIVE_UINT64; }
template <> const char* h5_type_name<uint64_t>() { return "uint64"; }

template <> hid_t h5_file_type<uint32_t>() { return H5T_STD_U32LE; }
template <> hid_t h5_mem_type<uint32_t>() { return H5T_NATIVE_UINT32; }
template <> const char* h5_type_name<uint32_t>() { return "uint32"; }

template <> hid_t h5_file_type<int32_t>() { return H5T_STD_I32LE; }
template <> hid_t h5_mem_type<int32_t>() { return H5T_NATIVE_INT32; }
template <> const char* h5_type_name<int32_t>() { return "int32"; }

template <> hid_t h5_file_type<uint16_t>() { return H5T_STD_U16LE; }
template <> hid_t h5_mem_type<uint16_t>() { return H5T_NATIVE_UINT16; }
template <> const char* h5_type_name<uint16_t>() { return "uint16"; }

template <> hid_t h5_file_type<uint8_t>() { return H5T_STD_U8LE; }
template <> hid_t h5_mem_type<uint8_t>() { return H5T_NATIVE_UINT8; }
template <> const char* h5_type_name<uint8_t>() { return "uint8"; }

template <typename T>
void verify_dataset_type(const fs::path& file, hid_t dset, const char* name) {
    const H5Handle dtype(H5Dget_type(dset), H5Tclose);
    if (dtype.id < 0 || H5Tequal(dtype.id, h5_file_type<T>()) <= 0)
        fail(file, "dataset '" + std::string(name) + "' has unexpected datatype (expected " + h5_type_name<T>() + ")");
}

template <typename T>
void write_dataset_1d(const fs::path& file, hid_t parent, const char* name, const std::vector<T>& data) {
    const hsize_t dim = data.size();
    const H5Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
    const H5Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (space.id < 0 || dcpl.id < 0 || H5Pset_obj_track_times(dcpl.id, false) < 0)
        fail(file, "failed to prepare dataset '" + std::string(name) + "'");
    const H5Handle dset(
        H5Dcreate2(parent, name, h5_file_type<T>(), space.id, H5P_DEFAULT, dcpl.id, H5P_DEFAULT), H5Dclose);
    if (dset.id < 0) fail(file, "failed to create dataset '" + std::string(name) + "'");
    if (!data.empty()) {
        if (H5Dwrite(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0)
            fail(file, "failed to write dataset '" + std::string(name) + "'");
    }
}

template <typename T, std::size_t K>
void write_dataset_2d(const fs::path& file, hid_t parent, const char* name,
                      const std::vector<std::array<T, K>>& data) {
    const hsize_t dims[2] = {data.size(), K};
    const H5Handle space(H5Screate_simple(2, dims, nullptr), H5Sclose);
    const H5Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (space.id < 0 || dcpl.id < 0 || H5Pset_obj_track_times(dcpl.id, false) < 0)
        fail(file, "failed to prepare dataset '" + std::string(name) + "'");
    const H5Handle dset(
        H5Dcreate2(parent, name, h5_file_type<T>(), space.id, H5P_DEFAULT, dcpl.id, H5P_DEFAULT), H5Dclose);
    if (dset.id < 0) fail(file, "failed to create dataset '" + std::string(name) + "'");
    if (!data.empty()) {
        std::vector<T> flat;
        flat.reserve(data.size() * K);
        for (const auto& row : data)
            for (std::size_t j = 0; j < K; ++j) flat.push_back(row[j]);
        if (H5Dwrite(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()) < 0)
            fail(file, "failed to write dataset '" + std::string(name) + "'");
    }
}

template <typename T>
std::vector<T> read_dataset_1d(const fs::path& file, hid_t parent, const char* name) {
    if (H5Lexists(parent, name, H5P_DEFAULT) <= 0)
        fail(file, "missing dataset '" + std::string(name) + "'");
    const H5Handle dset(H5Dopen2(parent, name, H5P_DEFAULT), H5Dclose);
    const H5Handle space(H5Dget_space(dset.id), H5Sclose);
    if (dset.id < 0 || space.id < 0) fail(file, "cannot open dataset '" + std::string(name) + "'");
    verify_dataset_type<T>(file, dset.id, name);
    if (H5Sget_simple_extent_type(space.id) != H5S_SIMPLE || H5Sget_simple_extent_ndims(space.id) != 1)
        fail(file, "dataset '" + std::string(name) + "' must be 1D");
    hsize_t dim = 0;
    H5Sget_simple_extent_dims(space.id, &dim, nullptr);
    std::vector<T> out(dim);
    if (dim > 0) {
        if (H5Dread(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0)
            fail(file, "failed to read dataset '" + std::string(name) + "'");
    }
    return out;
}

template <typename T, std::size_t K>
std::vector<std::array<T, K>> read_dataset_2d(const fs::path& file, hid_t parent, const char* name) {
    if (H5Lexists(parent, name, H5P_DEFAULT) <= 0)
        fail(file, "missing dataset '" + std::string(name) + "'");
    const H5Handle dset(H5Dopen2(parent, name, H5P_DEFAULT), H5Dclose);
    const H5Handle space(H5Dget_space(dset.id), H5Sclose);
    if (dset.id < 0 || space.id < 0) fail(file, "cannot open dataset '" + std::string(name) + "'");
    verify_dataset_type<T>(file, dset.id, name);
    if (H5Sget_simple_extent_type(space.id) != H5S_SIMPLE || H5Sget_simple_extent_ndims(space.id) != 2)
        fail(file, "dataset '" + std::string(name) + "' must be 2D");
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    if (dims[1] != K)
        fail(file, "dataset '" + std::string(name) + "' expected width " + std::to_string(K) + ", got " +
                       std::to_string(dims[1]));
    std::vector<std::array<T, K>> out(dims[0]);
    if (dims[0] > 0) {
        std::vector<T> flat(dims[0] * K);
        if (H5Dread(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()) < 0)
            fail(file, "failed to read dataset '" + std::string(name) + "'");
        for (hsize_t i = 0; i < dims[0]; ++i)
            for (std::size_t j = 0; j < K; ++j) out[i][j] = flat[i * K + j];
    }
    return out;
}

void require_bool_flag(const fs::path& file, uint8_t value, const char* name) {
    if (value != 0 && value != 1)
        fail(file, "invalid boolean flag value " + std::to_string(value) + " in '" + std::string(name) + "'");
}

// ---------------------------------------------------------------- JSON helpers

std::string escape_json(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (const unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                std::ostringstream oss;
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                out += oss.str();
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
    }
    out.push_back('"');
    return out;
}

std::string format_double(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    return oss.str();
}

std::string format_warnings(const std::vector<std::string>& warnings) {
    std::string out = "[";
    for (std::size_t i = 0; i < warnings.size(); ++i) {
        if (i > 0) out += ", ";
        out += escape_json(warnings[i]);
    }
    out += "]";
    return out;
}

void write_text_file(const fs::path& file, const std::string& content) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) fail(file, "cannot open file for writing");
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
    out.close();
    if (!out) fail(file, "failed to write or close file");
}

}   // namespace

// ---------------------------------------------------------------- MeshProperties

void write_mesh_properties(const fs::path& file, const MeshProperties& mesh) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "MeshProperties");

    write_dataset_2d(file, h5.id, "nodes", mesh.nodes);

    {
        const H5Handle g = create_group(file, h5.id, "elements");
        std::vector<std::array<uint32_t, 4>> nodes;
        std::vector<uint16_t> materials;
        nodes.reserve(mesh.elements.size());
        materials.reserve(mesh.elements.size());
        for (const auto& e : mesh.elements) {
            nodes.push_back(e.node);
            materials.push_back(e.material);
        }
        write_dataset_2d(file, g.id, "node", nodes);
        write_dataset_1d(file, g.id, "material", materials);
    }

    {
        const H5Handle g = create_group(file, h5.id, "boundary_elements");
        std::vector<std::array<uint32_t, 3>> nodes;
        std::vector<uint16_t> bcs;
        nodes.reserve(mesh.boundary_elements.size());
        bcs.reserve(mesh.boundary_elements.size());
        for (const auto& be : mesh.boundary_elements) {
            nodes.push_back(be.node);
            bcs.push_back(be.boundary_condition);
        }
        write_dataset_2d(file, g.id, "node", nodes);
        write_dataset_1d(file, g.id, "boundary_condition", bcs);
    }

    {
        const H5Handle g = create_group(file, h5.id, "boundary_conditions");
        const std::size_t n = mesh.boundary_conditions.size();
        std::vector<uint8_t> types;
        std::vector<double> values;
        std::vector<uint8_t> has_alpha;
        std::vector<double> alphas;
        types.reserve(n);
        values.reserve(n);
        has_alpha.reserve(n);
        alphas.reserve(n);
        for (const auto& bc : mesh.boundary_conditions) {
            types.push_back(static_cast<uint8_t>(bc.type));
            values.push_back(bc.value);
            has_alpha.push_back(bc.alpha.has_value() ? 1 : 0);
            alphas.push_back(bc.alpha.value_or(0.0));
        }
        write_dataset_1d(file, g.id, "type", types);
        write_dataset_1d(file, g.id, "value", values);
        write_dataset_1d(file, g.id, "has_alpha", has_alpha);
        write_dataset_1d(file, g.id, "alpha", alphas);
    }

    {
        const H5Handle g = create_group(file, h5.id, "materials");
        const std::size_t n = mesh.materials.size();
        std::vector<uint8_t> name_bytes;
        std::vector<uint64_t> name_offsets;
        std::vector<double> eps_r;
        std::vector<double> screening_length;
        std::vector<uint8_t> is_metal;
        std::vector<uint8_t> has_voltage;
        std::vector<double> voltage;
        name_offsets.reserve(n + 1);
        eps_r.reserve(n);
        screening_length.reserve(n);
        is_metal.reserve(n);
        has_voltage.reserve(n);
        voltage.reserve(n);
        name_offsets.push_back(0);
        for (const auto& m : mesh.materials) {
            name_bytes.insert(name_bytes.end(), m.name.begin(), m.name.end());
            name_offsets.push_back(name_bytes.size());
            eps_r.push_back(m.eps_r);
            screening_length.push_back(m.screening_length);
            is_metal.push_back(m.is_metal ? 1 : 0);
            has_voltage.push_back(m.voltage.has_value() ? 1 : 0);
            voltage.push_back(m.voltage.value_or(0.0));
        }
        write_dataset_1d(file, g.id, "name_bytes", name_bytes);
        write_dataset_1d(file, g.id, "name_offsets", name_offsets);
        write_dataset_1d(file, g.id, "eps_r", eps_r);
        write_dataset_1d(file, g.id, "screening_length", screening_length);
        write_dataset_1d(file, g.id, "is_metal", is_metal);
        write_dataset_1d(file, g.id, "has_voltage", has_voltage);
        write_dataset_1d(file, g.id, "voltage", voltage);
    }

    {
        const H5Handle g = create_group(file, h5.id, "charges");
        std::vector<std::array<double, 3>> positions;
        std::vector<int32_t> values;
        positions.reserve(mesh.charges.size());
        values.reserve(mesh.charges.size());
        for (const auto& c : mesh.charges) {
            positions.push_back(c.position);
            values.push_back(c.value);
        }
        write_dataset_2d(file, g.id, "position", positions);
        write_dataset_1d(file, g.id, "value", values);
    }

    {
        std::vector<std::array<uint32_t, 4>> neighbours;
        neighbours.reserve(mesh.neighbours.size());
        for (const auto& nb : mesh.neighbours) {
            std::array<uint32_t, 4> row{};
            for (std::size_t i = 0; i < 4; ++i) {
                if (nb[i] && *nb[i] == kNoneU32)
                    fail(file, "neighbour index collides with sentinel UINT32_MAX");
                row[i] = nb[i].value_or(kNoneU32);
            }
            neighbours.push_back(row);
        }
        write_dataset_2d(file, h5.id, "neighbours", neighbours);
    }

    h5.close_file_checked(file);
}

MeshProperties read_mesh_properties(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "MeshProperties");

    MeshProperties mesh;
    mesh.nodes = read_dataset_2d<double, 3>(file, h5.id, "nodes");

    {
        const H5Handle g = open_group(file, h5.id, "boundary_conditions");
        const auto types = read_dataset_1d<uint8_t>(file, g.id, "type");
        const auto values = read_dataset_1d<double>(file, g.id, "value");
        const auto has_alpha = read_dataset_1d<uint8_t>(file, g.id, "has_alpha");
        const auto alphas = read_dataset_1d<double>(file, g.id, "alpha");
        const std::size_t n = types.size();
        if (values.size() != n || has_alpha.size() != n || alphas.size() != n)
            fail(file, "boundary_conditions dataset size mismatch");
        mesh.boundary_conditions.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (types[i] > static_cast<uint8_t>(MeshProperties::BcType::Robin))
                fail(file, "invalid BcType value " + std::to_string(types[i]));
            require_bool_flag(file, has_alpha[i], "boundary_conditions/has_alpha");
            mesh.boundary_conditions[i] = {
                static_cast<MeshProperties::BcType>(types[i]),
                values[i],
                has_alpha[i] ? std::optional<double>(alphas[i]) : std::nullopt,
            };
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "materials");
        const auto name_bytes = read_dataset_1d<uint8_t>(file, g.id, "name_bytes");
        const auto name_offsets = read_dataset_1d<uint64_t>(file, g.id, "name_offsets");
        const auto eps_r = read_dataset_1d<double>(file, g.id, "eps_r");
        const auto screening_length = read_dataset_1d<double>(file, g.id, "screening_length");
        const auto is_metal = read_dataset_1d<uint8_t>(file, g.id, "is_metal");
        const auto has_voltage = read_dataset_1d<uint8_t>(file, g.id, "has_voltage");
        const auto voltage = read_dataset_1d<double>(file, g.id, "voltage");
        const std::size_t n = eps_r.size();
        if (name_offsets.size() != n + 1 || screening_length.size() != n || is_metal.size() != n ||
            has_voltage.size() != n || voltage.size() != n)
            fail(file, "materials dataset size mismatch");
        if (name_offsets.front() != 0 || name_offsets.back() != name_bytes.size())
            fail(file, "materials/name_offsets must span materials/name_bytes from 0 to size");
        mesh.materials.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t begin = name_offsets[i], end = name_offsets[i + 1];
            if (begin > end || end > name_bytes.size()) fail(file, "invalid materials/name_offsets");
            require_bool_flag(file, is_metal[i], "materials/is_metal");
            require_bool_flag(file, has_voltage[i], "materials/has_voltage");
            mesh.materials[i] = {
                std::string(reinterpret_cast<const char*>(name_bytes.data() + begin),
                            static_cast<std::size_t>(end - begin)),
                eps_r[i],
                screening_length[i],
                is_metal[i] != 0,
                has_voltage[i] ? std::optional<double>(voltage[i]) : std::nullopt,
            };
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "elements");
        const auto nodes = read_dataset_2d<uint32_t, 4>(file, g.id, "node");
        const auto materials = read_dataset_1d<uint16_t>(file, g.id, "material");
        if (nodes.size() != materials.size()) fail(file, "elements/node and elements/material size mismatch");
        mesh.elements.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (const uint32_t n_idx : nodes[i]) {
                if (n_idx >= mesh.nodes.size())
                    fail(file, "elements/node references out-of-range node index " + std::to_string(n_idx));
            }
            if (materials[i] >= mesh.materials.size())
                fail(file, "elements/material references out-of-range material index " + std::to_string(materials[i]));
            mesh.elements[i] = {nodes[i], materials[i]};
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "boundary_elements");
        const auto nodes = read_dataset_2d<uint32_t, 3>(file, g.id, "node");
        const auto bcs = read_dataset_1d<uint16_t>(file, g.id, "boundary_condition");
        if (nodes.size() != bcs.size())
            fail(file, "boundary_elements/node and boundary_elements/boundary_condition size mismatch");
        mesh.boundary_elements.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (const uint32_t n_idx : nodes[i]) {
                if (n_idx >= mesh.nodes.size())
                    fail(file, "boundary_elements/node references out-of-range node index " + std::to_string(n_idx));
            }
            if (bcs[i] >= mesh.boundary_conditions.size())
                fail(file, "boundary_elements/boundary_condition references out-of-range index " +
                               std::to_string(bcs[i]));
            mesh.boundary_elements[i] = {nodes[i], bcs[i]};
        }
    }

    {
        const H5Handle g = open_group(file, h5.id, "charges");
        const auto positions = read_dataset_2d<double, 3>(file, g.id, "position");
        const auto values = read_dataset_1d<int32_t>(file, g.id, "value");
        if (positions.size() != values.size()) fail(file, "charges dataset size mismatch");
        mesh.charges.resize(positions.size());
        for (std::size_t i = 0; i < positions.size(); ++i) mesh.charges[i] = {positions[i], values[i]};
    }

    {
        const auto raw = read_dataset_2d<uint32_t, 4>(file, h5.id, "neighbours");
        if (!raw.empty() && raw.size() != mesh.elements.size())
            fail(file, "neighbours length must be 0 or match elements count");
        mesh.neighbours.resize(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            for (std::size_t k = 0; k < 4; ++k) {
                if (raw[i][k] == kNoneU32) {
                    mesh.neighbours[i][k] = std::nullopt;
                } else if (raw[i][k] >= mesh.elements.size()) {
                    fail(file, "neighbours references out-of-range element index " + std::to_string(raw[i][k]));
                } else {
                    mesh.neighbours[i][k] = raw[i][k];
                }
            }
        }
    }

    return mesh;
}

// ---------------------------------------------------------------- FreeCharges

void write_free_charges(const fs::path& file, const FreeCharges& charges) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "FreeCharges");

    {
        const H5Handle g = create_group(file, h5.id, "electrons");
        const std::size_t n = charges.electrons.size();
        std::vector<std::array<double, 3>> position;
        std::vector<std::array<double, 3>> direction;
        std::vector<double> energy;
        std::vector<std::array<int32_t, 2>> pixel;
        position.reserve(n);
        direction.reserve(n);
        energy.reserve(n);
        pixel.reserve(n);
        for (const auto& e : charges.electrons) {
            position.push_back(e.position);
            direction.push_back(e.direction);
            energy.push_back(e.energy);
            pixel.push_back(e.pixel);
        }
        write_dataset_2d(file, g.id, "position", position);
        write_dataset_2d(file, g.id, "direction", direction);
        write_dataset_1d(file, g.id, "energy", energy);
        write_dataset_2d(file, g.id, "pixel", pixel);
    }

    h5.close_file_checked(file);
}

FreeCharges read_free_charges(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "FreeCharges");

    const H5Handle g = open_group(file, h5.id, "electrons");
    const auto position = read_dataset_2d<double, 3>(file, g.id, "position");
    const auto direction = read_dataset_2d<double, 3>(file, g.id, "direction");
    const auto energy = read_dataset_1d<double>(file, g.id, "energy");
    const auto pixel = read_dataset_2d<int32_t, 2>(file, g.id, "pixel");
    const std::size_t n = position.size();
    if (direction.size() != n || energy.size() != n || pixel.size() != n)
        fail(file, "electrons dataset size mismatch");

    FreeCharges result;
    result.electrons.resize(n);
    for (std::size_t i = 0; i < n; ++i) result.electrons[i] = {position[i], direction[i], energy[i], pixel[i]};
    return result;
}

// ---------------------------------------------------------------- AssembledProblem

void write_assembled_problem(const fs::path& file, const AssembledProblem& problem) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "AssembledProblem");

    {
        const H5Handle g_a = create_group(file, h5.id, "A");
        Eigen::SparseMatrix<double> compressed = problem.A;
        compressed.makeCompressed();

        const uint64_t rows = static_cast<uint64_t>(compressed.rows());
        const uint64_t cols = static_cast<uint64_t>(compressed.cols());
        const std::size_t nnz = static_cast<std::size_t>(compressed.nonZeros());

        write_string_attr(file, g_a.id, "storage_order", "col_major");
        write_u64_attr(file, g_a.id, "rows", rows);
        write_u64_attr(file, g_a.id, "cols", cols);

        std::vector<int32_t> outer_index(cols + 1, 0);
        if (cols > 0) {
            const auto* ptr = compressed.outerIndexPtr();
            for (uint64_t j = 0; j <= cols; ++j) outer_index[j] = static_cast<int32_t>(ptr[j]);
        }
        std::vector<int32_t> inner_index(nnz);
        std::vector<double> values(nnz);
        if (nnz > 0) {
            const auto* iptr = compressed.innerIndexPtr();
            const double* vptr = compressed.valuePtr();
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
    if (problem.A.rows() > 0 && static_cast<std::size_t>(problem.A.rows()) != problem.f.size())
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

// ---------------------------------------------------------------- SolverResult

void write_solver_result(const fs::path& file, const SolverResult& result) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "SolverResult");
    write_dataset_1d(file, h5.id, "x", result.x);
    h5.close_file_checked(file);
}

SolverResult read_solver_result(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "SolverResult");
    SolverResult result;
    result.x = read_dataset_1d<double>(file, h5.id, "x");
    return result;
}

// ---------------------------------------------------------------- NodalField

void write_nodal_field(const fs::path& file, const NodalField& field) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "NodalField");
    write_dataset_1d(file, h5.id, "u", field.u);

    {
        const H5Handle g = create_group(file, h5.id, "point_sources");
        const std::size_t n = field.point_sources.size();
        std::vector<std::array<double, 3>> positions;
        std::vector<double> values;
        std::vector<uint32_t> elements;
        positions.reserve(n);
        values.reserve(n);
        elements.reserve(n);
        for (const auto& ps : field.point_sources) {
            if (ps.element && *ps.element == kNoneU32)
                fail(file, "point_source element collides with sentinel UINT32_MAX");
            positions.push_back(ps.position);
            values.push_back(ps.value);
            elements.push_back(ps.element.value_or(kNoneU32));
        }
        write_dataset_2d(file, g.id, "position", positions);
        write_dataset_1d(file, g.id, "value", values);
        write_dataset_1d(file, g.id, "element", elements);
    }

    h5.close_file_checked(file);
}

NodalField read_nodal_field(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "NodalField");

    NodalField field;
    field.u = read_dataset_1d<double>(file, h5.id, "u");
    {
        const H5Handle g = open_group(file, h5.id, "point_sources");
        const auto positions = read_dataset_2d<double, 3>(file, g.id, "position");
        const auto values = read_dataset_1d<double>(file, g.id, "value");
        const auto elements = read_dataset_1d<uint32_t>(file, g.id, "element");
        const std::size_t n = positions.size();
        if (values.size() != n || elements.size() != n) fail(file, "point_sources dataset size mismatch");
        field.point_sources.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            field.point_sources[i] = {
                positions[i],
                values[i],
                elements[i] == kNoneU32 ? std::nullopt : std::optional<uint32_t>(elements[i]),
            };
        }
    }
    return field;
}

// ---------------------------------------------------------------- Parents

void write_parents(const fs::path& file, const Parents& parents) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "Parents");
    write_dataset_1d(file, h5.id, "elements", parents.elements);
    write_dataset_2d(file, h5.id, "nodes", parents.nodes);
    h5.close_file_checked(file);
}

Parents read_parents(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "Parents");
    Parents parents;
    parents.elements = read_dataset_1d<uint32_t>(file, h5.id, "elements");
    parents.nodes = read_dataset_2d<uint32_t, 2>(file, h5.id, "nodes");
    return parents;
}

// ---------------------------------------------------------------- RefinementHistory

void write_refinement_history(const fs::path& file, const RefinementHistory& history) {
    H5Handle h5 = create_deterministic_h5(file);
    write_schema_header(file, h5.id, "RefinementHistory");

    {
        const H5Handle g = create_group(file, h5.id, "families");
        std::vector<std::array<uint32_t, 4>> parent;
        std::vector<std::array<uint32_t, 6>> midpoint;
        parent.reserve(history.families.size());
        midpoint.reserve(history.families.size());
        for (const auto& fam : history.families) {
            parent.push_back(fam.parent);
            std::array<uint32_t, 6> mp{};
            for (std::size_t k = 0; k < 6; ++k) {
                if (fam.midpoint[k] && *fam.midpoint[k] == kNoneU32)
                    fail(file, "family midpoint collides with sentinel UINT32_MAX");
                mp[k] = fam.midpoint[k].value_or(kNoneU32);
            }
            midpoint.push_back(mp);
        }
        write_dataset_2d(file, g.id, "parent", parent);
        write_dataset_2d(file, g.id, "midpoint", midpoint);
    }

    std::vector<uint32_t> family_raw;
    family_raw.reserve(history.family.size());
    for (const auto& f : history.family) {
        if (f && *f == kNoneU32) fail(file, "family index collides with sentinel UINT32_MAX");
        family_raw.push_back(f.value_or(kNoneU32));
    }
    write_dataset_1d(file, h5.id, "family", family_raw);

    h5.close_file_checked(file);
}

RefinementHistory read_refinement_history(const fs::path& file) {
    const H5Handle h5 = open_readonly_h5(file);
    verify_schema_header(file, h5.id, "RefinementHistory");

    RefinementHistory history;
    {
        const H5Handle g = open_group(file, h5.id, "families");
        const auto parent = read_dataset_2d<uint32_t, 4>(file, g.id, "parent");
        const auto midpoint = read_dataset_2d<uint32_t, 6>(file, g.id, "midpoint");
        if (parent.size() != midpoint.size()) fail(file, "families/parent and families/midpoint size mismatch");
        history.families.resize(parent.size());
        for (std::size_t i = 0; i < parent.size(); ++i) {
            history.families[i].parent = parent[i];
            for (std::size_t k = 0; k < 6; ++k) {
                history.families[i].midpoint[k] =
                    midpoint[i][k] == kNoneU32 ? std::nullopt : std::optional<uint32_t>(midpoint[i][k]);
            }
        }
    }

    const auto family_raw = read_dataset_1d<uint32_t>(file, h5.id, "family");
    history.family.resize(family_raw.size());
    for (std::size_t i = 0; i < family_raw.size(); ++i) {
        if (family_raw[i] == kNoneU32) {
            history.family[i] = std::nullopt;
        } else if (family_raw[i] >= history.families.size()) {
            fail(file, "family references out-of-range family index " + std::to_string(family_raw[i]));
        } else {
            history.family[i] = family_raw[i];
        }
    }
    return history;
}

// ---------------------------------------------------------------- Info JSON sidecars

void write_info_json(const fs::path& file, const MeshImportResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"MeshImportResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"time_read\": " << format_double(info.time_read) << ",\n"
        << "  \"time_geometry\": " << format_double(info.time_geometry) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const FreeChargesImportResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"FreeChargesImportResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const AssembledProblem::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"AssembledProblem::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"time_assemble_A\": " << format_double(info.time_assemble_A) << ",\n"
        << "  \"time_assemble_f\": " << format_double(info.time_assemble_f) << ",\n"
        << "  \"nnz\": " << info.nnz << ",\n"
        << "  \"n_dirichlet\": " << info.n_dirichlet << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, const SolverResult::Info& info) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"SolverResult::Info\",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"iterations\": " << info.iterations << ",\n"
        << "  \"residual\": " << format_double(info.residual) << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << "\n"
        << "}\n";
    write_text_file(file, oss.str());
}

void write_info_json(const fs::path& file, AdaptationStatus status, const AdaptationResult::Info& info) {
    auto status_str = [](AdaptationStatus s) -> const char* {
        switch (s) {
        case AdaptationStatus::adapted: return "adapted";
        case AdaptationStatus::converged: return "converged";
        case AdaptationStatus::limit_reached: return "limit_reached";
        case AdaptationStatus::unchanged: return "unchanged";
        }
        return "unknown";
    };

    std::ostringstream oss;
    oss << "{\n"
        << "  \"type\": \"AdaptationResult::Info\",\n"
        << "  \"status\": " << escape_json(status_str(status)) << ",\n"
        << "  \"adapter\": " << escape_json(info.adapter) << ",\n"
        << "  \"time\": " << format_double(info.time) << ",\n"
        << "  \"elements_before\": " << info.elements_before << ",\n"
        << "  \"elements_after\": " << info.elements_after << ",\n"
        << "  \"error_estimate\": " << (info.error_estimate ? format_double(*info.error_estimate) : "null") << ",\n"
        << "  \"warnings\": " << format_warnings(info.warnings) << ",\n"
        << "  \"stages\": [";
    for (std::size_t i = 0; i < info.stages.size(); ++i) {
        const auto& st = info.stages[i];
        if (i > 0) oss << ",";
        oss << "\n    {\n"
            << "      \"stage\": " << escape_json(st.stage) << ",\n"
            << "      \"method\": " << escape_json(st.method) << ",\n"
            << "      \"time\": " << format_double(st.time) << ",\n"
            << "      \"metrics\": {";
        std::size_t m = 0;
        for (const auto& [k, v] : st.metrics) {
            if (m++ > 0) oss << ", ";
            oss << escape_json(k) << ": " << format_double(v);
        }
        oss << "},\n"
            << "      \"warnings\": " << format_warnings(st.warnings) << "\n"
            << "    }";
    }
    if (!info.stages.empty()) oss << "\n  ";
    oss << "]\n}\n";
    write_text_file(file, oss.str());
}

}   // namespace poisson::io
