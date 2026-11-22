#include "h5_common.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace poisson::io::detail {

[[noreturn]] void fail(const fs::path& file, const std::string& message) {
    throw std::runtime_error(file.string() + ": " + message);
}

void H5Handle::close_file_checked(const fs::path& file) {
    if (id < 0) return;
    const herr_t flush_err = H5Fflush(id, H5F_SCOPE_GLOBAL);
    const herr_t close_err = close(id);
    id = -1;
    if (flush_err < 0 || close_err < 0) fail(file, "failed to flush or close HDF5 file");
}

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

void require_bool_flag(const fs::path& file, uint8_t value, const char* name) {
    if (value != 0 && value != 1)
        fail(file, "invalid boolean flag value " + std::to_string(value) + " in '" + std::string(name) + "'");
}

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
    oss << std::setprecision(17) << canonicalize_scalar(v);
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

}   // namespace poisson::io::detail
