#pragma once

#include <hdf5.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace poisson::io::detail {

namespace fs = std::filesystem;

inline constexpr uint32_t kSchemaVersion = 1;
inline constexpr uint32_t kNoneU32 = std::numeric_limits<uint32_t>::max();

[[noreturn]] void fail(const fs::path& file, const std::string& message);

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

    void close_file_checked(const fs::path& file);
};

H5Handle create_deterministic_h5(const fs::path& file);
H5Handle open_readonly_h5(const fs::path& file);
H5Handle create_group(const fs::path& file, hid_t parent, const char* name);
H5Handle open_group(const fs::path& file, hid_t parent, const char* name);

void write_string_attr(const fs::path& file, hid_t obj, const char* name, std::string_view value);
std::string read_string_attr(const fs::path& file, hid_t obj, const char* name);

void write_u32_attr(const fs::path& file, hid_t obj, const char* name, uint32_t value);
uint32_t read_u32_attr(const fs::path& file, hid_t obj, const char* name);

void write_u64_attr(const fs::path& file, hid_t obj, const char* name, uint64_t value);
uint64_t read_u64_attr(const fs::path& file, hid_t obj, const char* name);

void write_schema_header(const fs::path& file, hid_t h5, std::string_view schema);
void verify_schema_header(const fs::path& file, hid_t h5, std::string_view expected_schema);

void require_bool_flag(const fs::path& file, uint8_t value, const char* name);

std::string escape_json(std::string_view s);
std::string format_double(double v);
std::string format_warnings(const std::vector<std::string>& warnings);
void write_text_file(const fs::path& file, const std::string& content);

template <typename T>
inline T canonicalize_scalar(T value) {
    if constexpr (std::is_same_v<T, double>) {
        // Normalize IEEE -0.0 to +0.0 so byte-level BLAKE3 hashes stay deterministic.
        return value == 0.0 ? 0.0 : value;
    } else {
        return value;
    }
}

template <typename T>
hid_t h5_file_type();
template <typename T>
hid_t h5_mem_type();
template <typename T>
const char* h5_type_name();

template <> inline hid_t h5_file_type<double>() { return H5T_IEEE_F64LE; }
template <> inline hid_t h5_mem_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> inline const char* h5_type_name<double>() { return "float64"; }

template <> inline hid_t h5_file_type<uint64_t>() { return H5T_STD_U64LE; }
template <> inline hid_t h5_mem_type<uint64_t>() { return H5T_NATIVE_UINT64; }
template <> inline const char* h5_type_name<uint64_t>() { return "uint64"; }

template <> inline hid_t h5_file_type<uint32_t>() { return H5T_STD_U32LE; }
template <> inline hid_t h5_mem_type<uint32_t>() { return H5T_NATIVE_UINT32; }
template <> inline const char* h5_type_name<uint32_t>() { return "uint32"; }

template <> inline hid_t h5_file_type<int32_t>() { return H5T_STD_I32LE; }
template <> inline hid_t h5_mem_type<int32_t>() { return H5T_NATIVE_INT32; }
template <> inline const char* h5_type_name<int32_t>() { return "int32"; }

template <> inline hid_t h5_file_type<uint16_t>() { return H5T_STD_U16LE; }
template <> inline hid_t h5_mem_type<uint16_t>() { return H5T_NATIVE_UINT16; }
template <> inline const char* h5_type_name<uint16_t>() { return "uint16"; }

template <> inline hid_t h5_file_type<uint8_t>() { return H5T_STD_U8LE; }
template <> inline hid_t h5_mem_type<uint8_t>() { return H5T_NATIVE_UINT8; }
template <> inline const char* h5_type_name<uint8_t>() { return "uint8"; }

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
        if constexpr (std::is_same_v<T, double>) {
            std::vector<double> clean;
            clean.reserve(data.size());
            for (double v : data) clean.push_back(canonicalize_scalar(v));
            if (H5Dwrite(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, clean.data()) < 0)
                fail(file, "failed to write dataset '" + std::string(name) + "'");
        } else {
            if (H5Dwrite(dset.id, h5_mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0)
                fail(file, "failed to write dataset '" + std::string(name) + "'");
        }
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
            for (std::size_t j = 0; j < K; ++j) flat.push_back(canonicalize_scalar(row[j]));
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

}   // namespace poisson::io::detail
