#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

// C ABI exported by libpoisson_blake3 for Python ctypes and C callers.
extern "C" {

void poisson_blake3_bytes(const uint8_t* data, std::size_t len, uint8_t out[32]);

// Hashes a file in streaming chunks. Returns 0 on success, or errno (or -1) on I/O error.
int poisson_blake3_file(const char* path, uint8_t out[32]);

const char* poisson_blake3_version();

}

namespace poisson {

std::array<uint8_t, 32> blake3_bytes(const void* data, std::size_t len);
std::string blake3_bytes_hex(const void* data, std::size_t len);
std::string blake3_file_hex(const std::filesystem::path& path);

}   // namespace poisson
