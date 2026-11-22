#include "blake3/blake3_hash.h"

#include "blake3.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>

extern "C" {

void poisson_blake3_bytes(const uint8_t* data, std::size_t len, uint8_t out[32]) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    if (data != nullptr && len > 0) {
        blake3_hasher_update(&hasher, data, len);
    }
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

int poisson_blake3_file(const char* path, uint8_t out[32]) {
    if (path == nullptr || out == nullptr) return EINVAL;

    constexpr std::size_t kChunkSize = 1024 * 1024;   // 1 MiB streaming buffer
    std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[kChunkSize]);
    if (!buffer) return ENOMEM;

    auto close_file = [](std::FILE* f) noexcept {
        if (f != nullptr) std::fclose(f);
    };
    std::unique_ptr<std::FILE, decltype(close_file)> fp(std::fopen(path, "rb"), close_file);
    if (!fp) return errno != 0 ? errno : -1;

    blake3_hasher hasher;
    blake3_hasher_init(&hasher);

    while (true) {
        const std::size_t n = std::fread(buffer.get(), 1, kChunkSize, fp.get());
        if (n > 0) {
            blake3_hasher_update(&hasher, buffer.get(), n);
        }
        if (n < kChunkSize) {
            if (std::ferror(fp.get())) {
                return errno != 0 ? errno : EIO;
            }
            break;
        }
    }
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    return 0;
}

const char* poisson_blake3_version() { return blake3_version(); }

}   // extern "C"

namespace poisson {
namespace {

std::string to_hex(const std::array<uint8_t, 32>& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '\0');
    for (std::size_t i = 0; i < 32; ++i) {
        out[2 * i] = kHex[digest[i] >> 4];
        out[2 * i + 1] = kHex[digest[i] & 0x0f];
    }
    return out;
}

}   // namespace

std::array<uint8_t, 32> blake3_bytes(const void* data, std::size_t len) {
    std::array<uint8_t, 32> out{};
    poisson_blake3_bytes(static_cast<const uint8_t*>(data), len, out.data());
    return out;
}

std::string blake3_bytes_hex(const void* data, std::size_t len) { return to_hex(blake3_bytes(data, len)); }

std::string blake3_file_hex(const std::filesystem::path& path) {
    std::array<uint8_t, 32> out{};
    const int err = poisson_blake3_file(path.string().c_str(), out.data());
    if (err != 0) {
        throw std::runtime_error(path.string() + ": failed to hash file: " + std::strerror(err));
    }
    return to_hex(out);
}

}   // namespace poisson
