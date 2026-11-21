#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>

#ifdef __unix__
#include <unistd.h>
#endif

namespace poisson::cli {

inline bool paths_alias(const std::filesystem::path& a, const std::filesystem::path& b) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(a, ec) && !ec && fs::exists(b, ec) && !ec) {
        const bool eq = fs::equivalent(a, b, ec);
        if (!ec && eq) return true;
    }
    const fs::path ca = fs::weakly_canonical(fs::absolute(a), ec);
    if (ec) return false;
    const fs::path cb = fs::weakly_canonical(fs::absolute(b), ec);
    if (ec) return false;
    return ca == cb;
}

inline std::filesystem::path make_staging_path(const std::filesystem::path& target, std::string_view tag) {
    namespace fs = std::filesystem;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef __unix__
    const long pid = static_cast<long>(::getpid());
#else
    const long pid = 0;
#endif
    const fs::path parent = target.has_parent_path() ? target.parent_path() : fs::path(".");
    const std::string name =
        "." + target.filename().string() + "." + std::string(tag) + "." + std::to_string(pid) + "." +
        std::to_string(now) + ".tmp";
    return parent / name;
}

struct TempFileCleanup {
    std::filesystem::path path;
    bool active = true;
    ~TempFileCleanup() {
        if (active && !path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    void release() { active = false; }
};

}   // namespace poisson::cli
