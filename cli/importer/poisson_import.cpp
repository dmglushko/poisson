#include "importer.h"
#include "serialization.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifdef __unix__
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

void print_usage(std::ostream& out) {
    out << "Usage:\n"
        << "  poisson_import --mesh <mesh.msh> [--material <mat.yaml|.mat>]... [--charges <file.cha>]\n"
        << "                 --out <mesh.h5> [--info-out <info.json>]\n"
        << "  poisson_import --primaries <file.pri> --out <free_charges.h5> [--info-out <info.json>]\n";
}

bool paths_alias(const fs::path& a, const fs::path& b) {
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

fs::path make_staging_path(const fs::path& target, std::string_view tag) {
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
    fs::path path;
    bool active = true;
    ~TempFileCleanup() {
        if (active && !path.empty()) {
            std::error_code ec;
            fs::remove(path, ec);
        }
    }
    void release() { active = false; }
};

}   // namespace

int main(int argc, char** argv) {
    std::optional<fs::path> mesh_file;
    std::vector<fs::path> material_files;
    std::optional<fs::path> charges_file;
    std::optional<fs::path> primaries_file;
    std::optional<fs::path> out_file;
    std::optional<fs::path> info_out_file;

    auto require_value = [&](int& i, std::string_view flag) -> fs::path {
        if (i + 1 >= argc) {
            std::cerr << "error: missing value for " << flag << "\n";
            std::exit(EXIT_FAILURE);
        }
        return fs::path(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        } else if (arg == "--mesh") {
            mesh_file = require_value(i, arg);
        } else if (arg == "--material") {
            material_files.push_back(require_value(i, arg));
        } else if (arg == "--charges") {
            charges_file = require_value(i, arg);
        } else if (arg == "--primaries") {
            primaries_file = require_value(i, arg);
        } else if (arg == "--out") {
            out_file = require_value(i, arg);
        } else if (arg == "--info-out") {
            info_out_file = require_value(i, arg);
        } else {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    if (mesh_file.has_value() == primaries_file.has_value()) {
        std::cerr << "error: specify exactly one of --mesh or --primaries\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    if (!out_file) {
        std::cerr << "error: --out is required\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    if (primaries_file && (!material_files.empty() || charges_file)) {
        std::cerr << "error: --material and --charges are only valid with --mesh\n";
        return EXIT_FAILURE;
    }

    // Reject path aliasing between outputs and any input (including symlinks and hard links).
    std::vector<fs::path> all_inputs;
    if (mesh_file) all_inputs.push_back(*mesh_file);
    for (const auto& mat : material_files) all_inputs.push_back(mat);
    if (charges_file) all_inputs.push_back(*charges_file);
    if (primaries_file) all_inputs.push_back(*primaries_file);

    for (const auto& inp : all_inputs) {
        if (paths_alias(*out_file, inp)) {
            std::cerr << "error: --out aliases input file '" << inp.string() << "'\n";
            return EXIT_FAILURE;
        }
        if (info_out_file && paths_alias(*info_out_file, inp)) {
            std::cerr << "error: --info-out aliases input file '" << inp.string() << "'\n";
            return EXIT_FAILURE;
        }
    }
    if (info_out_file && paths_alias(*out_file, *info_out_file)) {
        std::cerr << "error: --info-out must not alias --out\n";
        return EXIT_FAILURE;
    }

    try {
        const Importer importer;
        const fs::path staged_out = make_staging_path(*out_file, "out");
        TempFileCleanup out_guard{staged_out};
        fs::path staged_info;
        TempFileCleanup info_guard{{}};

        if (mesh_file) {
            const MeshImportResult imported = importer.import_mesh(*mesh_file, material_files, charges_file);
            poisson::io::write_mesh_properties(staged_out, imported.mesh);
            if (info_out_file) {
                staged_info = make_staging_path(*info_out_file, "info");
                info_guard.path = staged_info;
                poisson::io::write_info_json(staged_info, imported.info);
            }
        } else {
            const FreeChargesImportResult imported = importer.import_free_charges(*primaries_file);
            poisson::io::write_free_charges(staged_out, imported.charges);
            if (info_out_file) {
                staged_info = make_staging_path(*info_out_file, "info");
                info_guard.path = staged_info;
                poisson::io::write_info_json(staged_info, imported.info);
            }
        }

        fs::rename(staged_out, *out_file);
        out_guard.release();
        if (info_out_file) {
            fs::rename(staged_info, *info_out_file);
            info_guard.release();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
