#pragma once

// Shared plumbing of the verification CLIs: flag parsing, the aliasing checks of the stage
// CLI contract (cli/architecture.md), and publishing several outputs only after all of them
// were written.

#include "cli/common/cli_utils.h"

#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poisson::cli {

// Every flag takes one value. Throws std::runtime_error for an unknown flag, a missing value
// or a repeated flag. "-h" / "--help" gives {"--help": ""}.
inline std::map<std::string, std::filesystem::path> parse_flags(int argc, char** argv, const std::set<std::string>& allowed) {
    std::map<std::string, std::filesystem::path> out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") return {{"--help", ""}};
        if (!allowed.contains(arg)) throw std::runtime_error("unknown argument '" + arg + "'");
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
        if (out.contains(arg)) throw std::runtime_error(arg + " is given twice");
        out[arg] = argv[++i];
    }
    return out;
}

// Outputs must not alias inputs or each other (including symlinks and hard links), and must
// not be directories (checked up front, so a run does not publish some outputs and then fail
// to replace a directory with the next).
inline void reject_aliasing(const std::vector<std::filesystem::path>& inputs,
                            const std::vector<std::pair<std::string, std::filesystem::path>>& outputs) {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::error_code ec;
        if (std::filesystem::is_directory(outputs[i].second, ec))
            throw std::runtime_error(outputs[i].first + " '" + outputs[i].second.string() + "' is a directory");
        for (const auto& input : inputs)
            if (paths_alias(outputs[i].second, input))
                throw std::runtime_error(outputs[i].first + " aliases input file '" + input.string() + "'");
        for (std::size_t j = i + 1; j < outputs.size(); ++j)
            if (paths_alias(outputs[i].second, outputs[j].second))
                throw std::runtime_error(outputs[i].first + " aliases " + outputs[j].first);
    }
}

// Each output is written to a sibling temp file; publish() renames them all into place.
// Temp files of an unpublished set are removed.
class StagedOutputs {
public:
    template <class Write>
    void add(const std::filesystem::path& target, const std::string& tag, Write&& write) {
        Entry& e = entries_.emplace_back();
        e.target = target;
        e.staged = make_staging_path(target, tag);
        e.guard.path = e.staged;
        write(e.staged);
    }

    void publish() {
        for (Entry& e : entries_) {
            std::filesystem::rename(e.staged, e.target);
            e.guard.release();
        }
    }

private:
    struct Entry {
        std::filesystem::path target, staged;
        TempFileCleanup       guard{{}};
    };
    std::deque<Entry> entries_;   // deque: adding never moves (and so never deletes) earlier entries
};

}   // namespace poisson::cli
