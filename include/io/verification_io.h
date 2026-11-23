#pragma once

#include "verification/case_inputs.h"

#include <filesystem>

// JSON Info sidecar I/O for the verification stage (the case inputs themselves are text and
// binary files written by poisson_case).
namespace poisson::io {

void write_info_json(const std::filesystem::path& file, const CaseInputsResult::Info& info);

}   // namespace poisson::io
