# Stage CLIs and Deterministic HDF5 Serialization

## Folder Layout

Each stage variant lives in its own subfolder under `cli/<stage>/<variant>.cpp` and builds as
an independent executable in `build/`:

```text
cli/
  common/
    cli_utils.h             shared path-aliasing checks and atomic staging helpers
    simple_json.h           zero-dependency strict JSON parser for --config files
  importer/
    poisson_import.cpp      -> build/poisson_import
  amr/
    poisson_adapt.cpp       -> build/poisson_adapt
  assembler/                (future: e.g. p1_assemble.cpp, p2_assemble.cpp)
  solver/                   (future: e.g. cg_solve.cpp, amg_solve.cpp)
  driver/                   (future: poisson_drive.cpp once PoissonDriver is complete)
```

In `CMakeLists.txt`, shared stage I/O and hashing live in `poisson_io` (`src/serialization.cpp`,
`src/blake3_ffi.cpp`). Each stage CLI links only its own module library (`poisson_importer`,
`poisson_amr`, ...) plus `poisson_io` so editing one stage does not change the compiled binary
hash of unrelated stages.

## Stage CLI Contract

Every stage executable:
1. Reads its inputs from explicit CLI path flags (never hardcoded paths). For stages with many
   nested component options (such as AMR or `PoissonDriver`), prefer a single `--config
   <settings.json>` flag rather than dozens of flat flags.
2. Writes deterministic scientific output data to `--out <file.h5>` (using `poisson::io::write_*`).
3. Writes run-specific metadata (`Info` timings and warnings) separately to `--info-out <info.json>`.
4. Rejects path aliasing between outputs and inputs (including symlinks/hard links) and publishes
   outputs via sibling temp files + checked flush/close + atomic `std::filesystem::rename`.
5. Returns exit code `0` on success and non-zero with a clear `stderr` message on failure.

## Deterministic HDF5 Serialization (`include/serialization.h`, `src/serialization.cpp`)

- Supports `MeshProperties`, `FreeCharges`, `AssembledProblem` (including compressed
  `Eigen::SparseMatrix<double>` `A`), and `SolverResult`.
- Disables HDF5 object timestamp tracking (`H5Pset_obj_track_times(..., false)`) on the root
  group, all subgroups, and all datasets, and pins HDF5 format bounds (`H5F_LIBVER_V18`).
- Uses explicit little-endian numeric file types (`H5T_IEEE_F64LE`, `H5T_STD_U32LE`, etc.),
  flat Struct-of-Arrays datasets with zero struct padding, and UTF-8 `name_bytes` +
  `name_offsets` for strings.
- Validates scalar schema attributes, exact on-disk numeric types, boolean flags, and
  cross-array index bounds on read.

## Adding a New Stage CLI (Teammate Recipe)

When a new core C++ stage (e.g. `Assembly` or `Solver`) is ready, wiring it into the cache and
`compare(...)` runner takes three small files:

### 1. Create `cli/<stage>/<variant>.cpp`

```cpp
#include "cli/common/cli_utils.h"
#include "cli/common/simple_json.h"
#include "serialization.h"
#include "your_stage.h"

int main(int argc, char** argv) {
    // 1. Parse --in, --config, --out, --info-out (reject missing flags)
    // 2. Check path aliasing:
    //    poisson::cli::reject_output_aliasing(out_path, {in_path}, "--out");
    //    poisson::cli::reject_same_path(info_out_path, out_path, "--info-out", "--out");
    // 3. Read input & optional JSON config:
    //    auto problem = poisson::io::read_assembled_problem_hdf5(in_path);
    //    auto cfg = poisson::cli::parse_json_file(config_path);
    // 4. Run your core C++ stage:
    //    SolverResult result = solve(problem, ...);
    // 5. Publish atomically via sibling temp files:
    //    poisson::cli::atomic_write_file(out_path, [&](const auto& tmp) {
    //        poisson::io::write_solver_result_hdf5(result, tmp);
    //    });
    //    if (!info_out_path.empty()) {
    //        poisson::cli::atomic_write_file(info_out_path, [&](const auto& tmp) {
    //            poisson::io::write_info_json(result.info, tmp);
    //        });
    //    }
}
```

### 2. Add an isolated target in `CMakeLists.txt`

```cmake
add_executable(poisson_solve_cg cli/solver/cg_solve.cpp)
target_include_directories(poisson_solve_cg PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(poisson_solve_cg PRIVATE poisson_solver)  # links poisson_io + solver only
target_compile_options(poisson_solve_cg PRIVATE -Wall -Wextra -Wpedantic)
```

Linking only `poisson_io` + your own stage library guarantees that editing unrelated stages does
not change this binary's BLAKE3 hash.

### 3. Register in Python with `Stage.from_cli(...)`

```python
from poisson_runner import Stage

cg_solve = Stage.from_cli(
    "solve",
    binary="poisson_solve_cg",
    input_flags={"problem": "--in"},
    config_flag="--config",
    output_flags={"out": "--out"},
    output_specs={"out": ".h5"},
    info_flag="--info-out",
)
```

Now any experiment can call `run.run_stage(cg_solve, inputs={"problem": ap}, settings={...})`
with automatic BLAKE3 caching, `repeats`/`warmup`, `dry_run`, and SLURM sharding.
