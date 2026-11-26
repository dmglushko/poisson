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
  verification/
    case_config.h           case JSON -> VerificationCase (the only place naming case parameters)
    verification_cli.h      flag parsing, aliasing checks, staged multi-output publishing
    poisson_case.cpp        -> build/poisson_case
  assembler/                (future: e.g. p1_assemble.cpp, p2_assemble.cpp)
  solver/                   (future: e.g. cg_solve.cpp, amg_solve.cpp)
  driver/                   (future: poisson_drive.cpp once PoissonDriver is complete)
```

In `CMakeLists.txt`, core stage libraries (`poisson_common`, `poisson_importer`, `poisson_amr`,
`poisson_verification`)
stay free of HDF5 stage serialization and BLAKE3 hashing. Stage I/O lives in `poisson_io`,
split into one translation unit per data type (`src/io/h5_common.cpp`, `mesh_io.cpp`,
`free_charges_io.cpp`, `assembled_problem_io.cpp`, `solver_result_io.cpp`, `amr_io.cpp`,
`verification_io.cpp`). Each
stage CLI links only its own module library plus `poisson_io`, so the linker only pulls in the
serializers that stage calls and editing an unrelated stage's code or serializer does not change
other stage binaries' BLAKE3 hashes.

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

## `poisson_import` (`cli/importer/poisson_import.cpp`)

Imports either a mesh (with optional material and charge files) or a primary-electron `.pri` file:

```text
poisson_import --mesh <mesh.msh> [--material <mat.yaml|.mat>]... [--charges <file.cha>]
               [--screening on|off] --out <mesh.h5> [--info-out <info.json>]
poisson_import --primaries <file.pri> --out <free_charges.h5> [--info-out <info.json>]
```

## `poisson_adapt` (`cli/amr/poisson_adapt.cpp`)

One `ErrorBasedAdapter` cycle per call, always through the adapter itself (no separate code
path):

```text
poisson_adapt --mesh <mesh.h5> --solution <nodal_field.h5> --config <config.json>
              [--state <state.h5>] --out <mesh_out.h5>
              [--parents-out <parents.h5>] [--state-out <state.h5>]
              [--eta-out <eta.h5>] [--info-out <info.json>]
```

- `--state` / `--state-out` carry `ErrorBasedAdapter::State` (red-green history and the base
  error of `relative_tolerance`) from one call to the next; omit `--state` for an imported mesh.
- The outputs always describe the mesh to continue with: when the status is not `adapted`,
  `--out` is the input mesh, `--parents-out` is the identity map, and `--state-out` is the
  (possibly baseline-updated) input state. `--eta-out` holds `ErrorBasedExtras::eta` from
  `AdaptationResult::extras` for the input mesh (empty when nothing was estimated). The status
  is in `--info-out`.
- The Python `adapt_mesh` helper always requests all four `.h5` outputs.

## Verification CLI (`cli/verification/`)

`poisson_case` links only `poisson_verification` and `poisson_io` (see
[src/verification/architecture.md](../src/verification/architecture.md)):

```text
poisson_case --config <case.json> --geo-out <case.geo> --materials-out <case.yaml>
             --charges-out <case.cha> [--info-out <info.json>]
```

- `case.json` is `{"case": {"name": ..., "params": {...}}}`. It writes the case's input
  files, always all three (the `.cha` is empty without charges).
- Besides the aliasing checks, an output that is an existing directory is rejected before
  anything is written, so a failed run does not leave some outputs replaced.
- Comparing a solution with a case waits for the `Solution` format
  ([issue #6](https://github.com/dmglushko/poisson/issues/6)).

## Deterministic HDF5 Serialization (`include/io/`, `src/io/`)

- Supports `MeshProperties`, `FreeCharges`, `AssembledProblem` (including compressed
  `Eigen::SparseMatrix<double>` `A`), `SolverResult`, and AMR stage artifacts (`NodalField`,
  `Parents`, `RefinementHistory`, `AdaptationState` — `ErrorBasedAdapter::State`: group
  `/history` with the `RefinementHistory` layout plus `/initial_error` with 0 or 1 float64
  entries — and `ErrorIndicators` — `/eta`, float64 per tet), split into matching per-type headers/sources
  (`mesh_io.h/.cpp`, `free_charges_io.h/.cpp`, `assembled_problem_io.h/.cpp`,
  `solver_result_io.h/.cpp`, `amr_io.h/.cpp`, `verification_io.h/.cpp`).
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
#include "io/assembled_problem_io.h"
#include "io/solver_result_io.h"
#include "your_stage/your_stage.h"

int main(int argc, char** argv) {
    // 1. Parse --in, --config, --out, --info-out (reject missing flags)
    // 2. Check path aliasing:
    //    poisson::cli::reject_output_aliasing(out_path, {in_path}, "--out");
    //    poisson::cli::reject_same_path(info_out_path, out_path, "--info-out", "--out");
    // 3. Read input & optional JSON config:
    //    auto problem = poisson::io::read_assembled_problem(in_path);
    //    auto cfg = poisson::cli::parse_json_file(config_path);
    // 4. Run your core C++ stage:
    //    SolverResult result = solve(problem, ...);
    // 5. Publish atomically via sibling temp files:
    //    poisson::cli::atomic_write_file(out_path, [&](const auto& tmp) {
    //        poisson::io::write_solver_result(tmp, result);
    //    });
    //    if (!info_out_path.empty()) {
    //        poisson::cli::atomic_write_file(info_out_path, [&](const auto& tmp) {
    //            poisson::io::write_info_json(tmp, result.info);
    //        });
    //    }
}
```

### 2. Add an isolated target in `CMakeLists.txt`

```cmake
add_executable(poisson_solve_cg cli/solver/cg_solve.cpp)
target_include_directories(poisson_solve_cg PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(poisson_solve_cg PRIVATE poisson_solver poisson_io)
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
