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
