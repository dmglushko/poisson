# `poisson_runner` Architecture

`poisson_runner` is the Python orchestration, content-addressed caching, and loop-free
`compare(...)` benchmarking layer built on top of the per-stage C++ CLI binaries.

## Module Layout

| Module | Responsibility |
|---|---|
| `driver.py` | `compare(...)` replay, `RunContext.run_stage` (inputs -> request key -> lookup -> reuse -> execute -> publish), `run_experiment`, SLURM sharding |
| `benchmark.py` | `measure(...)`: warmup + repeated executions, per-metric statistics, output determinism check |
| `report.py` | `StageTrace`, `BranchResult`, `ExperimentReport` (table, JSON, wide/long CSV) |
| `stages.py` | `Stage` / `Stage.from_cli`, code hashing (binary + shared libraries), project stage factories and helpers (`import_mesh`, `import_free_charges`, `adapt_mesh`) |
| `cache.py` | Two-layer content-addressed cache, locking, leases, eviction |
| `executor.py` | `Executor` backends and per-process OS metrics |
| `hashing.py` | BLAKE3 bindings and canonical JSON |

Adding a pipeline stage means adding a `make_<stage>_stage` factory plus a helper function in
`stages.py` (`def solve(run, ...)`, built once per experiment via `run.get_stage(key, factory)`);
the driver itself does not change. `RunContext.import_mesh(...)` and friends are thin wrappers
over those helpers.

`adapt_mesh(run, mesh, solution, config=..., state=None)` runs one `poisson_adapt` cycle and
always returns `{"mesh", "parents", "state", "eta"}`; chain cycles by passing the previous
result's `mesh` and `state` (the status is `result["mesh"].info["status"]`).

## Hashing Engine (`hashing.py`, `third_party/blake3/`, `src/blake3/blake3_hash.cpp`)

- Uses vendored official BLAKE3 v1.8.7 with runtime CPUID SIMD dispatch on x86-64 Linux
  (`SSE2`/`SSE4.1`/`AVX2`/`AVX-512`), little-endian ARM64 NEON, and portable C fallback.
- Compiled into `libpoisson_blake3.so` and loaded via `ctypes` for 1 MiB chunked streaming
  file hashing with zero external Python or system package dependencies.

## Two-Layer Content-Addressed Cache (`cache.py`)

```text
<cache_dir>/
  objects/<content_id><ext>    Immutable output files keyed by BLAKE3 of their actual bytes
  requests/<request_key>.json  Provenance, output content IDs, stage Info, and process metrics
  tmp/                         In-flight staging files (atomically moved via os.replace)
  ephemeral/                   Outputs of non-writing runs (cache="off" or "read_only" miss)
  leases/                      Per-branch flock-backed leases protecting live objects
  .cache.lock                  Cross-process flock coordinating lookup, publish, and eviction
```

- **Request key:** `BLAKE3(v=2, stage, code_hash, input_content_ids, settings, output_specs, env)`.
  Integral floats in `settings` hash as ints (`1.0` and `1` share an entry); the settings passed
  to the binary are unchanged.
- **Code hash:** BLAKE3 of the stage executable, any `code_files` / `version`, and the shared
  libraries the executable links *directly* (`readelf -d` `NEEDED` entries resolved via `ldd`,
  keyed by soname, e.g. HDF5, libstdc++, libc). Transitive dependencies (libcurl, krb5, ...) are
  not tracked. Disable per stage with `track_shared_libs=False` (e.g. when cluster nodes carry
  different but compatible library builds and should share a cache). Scripts, static binaries,
  and hosts without `ldd` track no libraries.
- **Content deduplication:** Upstream refactors or binary edits invalidate that stage's
  `request_key`, but if it produces identical output bytes, its `content_id` is unchanged and
  all downstream stages still get cache hits.
- **Safety:** Validates request JSON schema, `output_specs`, and recorded `size_bytes` on hit
  (with optional full digest verification via `Cache(verify_hashes=True)`), detects mid-run
  input/binary drift before publishing, and supports three cross-process locking modes via
  `Cache(lock_mode=...)` or `POISSON_CACHE_LOCK_MODE`:
  - `"flock"` *(default)*: POSIX `fcntl.flock` on `.cache.lock` and `leases/` (auto-falls back
    to `"lockfile"` if the filesystem returns `ENOSYS`/`EOPNOTSUPP`/`ENOLCK`).
  - `"lockfile"`: POSIX atomic `O_CREAT | O_EXCL` lockfile with stale-age recovery, safe across
    multi-node Lustre/GPFS/NFSv4 shared `/scratch`.
  - `"none"`: Lock-free reads/writes via unique temp files + atomic `os.replace` (requires
    `max_bytes=None` during sweeps; trim between sweeps with `cache clean --max-gb`).

## 4-Mode Layered Cache Control

| Mode | Reads cache | Writes cache | Shorthand |
|---|:---:|:---:|---|
| `"use"` *(default)* | Yes | Yes | `True` |
| `"read_only"` | Yes | No | — |
| `"refresh"` | No | Yes | — |
| `"off"` | No | No | `False` |

Precedence (most specific wins):
1. Per-call override: `run.run_stage(..., cache=...)` / `run.import_mesh(..., cache=...)`
2. Per-experiment per-stage override: `run_experiment(..., stage_cache_modes={"solve": "refresh"})`
3. Per-experiment global override: `run_experiment(..., cache_mode="read_only")`
4. Stage default: `Stage(..., default_cache="use")`

**In-experiment reuse:** every stage request executed during one `run_experiment` call is
remembered by `request_key`. A later branch making the same request reuses those outputs
(status `REUSED`) instead of executing again, so a shared prefix stage runs once per experiment
even under `refresh`, `off`, or a `read_only` miss. Lookup order in `run_stage` is: persistent
cache (if the mode reads) -> in-experiment reuse -> execute. Dry runs apply the same reuse to
their placeholders, so a dry run predicts the same number of executions as the real run. Each
SLURM shard is a separate process and keeps its own record.

## Execution Backend, Metrics & Replay Driver (`executor.py`, `driver.py`, `stages.py`)

- `Executor` abstracts stage execution (`LocalExecutor` runs isolated subprocesses and collects
  per-child wall time, `ru_maxrss`, `cpu_user_s`, `cpu_sys_s`, `cpu_efficiency`, page faults,
  context switches via `os.wait4`, plus host HPC metadata: `cpu_model`, `cpu_count`,
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`, and `SLURM_*` variables).
- **Per-stage environment (`env=`):** `run.run_stage(..., env={"OMP_NUM_THREADS": "8"})` passes
  environment overrides to the child process and includes non-empty `env` in `request_key` so
  thread-scaling sweeps cache separate timings while still deduplicating identical output `.h5`
  bytes in `objects/`.
- **SLURM Job Array sharding (`shard=`):** `run_experiment(..., shard=(index, total))` or
  `shard="slurm"` (reads `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`, falling back to
  `(0, 1)` locally) discovers the full `compare(...)` tree and executes only branches where
  `branch_index % total == index` against a shared `/scratch` cache. A final
  `cache_mode="read_only"` run collects the complete `ExperimentReport` with zero recomputation.
- **Stage registration:** `Stage.from_cli(...)` declaratively maps input flags, flat setting
  flags, and/or `--config <settings.json>` (for deeply nested configs like AMR) with minimal
  boilerplate.
- **Repeated runs & statistics:** `repeats=N` and `warmup=K` record `min`, `median`, `mean`,
  `max`, `std`, and raw per-run samples for `info["time"]`, `wall_time_s`, `max_rss_kb`, and CPU
  metrics inside `requests/<request_key>.json`. Precedence (most specific wins): per-call
  `repeats=`/`warmup=` > `run_experiment(stage_repeats={...}, stage_warmup={...})` >
  `run_experiment(repeats=, warmup=)`. Repeats are **not** part of the request key and do not
  affect lookup: a cached entry is a hit whatever repeat count it was measured with (the trace
  reports the measured count). Re-measure explicitly with `refresh` for that stage.
- **Determinism check:** when a stage runs more than once (repeats + warmup > 1), each run's
  outputs are hashed and compared with the first run. Differences are listed in
  `execution["nondeterministic_outputs"]` and reported as warnings (stored with the cache entry,
  shown in `StageTrace.warnings`, the `warnings` CSV column, and below `format_table()`).
- **Headline timing:** `StageTrace.time_s` is the stage's self-reported `info["time"]` (median
  over repeats) and is what `format_table()` prints; it falls back to process wall time (marked
  `wall`) for stages that report no time. Wall time includes process startup and HDF5 I/O. CSV
  exports list the `info_time_*` columns before the `wall_*` columns.
- **Dry-run & exports:** `run_experiment(..., dry_run=True)` previews all branches and reports
  `HIT` vs `WOULD_RUN` without launching binaries. `ExperimentReport.save_csv(..., layout="long")`
  writes a loop-friendly tidy CSV (one row per `(branch, step_index, stage)` including HPC
  metadata), while `layout="wide"` disambiguates repeated stage calls (`stage#0`, `stage#1`, ...).
