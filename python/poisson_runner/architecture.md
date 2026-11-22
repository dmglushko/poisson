# `poisson_runner` Architecture

`poisson_runner` is the Python orchestration, content-addressed caching, and loop-free
`compare(...)` benchmarking layer built on top of the per-stage C++ CLI binaries.

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

- **Request key:** `BLAKE3(v=2, stage, code_hash, input_content_ids, settings, output_specs)`.
  `code_hash` includes the BLAKE3 hash of the stage executable (and any `code_files` / `version`).
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
- **Repeated runs & statistics:** `repeats=N` and `warmup=K` (per-experiment or per-call) record
  `min`, `median`, `mean`, `max`, `std`, and raw per-run samples for `wall_time_s`,
  `info["time"]`, `max_rss_kb`, and CPU metrics inside `requests/<request_key>.json`.
- **Dry-run & exports:** `run_experiment(..., dry_run=True)` previews all branches and reports
  `HIT` vs `WOULD_RUN` without launching binaries. `ExperimentReport.save_csv(..., layout="long")`
  writes a loop-friendly tidy CSV (one row per `(branch, step_index, stage)` including HPC
  metadata), while `layout="wide"` disambiguates repeated stage calls (`stage#0`, `stage#1`, ...).
