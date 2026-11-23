# `poisson_runner` Architecture

`poisson_runner` is the Python orchestration, content-addressed caching, and loop-free
`compare(...)` benchmarking layer built on top of the per-stage C++ CLI binaries.

## Module Layout

| Module | Responsibility |
|---|---|
| `driver.py` | `compare(...)` replay, `RunContext.run_stage` (inputs -> request key -> lookup -> reuse -> execute -> publish), `run_experiment`, SLURM sharding |
| `benchmark.py` | `measure(...)`: warmup + repeated executions, per-metric statistics, output determinism check |
| `report.py` | `StageTrace`, `BranchResult`, `ExperimentReport` (table, JSON, wide/long CSV, DataFrame, `save`), `info_scalars` |
| `records.py` | Tagged JSON encoding of branch outputs and choice values; explicit `register_record` registry |
| `results.py` | `load_run` / `SavedRun` (`select`, `one`, `match`) and `evaluate` for post-hoc evaluation |
| `stages.py` | `Stage` / `Stage.from_cli`, code hashing (binary + shared libraries), project stage factories and helpers (`import_mesh`, `import_free_charges`, `adapt_mesh`, and the verification stages `case_inputs`, `gmsh_mesh`, `case_mesh`) |
| `cases.py` | `Case(name, params)`: a verification case as a name and its parameters (a registered record), and `dirichlet` / `neumann` / `robin` helpers |
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

`import_mesh(..., screening=True)` passes `--screening on|off` to the importer.

### Verification stages

The verification cases live in C++ ([src/verification/architecture.md](../../src/verification/architecture.md));
a Python `Case` only carries `name` and `params` (stored read-only, so its hash cannot
change) and gives the C++ stages `case.spec()` (`{"name", "params"}`), which also puts every
parameter into the request key. Equality and hashing use the normalised spec (`1 == 1.0`,
`true != 1`).

| Helper | Stage | Result |
|---|---|---|
| `case_inputs(run, case)` | `poisson_case` | `{"geometry": .geo, "materials": .yaml, "charges": .cha}` |
| `gmsh_mesh(run, geo, h_max=..., h_min=, h_charge=, r_fine=, r_coarse=, algorithm=, optimize=, optimize_netgen=, random_factor=)` | system `gmsh` (`-3 -format msh22 -order 1 -nt 1`; sizes as `-setnumber` constants) | `.msh` |
| `case_mesh(run, case, **gmsh_options)` | the two above, then `import_mesh` with `screening=case.screened` (an empty `.cha` is left out) | `MeshProperties` `.h5` |

`case_mesh` is also a `RunContext` method (`run.case_mesh(case, h_max=...)`); `case_inputs`
and `gmsh_mesh` are module functions taking `run`. `gmsh_mesh` only accepts self-contained
`.geo` scripts (no `Include`, `Merge`, ...), since the cache hashes the script's own text.
An experiment returns `{"mesh", "case"}` (the case is saved with its parameters and rebuilt
as a `Case` on load). Comparing a solution with the case's exact solution waits for the
`Solution` format ([issue #6](https://github.com/dmglushko/poisson/issues/6)).

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
  - `"flock"` *(default locally)*: POSIX `fcntl.flock` on `.cache.lock` and `leases/` (auto-falls back
    to `"lockfile"` if the filesystem returns `ENOSYS`/`EOPNOTSUPP`/`ENOLCK`). The lock mode is
    fixed when a cache is initialized and stored in `.lock_mode`; a new cache created inside a
    SLURM job (`SLURM_JOB_ID` set) defaults to `"lockfile"`, while opening an existing cache with
    a conflicting mode raises `ValueError` so concurrent processes never split mutual exclusion.
  - `"lockfile"`: POSIX atomic hard-link / `O_CREAT | O_EXCL` lockfile with fenced stale-age
    recovery, safe across multi-node Lustre/GPFS/BeeGFS/NFSv4 shared `/scratch`. Holders refresh
    the lockfile and their lease every `lock_stale_s / 3`; one older than `lock_stale_s`
    (default 60 s) whose owner is on another host counts as abandoned. Waiters give up after
    `lock_timeout_s` (default 300 s), which must exceed `lock_stale_s` so a waiter can reclaim a
    lock left by a killed job. Routine `tmp/` and `ephemeral/` cleanup also checks active remote
    leases so live multi-hour jobs are never pruned by age.
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
  no sharding outside an array job) discovers the full `compare(...)` tree under a short-lived
  per-branch-scoped lease, then executes only branches where `branch_index % total == index`
  under its own shard lease against a shared `/scratch` cache. A final
  `cache_mode="read_only", require_cached=True` run collects the complete `ExperimentReport`
  with guaranteed zero recomputation.
- **Termination:** while `run_experiment` runs in the main thread with no SIGTERM handler set,
  SIGTERM (what SLURM sends at the time limit or on `scancel`) raises `SystemExit(143)`. Lockfile
  and lease acquisition defer SIGTERM across the atomic create/arm boundary, and each stage runs
  in a runner-owned staging directory so the running stage process is killed and all locks,
  leases, and sibling staging files are cleaned up. SIGKILL (out-of-memory kills) can't be caught;
  stale-age recovery covers it.
- **Experiment command line (`experiment_cli.py`):** an experiment script ending in
  `raise SystemExit(experiment_main(pipeline))` accepts `--count` (dry run; prints
  `branches: N`), `--shard I/N`, `--warmup` (first branch only, to fill shared early steps) and
  `--collect --out DIR` (read-only gather with `require_cached=True` and `save`; refuses with
  exit code 2 if any stage misses the cache, unless `--allow-recompute`). Inside an array job
  without `--shard` it uses `shard="slurm"`. Exit code 1 means a branch failed, so a SLURM
  `afterok` dependency keeps the collect job from starting. `scripts/delftblue/` wraps this for
  DelftBlue.
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

## Results: Named Outputs, Metric Columns, Saved Runs & Evaluation (`report.py`, `records.py`, `results.py`)

Running an experiment and evaluating its results are separate steps. The experiment only
declares what it produced; what is done with the results is decided later, in another script.

- **Named outputs:** the pipeline's return value is the branch's output contract. Returning a
  dict (`{"mesh": mesh, "solution": sol, "dofs": 5200, "case": case}`) gives
  `BranchResult.outputs`, `branch["mesh"]`, and `BranchResult.scalars` (the scalar entries).
  `None` means no outputs; any other value is stored as `outputs["result"]`. Every
  `StageTrace.outputs` additionally records that stage call's `Artifact`s for provenance.
- **Metric columns:** both CSV layouts and `to_dataframe()` (pandas imported lazily) carry
  `res.<key>` for each scalar output and `<stage>.<field>` for every scalar of a stage's
  `.info.json` (`info_scalars`): top-level scalars except `type`, `time` (already
  `info_time_s`) and `warnings`; top-level dicts one level deep (`details.<k>`); a composite's
  `stages` list by step name (`adapt_mesh.estimate.total`, repeated names as `refine#0`).
  Missing values are empty cells. Column names never collide: a second source flattening to
  an existing name (a stage literally named `res`, or a literal `details.count` key next to a
  `details` dict) gets `~2`, `~3`, ... appended. Metric columns are sorted by name within each
  stage, so cold and cached runs produce identical headers. In the wide layout, repeated stage calls are `stage#i.<field>`;
  in the long layout the column is `stage.<field>` and `step_index` separates calls. Choice names
  that clash with any fixed or data column are renamed `choice_<name>` in both layouts. Info
  field names therefore are an interface for analysis scripts: rename a field rather than
  changing its meaning or units.
- **Saving:** `report.save(dir)` writes `report.json` (`format_version`, `cache_root`, choice
  labels and values, outputs as tagged JSON with artifact descriptors), `report.csv` (wide) and
  `report_long.csv`. It encodes everything first, stages all three files, then publishes them
  under a `.report.lock` flock, with `report.json` (the commit point) last; the old
  `report.json` is removed before publishing, so a failure never leaves a mixed bundle. It
  refuses dry runs, partial SLURM shards (unless
  `allow_partial=True`), an existing report (unless `overwrite=True`), and returned values that
  cannot be encoded (`RecordError`). Choice values are lenient: an unsavable one (a callable)
  is stored as an `Unserializable` placeholder and matching uses labels. Output files are not
  copied; they stay in the cache (protecting saved runs from eviction is a separate, open issue).
- **Records:** `@register_record("name")` on a dataclass (or a class with `to_dict`/`from_dict`)
  saves instances as `{"$record": "name", "data": ...}`. Loading never imports modules named in
  the file: import the module that registers the type first, else the value comes back as an
  `UnresolvedRecord`. Other tags: `$artifact`, `$path`, `$float` (non-finite), `$map` (dicts with
  `$`-prefixed keys), `$unserializable`.
- **Codec guarantees:** tuples reload as tuples (`$tuple`); returned dicts with non-string
  keys are rejected rather than coerced; an `UnresolvedRecord` can be re-saved and loads as the
  real type once registered; registering a name already used by a different class (module +
  qualname) fails, as does a dataclass with `init=False` fields without `to_dict`/`from_dict`.
  Artifact descriptors keep `info` and `execution`. Only `save()` marks a file as a saved run
  (`saved_run`, `format_version`); `save_json()` exports are not loadable.
- **Loading:** `load_run(dir, cache=None)` rebuilds the `ExperimentReport` as a `SavedRun`.
  Artifacts inside the cache are re-rooted under `cache.objects_dir` when `cache=` is given (a
  different mount of the same cache), else under the saved `cache_root`. `branch["name"]` raises
  `FileNotFoundError` if the file is gone. `select(**labels)` returns all matches (failed
  branches included), `one(**labels)` exactly one successful match, and `match(branch,
  **overrides)` the branch with exactly the same set of choices except the overridden ones
  (`key=None` ignores a choice on both sides, e.g. one that only some branches have); both raise `LookupError` on zero, several, or a failed match.
- **Evaluation:** `evaluate(saved, fn, **run_experiment_options)` is `run_experiment` over the
  successful saved branches: one evaluation branch per saved branch, selected by the reserved
  choice `source_branch`, calling `fn(run, branch)`. Stages called in `fn` are cached like any
  other, so re-running an analysis script is all hits, and the sweep never re-runs. The returned
  report merges each saved branch's choices into its rows; evaluation choices that clash with a
  saved choice name are renamed `eval.<name>` (`eval.<name>~2`, ... if needed), allocated once
  per report. `cache` is where evaluation stages are cached and defaults to the saved run's
  cache; saved outputs are always read from where `load_run` resolved them. Dry-run reports are
  rejected. The result is an ordinary `ExperimentReport` (CSV, DataFrame, `save`).
  As with any experiment, sharded evaluation runs a dry-run discovery pass first, where stage
  outputs are placeholders: an `fn` that reads a file produced by a stage it just ran must
  guard that read with `if not run.dry_run`.
