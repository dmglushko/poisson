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
| `dag.py` | DAG mode: `plan_dag` (stage calls -> tasks -> Slurm array groups, plan file) and `run_dag_task` |
| `experiment_cli.py` | `experiment_main`: standard CLI for experiment scripts (`--count`, `--shard`, `--collect`, `--dag-plan`, `--dag-task`) |
| `cli.py` | `python3 -m poisson_runner` CLI for cache management (`cache status\|inspect\|clean`) and file hashing |
| `hashing.py` | BLAKE3 bindings and canonical JSON |

Adding a pipeline stage means adding a `make_<stage>_stage` factory plus a helper function in
`stages.py` (`def solve(run, ...)`, built once per experiment via `run.get_stage(key, factory)`);
the driver itself does not change. `RunContext.import_mesh(...)` and friends are thin wrappers
over those helpers.

`adapt_mesh(run, mesh, solution, config=..., state=None)` runs one `poisson_adapt` cycle and
always returns `{"mesh", "parents", "state", "eta", "marked"}`; chain cycles by passing the previous
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
  input/binary drift before publishing (by file signature; inputs that are cache objects,
  which other processes touch for LRU or re-publish with identical content, only by presence
  and size), and supports three cross-process locking modes via
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
  bytes in `objects/`. `threads=N` is the short form: it sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to N (an `env` value for one of them that differs
  is refused). Without either, a stage inherits the job's thread count (`job.sbatch` sets it to the
  allocation) and its cache entry is reused whatever that count was. Thread counts must be
  positive integers (`OMP_NUM_THREADS` may list nested levels, `4,2`, and counts as its first).
  A call whose own `resources=` gives fewer `cpus` than its env's thread count is refused in
  every mode, and a run whose effective thread count (set or inherited, from the executor's host
  metadata) exceeds its usable CPUs stores an oversubscription warning with the cache entry.
- **SLURM Job Array sharding (`shard=`):** `run_experiment(..., shard=(index, total))` or
  `shard="slurm"` (reads `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`, falling back to
  no sharding outside an array job) discovers the full `compare(...)` tree under a short-lived
  per-branch-scoped lease, then executes only this shard's branches under its own shard lease
  against a shared `/scratch` cache. `shard_order="stride"` (default) takes the branches where
  `branch_index % total == index`; `"block"` takes one contiguous run of the DFS order, so
  siblings sharing a prefix stage land in one shard. The cache has no in-flight claim, so
  shards that start together and miss the same request all execute it; block order avoids that
  for siblings, stride balances load when cost follows the outermost choice. A final
  `cache_mode="read_only", require_cached=True` run collects the complete `ExperimentReport`
  with guaranteed zero recomputation.
- **Collect mode (`force_cache_mode=`):** `run_experiment(..., force_cache_mode="read_only")`
  overrides every cache mode, per-call ones included; `--collect` uses it, so a call that asks
  for `refresh` still finds the result a piece or DAG task published.
- **Branch selection (`select=`):** `run_experiment(..., select={"mesh": "fine"})` (a label or a
  list of labels per `compare()` name; several names must all match) runs only matching branches,
  found by the same discovery pass. Branch indices stay those of the full experiment, sharding
  splits the selected branches, and a branch that never reaches a selected choice is left out.
  Unknown names or labels, or an empty selection, raise `ValueError`. The report records
  `selection`, and `save` refuses a selected report unless `allow_partial=True`.
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
  exit code 2 if any stage misses the cache, unless `--allow-recompute`). Every mode takes
  `--select NAME=LABEL` (repeatable) and `--shard-order stride|block`; DAG mode adds
  `--dag-plan FILE` and `--dag-task G/I` (below). Inside an array job
  without `--shard` it uses `shard="slurm"`. Exit code 1 means a branch failed, so a SLURM
  `afterok` dependency keeps the collect job from starting. `scripts/delftblue/` wraps this for
  DelftBlue.
- **Execution defaults in the script:** `experiment_main(pipeline, mode=, resources=,
  stage_resources=, split_by=, fuse=, max_fused_time=, max_array_size=, shard_order=)` sets the
  script's defaults, validated before anything runs; a command-line flag wins over each
  (`--resources` and `--stage-resources` key by key, `--split-by` replaces `split_by=` and
  `compare(..., split=True)`, `--no-split` clears both, `--fuse`/`--no-fuse`). `mode`
  (`"shard"` or `"dag"`) only tells `submit.sh` how to submit; a plain `python3 sweep.py`
  always runs every branch. Shard pieces use only `resources`. `--show-settings [--resources ...]
  [--shard-order ...]` prints `setting mode=|cpus=|time=|mem_per_cpu=|shard_order=` lines with
  the flags applied, without creating a cache or running the pipeline, so `submit.sh` keeps no
  copy of these defaults.
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
  `StageTrace.warnings` lists these runner warnings first, then the stage's own
  `info["warnings"]`.
- **Headline timing:** `StageTrace.time_s` is the stage's self-reported `info["time"]` (median
  over repeats) and is what `format_table()` prints; it falls back to process wall time (marked
  `wall`) for stages that report no time. Wall time includes process startup and HDF5 I/O. CSV
  exports list the `info_time_*` columns before the `wall_*` columns.
- **Dry-run & exports:** `run_experiment(..., dry_run=True)` previews all branches and reports
  `HIT` vs `WOULD_RUN` without launching binaries. `ExperimentReport.save_csv(..., layout="long")`
  writes a loop-friendly tidy CSV (one row per `(branch, step_index, stage)` including HPC
  metadata), while `layout="wide"` disambiguates repeated stage calls (`stage#0`, `stage#1`, ...).

## DAG Mode (`dag.py`)

An opt-in alternative to shards, where each stage run is its own Slurm array task and an
array starts once the arrays it reads from have finished. Shards remain the default.

- **Planning (`plan_dag`, `--dag-plan FILE`):** discovery finds the (selected) branches, then
  each one is dry-run with the experiment's real cache modes while `run_stage` logs every call
  (`_StageCallRecord`): `hit` (cached: no task), `run` (a new task) or `reused` (the same
  request as an earlier task). While planning, a request planned earlier wins over an old cache
  entry, so a `refresh` call and a later `use` call of it agree (plain dry runs keep the
  lookup-first order of real runs). A task's parents are the tasks whose placeholder outputs it
  reads (`Artifact.request_key`). Refused: a planned call whose cache mode doesn't write (`off`,
  `read_only` miss), since nothing downstream could read it, and refreshing a request that an
  earlier call already read from the cache, since its new output would orphan what was built
  on the old one.
- **Tasks and the guard:** a task is identified by its branch's choice path and its stage call
  number in that branch, plus `check`: the request key computed without inputs (stage, code
  hash, settings, env, output specs), and `known_inputs`: the content ids of the inputs known
  when planning (raw files, cached results; `null` for a parent task's output). They catch a
  rebuilt binary, an edited experiment or input file, and nodes linking different shared
  libraries. The task also stores its effective cache mode, repeats and warmup, which the task
  run uses whatever options it is given.
- **Resources:** merged key by key (`cpus`, `time`, `mem_per_cpu`), weakest first:
  `DEFAULT_RESOURCES` < `resources` (the script's, with `--resources` over it) <
  `stage_resources[stage]` (the script's) < the call's own `run_stage(..., resources=)` <
  `stage_resource_overrides[stage]` (`--stage-resources`). So within Python the more specific
  setting wins, and the command line wins over the Python setting it overrides. A call's
  `threads=N` then raises its `cpus` to at least N, unless the call's own resources or the
  override set `cpus`, which must then be at least N (so a fixed `stage_resources` allocation
  stays for a thread-scaling sweep); a raw env `*_NUM_THREADS` above the resolved `cpus` is
  refused. A task's resources are resolved once all calls of its request were seen (the first
  explicit `resources=`, and the floor if any call used `threads=`), so they don't depend on
  branch order. Resources stay out of the request key. Two calls of one request asking for
  different explicit resources are refused, except for keys `--stage-resources` sets.
- **Grouping:** tasks go to arrays, in planning order (a topological order), keyed by
  `(stage, resources, split labels, parent arrays)`. Different resources already separate the
  chains below them. `split_by` (one or more names) additionally separates the options of those
  `compare()`s (each must exist on some branch, selected or not), and with `compare_splits` (the
  default) so does every `compare(..., split=True)` of the selected branches. A task's split is a tuple with, per name, the label all its branches
  agree on (`None` where they differ, e.g. a shared prefix). The `split` flag lives on the
  recorded choice step, outside its equality, so replays are checked as before. Arrays
  above `max_array_size` (default 1000) are split before their children are grouped. Arrays
  depend on whole parent arrays (`afterok`), so a task can wait for siblings of its parent.
- **Chain fusion (default; `fuse=False`, `--no-fuse` to turn off):** every array costs a
  scheduler round (measured on DelftBlue: 1-31 s between a parent finishing and its child
  starting). So before grouping, `_fuse_chains` merges tasks into units run back to back in one
  job: task `c` joins its parent `t`'s chain when, in the transitive reduction of the task graph,
  `t` is `c`'s only parent and `c` is `t`'s only child, and both need the same `cpus` and
  `mem_per_cpu`. The reduction drops a parent that is also an ancestor of another parent (the
  skip edges of `case_mesh` and of `adapt_mesh` reading both mesh and solution), which adds no
  waiting. Joins are not fused (that would make `t` wait for `c`'s other parent too), nor are
  fan-outs (the children would wait for each other), nor a chain the replay would run out of
  order (a pipeline sharing artifacts between branches). A unit's time limit is the sum of its
  members', each rounded up to a minute as Slurm does; a chain stops growing before it passes
  `max_fused_time` (`--max-fused-time`, default `24:00:00`, DelftBlue's education limit), and
  tasks with no time limit (`0`) stay alone. Its split label is its last member's. Inside a
  unit a planned `refresh` always executes, even if an earlier member produced the same request. Units, not tasks, are grouped as above, keyed
  by their tuple of stages; an array of fused units is labelled `a+b+c` (or `a+N+z` for long
  chains, at most 48 safe characters, for job names and log files).
- **Running a task (`run_dag_task`, `--dag-plan FILE --dag-task G/I`):** element I of array G
  is a unit. Its branch is its last member's, which calls every earlier member too (each reads
  the one before); `steps` are the members' call numbers there. The replay is a dry run in
  which every other stage call only looks up the cache (`READ_ONLY`, a miss gives a
  placeholder), so only the unit's parent arrays need to be done. Each member's call must have no
  placeholder inputs, the same stage, `check` and known inputs, and a writing mode; it executes
  (or is found cached) and never reuses a placeholder that an earlier call of the replay left
  for the same request (two producers with identical bytes give their consumers one real key).
  Later members read earlier members' real outputs in memory. After the last member,
  `_DagTaskDone`, a `BaseException`, ends the replay; the unit then fails unless every member
  finished and is published in the cache (a pipeline catching a member's failure and carrying on
  with a fallback input must not count as done). `run.dry_run` stays `True` for the pipeline's
  code, as in `--count`.
- **Limits:** a pipeline whose shape depends on stage outputs (e.g. an AMR loop that reads
  `info["status"]`) can't be planned, since a dry run has no outputs; a task then fails with
  "Plan mismatch" and the read-only collect refuses. Use shards for those. Leases don't span
  Slurm jobs, so an output evicted between two arrays fails the later task: run DAG sweeps
  with `max_bytes=None` or a cache larger than the sweep. Rerunning a failed sweep means
  planning it again: only uncached tasks are submitted.
- **Plan file:** JSON with `format_version` (3), `selection`, `split_by` (a list), `fused`, the choice
  paths of the branches that run a unit, `tasks` (id, branch, step, stage, check, parents,
  resources, split, unit), `units` (id, branch, tasks, steps, stages, parents, resources,
  split, group) and `groups` (id, stage label, stages, resources, split, parents, units). `--dag-plan`
  also prints one `dag-group id|size|parents|cpus|time|mem_per_cpu|stage|leaf` line per
  array (`-` for no parents) for `submit.sh --dag`.

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
