# Running experiments on DelftBlue

These scripts build the stage programs on DelftBlue and run an experiment as parallel Slurm
jobs. Each job does part of the branches, and a final collect job gathers the results into one
saved run. Jobs share their results through one cache on `/scratch`.

| File | What it does |
|---|---|
| `env.sh` | Loads the modules and sets the paths and settings every other script uses. **Edit the settings at the top once.** |
| `build.sh` | Configures and compiles the C++ stage programs into `build-delftblue/`. |
| `submit.sh` | Submits an experiment: optional warm-up job, the parallel pieces (or, with `--dag`, one array per group of stage runs), and the collect job. |
| `job.sbatch` | The Slurm job script that `submit.sh` submits. You don't run it yourself. |
| `smoke_test.py` | A tiny 3-branch experiment for checking the setup. |

## One-time setup

```bash
ssh <netid>@login.delftblue.tudelft.nl
git clone https://github.com/dmglushko/poisson.git ~/poisson && cd ~/poisson
```

Optionally edit the top of `scripts/delftblue/env.sh`:

- `POISSON_SLURM_ACCOUNT`: your project account if you have more than one (leave empty to use
  your default account; `sacctmgr show associations user=$USER format=account` lists available
  accounts).
- `POISSON_MODULES`: defaults to DelftBlue's verified `2026 compilers cpu gcc/13.4.0 cmake hdf5 eigen python`
  stack (AlmaLinux 9 / `x86_64_v3`, portable across both `compute-p1` and `compute-p2`).

Jobs default to partition `compute-p1` with `3900M` per CPU (the `compute-p1` limit is `3996M`).
Override with `--partition` / `--mem-per-cpu` if needed.

Then build, and again after every C++ change (but not while an experiment is queued or running:
a rebuild changes the programs, so the cache treats their results as different):

```bash
bash scripts/delftblue/build.sh
```

Check the setup with the smoke test:

```bash
bash scripts/delftblue/submit.sh scripts/delftblue/smoke_test.py --time 00:05:00
```

## Running an experiment

An experiment is the same Python file you run on your laptop, ending with:

```python
from poisson_runner import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main(pipeline))
```

Then:

```bash
source scripts/delftblue/env.sh             # modules and paths, once per login shell
python3 experiments/sweep.py --count        # how many branches (on the login node, computes nothing)
bash scripts/delftblue/submit.sh experiments/sweep.py --cpus 8 --time 02:00:00
squeue --me                                 # watch the jobs
```

`submit.sh` first runs `--count` on the login node, which catches errors in the script before
anything waits in the queue. By default it submits one piece per branch; `--pieces N` uses fewer,
larger pieces. `--warmup` first runs branch 0 alone, so early steps shared by all branches (such
as a mesh import) are computed once instead of by every piece at the same time.
`--print-only` shows the `sbatch` commands without submitting. `--help` lists all options.

Logs go to `logs/<name>/`, one file per job. The collect job saves the run (`report.json`,
`report.csv`, `report_long.csv`) to `results/<name>/<job id>/`. It starts only after every piece
has succeeded. It also refuses to save if any result is missing from the cache, instead of quietly
recomputing everything in one job.

### Settings in the script

The script can keep the options it always needs, so `submit.sh` needs no flags for them:

```python
raise SystemExit(experiment_main(
    pipeline,
    mode="dag",                                        # or "shard" (the default)
    resources={"cpus": 1, "time": "00:30:00"},         # every task, and every piece in shard mode
    stage_resources={"solve": {"cpus": 16, "time": "04:00:00"}},  # DAG tasks of one stage
    split_by="mesh",                                   # or compare("mesh", [...], split=True)
    shard_order="block",
))
```

An option given to `submit.sh` wins over the script's: `--cpus`, `--time` and `--mem-per-cpu`
key by key, `--stage-resources` even over a `run_stage(..., resources=)` in the pipeline,
`--split-by` replaces the script's splits and `--no-split` drops them, `--dag` / `--shards`
choose the mode. Anything set nowhere gets the built-in default (`1` CPU, `01:00:00`, `3900M`,
pieces, `stride`). `python3 sweep.py --show-settings` prints what the script resolves to. The
collect job always uses 1 CPU, 30 minutes and `3900M` (or `--mem-per-cpu` when given).

### Choosing branches and how they are split

`--select NAME=LABEL` submits only the branches whose `compare(NAME)` chose `LABEL`, for example
`--select mesh=fine`. Repeat it for several labels or names. Together with `--no-collect` and
`--collect-only` this gives each group its own resources, followed by one collect job:

```bash
a=$(bash scripts/delftblue/submit.sh experiments/sweep.py --select mesh=coarse --no-collect --print-id)
b=$(bash scripts/delftblue/submit.sh experiments/sweep.py --select mesh=fine --cpus 16 --time 04:00:00 \
      --no-collect --print-id)
bash scripts/delftblue/submit.sh experiments/sweep.py --collect-only --after "$a,$b"
```

With `--dag`, `--print-id` prints nothing when everything is cached already; `--after` skips
such empty ids (if both are empty, nothing was submitted: run `--collect-only` without `--after`).

With fewer pieces than branches, `--shard-order` decides which branches share a piece. `stride`
(the default) deals them out round-robin, which evens out the work when, say, all fine-mesh
branches come last. `block` gives each piece a run of neighbouring branches. Neighbours share
their early steps (the same mesh with different solvers), and pieces that start at the same time
would each compute a shared step that isn't cached yet, so `block` computes it once per piece.

### DAG mode (each stage run is its own job)

`--dag` plans the experiment and submits one job array per group of stage runs
that aren't cached yet. An array starts once the arrays it reads from have finished, so a
coarse-mesh solve doesn't wait for the fine mesh to be imported:

```bash
bash scripts/delftblue/submit.sh experiments/sweep.py --dag --time 00:30:00 \
    --stage-resources solve:cpus=16,time=04:00:00 --split-by mesh
```

- `--cpus`, `--time` and `--mem-per-cpu` are the defaults for every task, the script's
  `stage_resources` override them for one stage, an experiment can pass
  `resources={"cpus": 16, "time": "04:00:00"}` to a `run_stage` call to override both (for
  example only for fine meshes), and `--stage-resources` overrides all of these for one stage,
  key by key: `--stage-resources solve:time=06:00:00` gives every solve 6 hours and keeps each
  call's cpus.
- Arrays wait for whole parent arrays. Runs with different resources get separate arrays anyway,
  and `--split-by NAME` (repeatable), `split_by=` or `compare(NAME, [...], split=True)` also
  separate the options of a `compare()`.
- Each array waits in the queue on its own: on DelftBlue, 1-31 s passed between an array
  finishing and the next one starting. To save those rounds, a chain of stage runs where each
  one only feeds the next (and needs the same cpus and memory) runs back to back in one task,
  shown as e.g. `assemble+solve`, with the sum of their time limits. A stage shared by several
  branches, or one that reads two separate chains, still gets its own array. A chain stops
  before its summed time passes `--max-fused-time` (default `24:00:00`, the education account
  limit; research accounts can allow more, e.g. `5-00:00:00`). `--no-fuse` gives every stage run
  its own array again.
- DAG mode pays off when shared early steps are heavy, or stages need very different resources;
  when stages take seconds, pieces are simpler.
- It needs an experiment whose stage calls don't depend on stage results. An AMR loop that reads
  `info["status"]` to decide whether to continue can't be planned ahead. Its tasks fail with
  "Plan mismatch", so use pieces for it.
- If a task fails, cancel the queued jobs the message names, fix the cause, and run the same
  command again. It plans again and submits only what's still missing.
- Don't rebuild while it runs; a task notices a changed program and refuses.
- The plan is made where you run `submit.sh`. On DelftBlue the login nodes and compute nodes
  have different builds of the system `libc`/`libm`, so a plan made on a login node makes every
  task refuse with "Plan mismatch" (the same cause as
  [#3](https://github.com/dmglushko/poisson/issues/3)). Until #3 is decided, run the submission
  itself on a compute node, which also plans faster than a busy login node:

  ```bash
  srun -p compute-p1 -n1 -c1 -t 00:10:00 bash scripts/delftblue/submit.sh experiments/sweep.py --dag
  ```

  `submit.sh` warns when it plans a DAG outside a Slurm job, which matters most when the mode
  comes from the script's `mode="dag"`.

### When a piece fails

The collect job then stays queued with reason `DependencyNeverSatisfied`. Read the piece's log,
cancel the collect job, and rerun only the failed pieces with the same `--pieces` count:

```bash
scancel <collect job id>
bash scripts/delftblue/submit.sh experiments/sweep.py --pieces 16 --only 3,7 --time 04:00:00
```

(Add `--shards` if the script sets `mode="dag"`; the command `submit.sh` prints after submitting
already has it.)

The other pieces' results are reused from the cache, and a new collect job is queued.

## Good to know

- **`/scratch` is cleaned automatically.** The cache lives in `/scratch/$USER/poisson_cache`.
  The saved report's CSV files are permanent, but the `.h5` result files they refer to stay in the
  cache and disappear when `/scratch` is cleaned
  ([#4](https://github.com/dmglushko/poisson/issues/4)).
- **Threads.** Each job sets `OMP_NUM_THREADS` (and the BLAS equivalents) to its `--cpus`. A
  stage run that way is cached whatever its CPU count, so a later run with other resources
  reuses it. `run_stage(..., threads=N)` instead sets the thread count for that stage and makes
  it part of the cache key (one entry per thread count, for scaling studies); in DAG mode it also
  gives the task at least N CPUs. A run with more threads than CPUs gets a warning in the report.
- **Locking.** `env.sh` sets `POISSON_CACHE_LOCK_MODE=lockfile`, which works across nodes, and
  turns off HDF5's own file locking. The runner also picks `lockfile` by itself inside any Slurm
  job.
- **Time limits.** When a job hits its time limit, Slurm stops it and the runner cleans up its
  locks and temporary files, so other jobs aren't held up. By default pieces are dealt out round-robin
  (`--shard-order stride`), so set `--time` for the slowest piece.
- **Python packages.** `poisson_runner` itself uses only the Python standard library, so running
  experiments works with the default `2026` modules. The `2026` stack does not include a `py-h5py`
  module, so running the `test_runner.py` suite under `ctest` (or writing post-processing scripts
  that import `h5py`) requires installing `h5py` in a small `python-venv`.
