# Running experiments on DelftBlue

These scripts build the stage programs on DelftBlue and run an experiment as parallel Slurm
jobs. Each job does part of the branches, and a final collect job gathers the results into one
saved run. Jobs share their results through one cache on `/scratch`.

| File | What it does |
|---|---|
| `env.sh` | Loads the modules and sets the paths and settings every other script uses. **Edit the settings at the top once.** |
| `build.sh` | Configures and compiles the C++ stage programs into `build-delftblue/`. |
| `submit.sh` | Submits an experiment: optional warm-up job, the parallel pieces, and the collect job. |
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

### When a piece fails

The collect job then stays queued with reason `DependencyNeverSatisfied`. Read the piece's log,
cancel the collect job, and rerun only the failed pieces with the same `--pieces` count:

```bash
scancel <collect job id>
bash scripts/delftblue/submit.sh experiments/sweep.py --pieces 16 --only 3,7 --time 04:00:00
```

The other pieces' results are reused from the cache, and a new collect job is queued.

## Good to know

- **`/scratch` is cleaned automatically.** The cache lives in `/scratch/$USER/poisson_cache`.
  The saved report's CSV files are permanent, but the `.h5` result files they refer to stay in the
  cache and disappear when `/scratch` is cleaned
  ([#4](https://github.com/dmglushko/poisson/issues/4)).
- **Threads.** Each job sets `OMP_NUM_THREADS` (and the BLAS equivalents) to its `--cpus`.
- **Locking.** `env.sh` sets `POISSON_CACHE_LOCK_MODE=lockfile`, which works across nodes, and
  turns off HDF5's own file locking. The runner also picks `lockfile` by itself inside any Slurm
  job.
- **Time limits.** When a job hits its time limit, Slurm stops it and the runner cleans up its
  locks and temporary files, so other jobs aren't held up. Pieces are dealt out round-robin, so
  set `--time` for the slowest piece.
- **Python packages.** `poisson_runner` itself uses only the Python standard library, so running
  experiments works with the default `2026` modules. The `2026` stack does not include a `py-h5py`
  module, so running the `test_runner.py` suite under `ctest` (or writing post-processing scripts
  that import `h5py`) requires installing `h5py` in a small `python-venv`.
