# Shared environment for building and running on DelftBlue. Sourced by build.sh, submit.sh
# and job.sbatch; you can also `source scripts/delftblue/env.sh` in a login shell.
#
# ---------------------------------------------------------------- settings you may need to edit
# Each can also be set in your shell before running the scripts.

# Slurm account to charge, e.g. research-<faculty>-<department> or education-<faculty>-<programme>.
# Leave empty to use your default DelftBlue account; `sacctmgr show associations user=$USER format=account`
# lists the ones you can use.
: "${POISSON_SLURM_ACCOUNT:=}"

# Software stack and modules (verified on DelftBlue's 2026 AlmaLinux 9 / x86_64_v3 stack, portable
# across both compute-p1 Intel and compute-p2 AMD nodes).
# Provides GCC 13.4 (C++23), CMake 3.31, HDF5 1.14, Eigen 3.4, and Python 3.13.
: "${POISSON_MODULES:=2026 compilers cpu gcc/13.4.0 cmake hdf5 eigen python}"

# Shared cache for all jobs. /scratch is fast and large, but old files are deleted automatically.
: "${POISSON_CACHE_DIR:=/scratch/${USER}/poisson_cache}"
# ---------------------------------------------------------------------------------------------

_poisson_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${POISSON_REPO:=$(cd "${_poisson_env_dir}/../.." && pwd)}"
: "${POISSON_BUILD_DIR:=${POISSON_REPO}/build-delftblue}"
unset _poisson_env_dir

if command -v module >/dev/null 2>&1; then
    # Lmod's shell functions trip over `set -u`; switch it off around them.
    _poisson_u_was_set=0; [[ $- == *u* ]] && _poisson_u_was_set=1; set +u
    module purge
    # shellcheck disable=SC2086  # word splitting of the module list is intended
    if ! module load ${POISSON_MODULES}; then
        echo "env.sh: 'module load ${POISSON_MODULES}' failed; fix POISSON_MODULES in scripts/delftblue/env.sh" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ${_poisson_u_was_set} == 1 ]]; then set -u; fi
    unset _poisson_u_was_set
else
    echo "env.sh: no 'module' command here; using the tools already on PATH" >&2
fi

export POISSON_REPO POISSON_BUILD_DIR POISSON_CACHE_DIR POISSON_SLURM_ACCOUNT POISSON_MODULES
export PYTHONPATH="${POISSON_REPO}/python${PYTHONPATH:+:${PYTHONPATH}}"
# Cross-node locking for the shared cache (flock may not coordinate between nodes).
export POISSON_CACHE_LOCK_MODE="${POISSON_CACHE_LOCK_MODE:-lockfile}"
# HDF5's own file locking needs flock on the shared filesystem; the stage files don't need it.
export HDF5_USE_FILE_LOCKING=FALSE
