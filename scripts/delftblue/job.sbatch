#!/bin/bash
# One Slurm job running an experiment script. submit.sh submits it three ways:
#   warm-up:   sbatch job.sbatch --repo DIR EXPERIMENT.py --warmup
#   pieces:    sbatch --array=0-15 job.sbatch --repo DIR --shard-total 16 EXPERIMENT.py
#   collect:   sbatch job.sbatch --repo DIR EXPERIMENT.py --collect --out DIR
# Resources (--cpus-per-task, --time, ...) come from the sbatch command line.
#SBATCH --job-name=poisson
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=01:00:00
set -euo pipefail

shard_total="${POISSON_SHARD_TOTAL:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) POISSON_REPO="$2"; shift 2 ;;
        --shard-total) shard_total="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

# Slurm runs a copy of this file, so env.sh is found through --repo / POISSON_REPO.
: "${POISSON_REPO:?POISSON_REPO is not set; submit with scripts/delftblue/submit.sh}"
export POISSON_REPO
source "${POISSON_REPO}/scripts/delftblue/env.sh"

threads="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}"

args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${shard_total}" ]]; then
    # Piece number = array index, so resubmitting `--array=5` reruns exactly piece 5.
    args+=(--shard "${SLURM_ARRAY_TASK_ID}/${shard_total}")
fi

echo "job ${SLURM_JOB_ID:-?} on $(hostname): python3 ${args[*]}"
cd "${POISSON_REPO}"
exec python3 "${args[@]}"
