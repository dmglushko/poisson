#!/bin/bash
# Submit an experiment to Slurm as parallel pieces plus a collect job that runs after them.
#
#   bash scripts/delftblue/submit.sh EXPERIMENT.py [options] [-- extra sbatch options]
#
# Options (defaults in brackets):
#   --pieces N        number of parallel pieces [one per branch, from `EXPERIMENT.py --count`]
#   --cpus C          CPUs per piece [1]
#   --mem-per-cpu M   memory per CPU [3900M] (compute-p1 max is 3996M)
#   --time T          time limit per piece, e.g. 02:00:00 [01:00:00]
#   --partition P     Slurm partition [compute-p1]
#   --account A       Slurm account [POISSON_SLURM_ACCOUNT from env.sh, or Slurm default]
#   --warmup          first run branch 0 alone, so shared early steps are computed once
#   --only LIST       rerun only these pieces, e.g. 5 or 3,7-9 (needs --pieces as before)
#   --out DIR         where the collect job saves the run [results/<name>/<pieces job id>]
#   --name NAME       job and results name [experiment file name without .py]
#   --print-only      show the sbatch commands without submitting anything
#
# Example:
#   bash scripts/delftblue/submit.sh experiments/sweep.py --cpus 8 --time 02:00:00
set -euo pipefail

usage() { sed -n '2,/^set -euo/{/^#/s/^# \{0,1\}//p}' "$0"; exit "${1:-0}"; }
require_arg() { [[ $# -ge 2 ]] || { echo "submit.sh: $1 requires a value" >&2; exit 1; }; }

[[ $# -ge 1 ]] || usage 1
if [[ $1 == -h || $1 == --help ]]; then usage 0; fi
experiment="$1"; shift
[[ -f "${experiment}" ]] || { echo "submit.sh: no such file: ${experiment}" >&2; exit 1; }
experiment="$(cd "$(dirname "${experiment}")" && pwd)/$(basename "${experiment}")"

pieces="" cpus=1 mem_per_cpu=3900M time_limit=01:00:00 partition=compute-p1 account="" warmup=0
only="" out="" name="" print_only=0 extra=() warmup_id=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pieces) require_arg "$@"; pieces="$2"; shift 2 ;;
        --cpus) require_arg "$@"; cpus="$2"; shift 2 ;;
        --mem-per-cpu) require_arg "$@"; mem_per_cpu="$2"; shift 2 ;;
        --time) require_arg "$@"; time_limit="$2"; shift 2 ;;
        --partition) require_arg "$@"; partition="$2"; shift 2 ;;
        --account) require_arg "$@"; account="$2"; shift 2 ;;
        --warmup) warmup=1; shift ;;
        --only) require_arg "$@"; only="$2"; shift 2 ;;
        --out) require_arg "$@"; out="$2"; shift 2 ;;
        --name) require_arg "$@"; name="$2"; shift 2 ;;
        --print-only) print_only=1; shift ;;
        -h|--help) usage 0 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "submit.sh: unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
done

for opt in "${extra[@]}"; do
    case "${opt}" in
        --array|--array=*|-a|-a*|--dependency|--dependency=*|-d|-d*|--export|--export=*|--parsable)
            echo "submit.sh: extra sbatch option '${opt}' is reserved by the workflow (array, dependency, export, parsable)" >&2
            exit 1
            ;;
    esac
done

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
account="${account:-${POISSON_SLURM_ACCOUNT}}"
name="${name:-$(basename "${experiment}" .py)}"

if [[ -n "${only}" && -z "${pieces}" ]]; then
    echo "submit.sh: --only needs --pieces with the same number as the original submission" >&2
    exit 1
fi

# Dry run on the login node: catches errors in the script before anything waits in the queue.
echo "Checking ${experiment} ..."
count_output="$(cd "${POISSON_REPO}" && python3 "${experiment}" --count)" || {
    echo "submit.sh: '${experiment} --count' failed; fix the experiment before submitting" >&2
    exit 1
}
echo "${count_output}"
branches="$(sed -n 's/^branches: \([0-9][0-9]*\)$/\1/p' <<<"${count_output}" | tail -n 1)"
[[ -n "${branches}" ]] || { echo "submit.sh: could not read the branch count (does the script call experiment_main?)" >&2; exit 1; }
pieces="${pieces:-${branches}}"
if (( pieces < 1 )); then echo "submit.sh: nothing to run (0 branches)" >&2; exit 1; fi
if (( pieces > branches )); then
    echo "Only ${branches} branches: using ${branches} pieces instead of ${pieces}"
    pieces="${branches}"
fi

logs="${POISSON_REPO}/logs/${name}"
mkdir -p "${logs}" "${POISSON_CACHE_DIR}"

common=(--parsable --ntasks=1 --partition="${partition}")
if [[ -n "${account}" ]]; then common+=(--account="${account}"); fi
job="${POISSON_REPO}/scripts/delftblue/job.sbatch"

submit() {  # submit LABEL SBATCH_ARGS...; prints the numeric job id
    local label="$1"; shift
    if (( print_only )); then
        echo "sbatch $*" >&2
        echo "<${label}-job-id>"
        return 0
    fi
    local raw job_id
    raw="$(sbatch "$@")"
    raw="${raw//$'\r'/}"
    raw="${raw##*$'\n'}"
    job_id="${raw%%;*}"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "submit.sh: unexpected sbatch --parsable output for ${label}: '${raw}'" >&2
        return 1
    fi
    echo "${job_id}"
}

dependency=()
if (( warmup )); then
    warmup_id="$(submit warmup "${common[@]}" --job-name="${name}-warmup" \
        --cpus-per-task="${cpus}" --mem-per-cpu="${mem_per_cpu}" --time="${time_limit}" \
        --output="${logs}/warmup_%j.out" \
        "${extra[@]}" "${job}" --repo "${POISSON_REPO}" "${experiment}" --warmup)"
    dependency=(--dependency="afterok:${warmup_id}")
fi

array="${only:-0-$(( pieces - 1 ))}"
pieces_id="$(submit pieces "${common[@]}" "${dependency[@]}" --job-name="${name}" --array="${array}" \
    --cpus-per-task="${cpus}" --mem-per-cpu="${mem_per_cpu}" --time="${time_limit}" \
    --output="${logs}/piece_%A_%a.out" \
    "${extra[@]}" "${job}" --repo "${POISSON_REPO}" --shard-total "${pieces}" "${experiment}")"

out="${out:-${POISSON_REPO}/results/${name}/${pieces_id}}"
collect_id="$(submit collect "${common[@]}" --job-name="${name}-collect" \
    --dependency="afterok:${pieces_id}" \
    --cpus-per-task=1 --mem-per-cpu="${mem_per_cpu}" --time=00:30:00 --output="${logs}/collect_%j.out" \
    "${extra[@]}" "${job}" --repo "${POISSON_REPO}" "${experiment}" --collect --out "${out}" --overwrite)"

if (( print_only )); then echo; echo "(--print-only: nothing was submitted)"; exit 0; fi
cat <<EOF

Submitted ${name}: ${pieces} pieces (array ${array}) as job ${pieces_id}, collect job ${collect_id}${warmup_id:+, warm-up job ${warmup_id}}.
  watch:    squeue --me
  logs:     ${logs}/
  results:  ${out}/  (written by the collect job once every piece succeeded)
If a piece fails, the collect job won't start (squeue shows it waiting on DependencyNeverSatisfied).
Cancel it with 'scancel ${collect_id}', then rerun only the failed pieces, e.g. pieces 3 and 7:
  bash scripts/delftblue/submit.sh ${experiment} --pieces ${pieces} --only 3,7 [--time ...]
That queues a new collect job; everything the other pieces finished is reused from the cache.
EOF
