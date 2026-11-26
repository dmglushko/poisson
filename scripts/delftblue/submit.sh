#!/bin/bash
# Submit an experiment to Slurm as parallel pieces plus a collect job that runs after them.
#
#   bash scripts/delftblue/submit.sh EXPERIMENT.py [options] [-- extra sbatch options]
#
# Options (defaults in brackets):
#   --pieces N          number of parallel pieces [one per branch, from `EXPERIMENT.py --count`]
#   --shard-order O     how branches are dealt to pieces: stride (every N-th) or block
#                       (contiguous runs, so branches sharing early steps stay together) [stride]
#   --select NAME=LABEL only branches whose compare(NAME) chose LABEL; repeatable [all branches]
#   --cpus C            CPUs per piece [1]
#   --mem-per-cpu M     memory per CPU [3900M] (compute-p1 max is 3996M)
#   --time T            time limit per piece, e.g. 02:00:00 [01:00:00]
#   --partition P       Slurm partition [compute-p1]
#   --account A         Slurm account [POISSON_SLURM_ACCOUNT from env.sh, or Slurm default]
#   --warmup            first run the first (selected) branch alone, so shared early steps run once
#   --only LIST         rerun only these pieces, e.g. 5 or 3,7-9 (needs --pieces and
#                       --shard-order as before)
#   --no-collect        don't queue a collect job (e.g. one submission per --select group)
#   --collect-only      queue only a collect job, over all branches (with --after)
#   --after IDS         with --collect-only: start after these jobs succeeded, e.g. 4101,4102
#   --out DIR           where the collect job saves the run [results/<name>/<job id>]
#   --name NAME         job and results name [experiment file name without .py]
#   --print-only        show the sbatch commands without submitting anything
#   --print-id          print only the main job id on stdout (for scripting; messages go to stderr);
#                       with --dag --no-collect, the comma-separated ids of the last arrays
#
# DAG mode (opt-in): every stage run that isn't cached yet becomes a Slurm array task, and
# each array starts when the arrays it reads from have finished. --cpus/--time/--mem-per-cpu
# are the defaults for every task.
#   --dag                    plan here, then submit one array per group of tasks (on DelftBlue,
#                            run this on a compute node: see README "DAG mode")
#   --stage-resources S:K=V  resources for stage S's tasks, e.g. solve:cpus=8,time=02:00:00
#                            (repeatable; a pipeline's run_stage(..., resources=) wins)
#   --split-by NAME          separate arrays per option of compare(NAME), so e.g. coarse never
#                            waits for fine even with the same resources
#   --max-array-size N       split arrays larger than this [1000]
#   --no-fuse                one task per stage call; by default a chain of calls where each
#                            one only feeds the next (same cpus and memory) runs back to back
#                            in one task, with the sum of their time limits
#   --max-fused-time T       don't fuse a chain past this summed time [24:00:00, the education
#                            account limit; research accounts may set more, e.g. 5-00:00:00]
#   Rerunning the same --dag command after a failure plans and submits only what's still missing.
#
# Examples:
#   bash scripts/delftblue/submit.sh experiments/sweep.py --cpus 8 --time 02:00:00
#   # coarse and fine meshes with their own resources, then one collect job:
#   a=$(bash scripts/delftblue/submit.sh sweep.py --select mesh=coarse --no-collect --print-id)
#   b=$(bash scripts/delftblue/submit.sh sweep.py --select mesh=fine --cpus 16 --time 04:00:00 \
#         --no-collect --print-id)
#   bash scripts/delftblue/submit.sh sweep.py --collect-only --after "$a,$b"
set -euo pipefail

usage() { sed -n '2,/^set -euo/{/^#/s/^# \{0,1\}//p}' "$0"; exit "${1:-0}"; }
require_arg() { [[ $# -ge 2 ]] || { echo "submit.sh: $1 requires a value" >&2; exit 1; }; }

[[ $# -ge 1 ]] || usage 1
if [[ $1 == -h || $1 == --help ]]; then usage 0; fi
experiment="$1"; shift
[[ -f "${experiment}" ]] || { echo "submit.sh: no such file: ${experiment}" >&2; exit 1; }
experiment="$(cd "$(dirname "${experiment}")" && pwd)/$(basename "${experiment}")"

pieces="" cpus=1 mem_per_cpu=3900M time_limit=01:00:00 partition=compute-p1 account="" warmup=0
only="" out="" name="" print_only=0 extra=() warmup_id= shard_order=stride selects=()
collect=1 collect_only=0 after="" print_id=0 shard_order_set=0
dag=0 stage_resources=() split_by=() max_array_size=() no_fuse=() max_fused_time=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pieces) require_arg "$@"; pieces="$2"; shift 2 ;;
        --shard-order) require_arg "$@"; shard_order="$2"; shard_order_set=1; shift 2 ;;
        --select) require_arg "$@"; selects+=(--select "$2"); shift 2 ;;
        --cpus) require_arg "$@"; cpus="$2"; shift 2 ;;
        --mem-per-cpu) require_arg "$@"; mem_per_cpu="$2"; shift 2 ;;
        --time) require_arg "$@"; time_limit="$2"; shift 2 ;;
        --partition) require_arg "$@"; partition="$2"; shift 2 ;;
        --account) require_arg "$@"; account="$2"; shift 2 ;;
        --warmup) warmup=1; shift ;;
        --only) require_arg "$@"; only="$2"; shift 2 ;;
        --no-collect) collect=0; shift ;;
        --collect-only) collect_only=1; shift ;;
        --after) require_arg "$@"; after="$2"; shift 2 ;;
        --out) require_arg "$@"; out="$2"; shift 2 ;;
        --name) require_arg "$@"; name="$2"; shift 2 ;;
        --print-only) print_only=1; shift ;;
        --print-id) print_id=1; shift ;;
        --dag) dag=1; shift ;;
        --stage-resources) require_arg "$@"; stage_resources+=(--stage-resources "$2"); shift 2 ;;
        --split-by) require_arg "$@"; split_by=(--split-by "$2"); shift 2 ;;
        --max-array-size) require_arg "$@"; max_array_size=(--max-array-size "$2"); shift 2 ;;
        --no-fuse) no_fuse=(--no-fuse); shift ;;
        --max-fused-time) require_arg "$@"; max_fused_time=(--max-fused-time "$2"); shift 2 ;;
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

case "${shard_order}" in
    stride|block) ;;
    *) echo "submit.sh: --shard-order must be stride or block, got '${shard_order}'" >&2; exit 1 ;;
esac
if (( collect_only )) && { (( ! collect )) || (( warmup )) || [[ -n "${only}${pieces}" ]] || (( ${#selects[@]} )); }; then
    echo "submit.sh: --collect-only takes no --no-collect, --warmup, --only, --pieces or --select" >&2
    exit 1
fi
if (( dag )) && { (( warmup )) || (( collect_only )) || (( shard_order_set )) || [[ -n "${only}${pieces}" ]]; }; then
    echo "submit.sh: --dag takes no --warmup, --pieces, --only, --shard-order or --collect-only" >&2
    exit 1
fi
if (( ! dag )) && { (( ${#stage_resources[@]} )) || (( ${#split_by[@]} )) || (( ${#max_array_size[@]} )) \
    || (( ${#no_fuse[@]} )) || (( ${#max_fused_time[@]} )); }; then
    echo "submit.sh: --stage-resources, --split-by, --max-array-size, --no-fuse and --max-fused-time need --dag" >&2
    exit 1
fi
if [[ -n "${after}" ]] && (( ! collect_only )); then
    echo "submit.sh: --after only goes with --collect-only" >&2
    exit 1
fi
after_ids=()
if [[ -n "${after}" ]]; then
    if [[ ! "${after}" =~ ^[0-9]+([,:][0-9]+)*$ ]]; then
        echo "submit.sh: --after needs job ids like 4101,4102, got '${after}'" >&2
        exit 1
    fi
    IFS=',:' read -r -a after_ids <<<"${after}"
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
account="${account:-${POISSON_SLURM_ACCOUNT}}"
name="${name:-$(basename "${experiment}" .py)}"

if [[ -n "${only}" && -z "${pieces}" ]]; then
    echo "submit.sh: --only needs --pieces with the same number as the original submission" >&2
    exit 1
fi

logs="${POISSON_REPO}/logs/${name}"
mkdir -p "${logs}" "${POISSON_CACHE_DIR}"

common=(--parsable --ntasks=1 --partition="${partition}")
if [[ -n "${account}" ]]; then common+=(--account="${account}"); fi
job="${POISSON_REPO}/scripts/delftblue/job.sbatch"

say() { if (( print_id )); then echo "$@" >&2; else echo "$@"; fi; }

submit() {  # submit LABEL SBATCH_ARGS...; prints the numeric job id
    local label="$1"; shift
    if (( print_only )); then
        echo "sbatch $*" >&2
        echo "<${label}-job-id>"
        return 0
    fi
    local raw job_id
    raw="$(sbatch "$@")" || { echo "submit.sh: sbatch failed for ${label}" >&2; return 1; }
    raw="${raw//$'\r'/}"
    raw="${raw##*$'\n'}"
    job_id="${raw%%;*}"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "submit.sh: unexpected sbatch --parsable output for ${label}: '${raw}'" >&2
        return 1
    fi
    echo "${job_id}"
}

submit_collect() {  # submit_collect DEPENDENCY_IDS... ; prints the collect job id
    local dep=()
    if (( $# )); then dep=(--dependency="afterok:$(IFS=:; echo "$*")"); fi
    submit collect "${common[@]}" --job-name="${name}-collect" "${dep[@]}" \
        --cpus-per-task=1 --mem-per-cpu="${mem_per_cpu}" --time=00:30:00 --output="${logs}/collect_%j.out" \
        "${extra[@]}" "${job}" --repo "${POISSON_REPO}" "${experiment}" --collect --out "${out}" --overwrite \
        "${selects[@]}"
}

if (( collect_only )); then
    out="${out:-${POISSON_REPO}/results/${name}/collect_$(date +%Y%m%d_%H%M%S)}"
    collect_id="$(submit_collect "${after_ids[@]}")"
    if (( print_id )); then echo "${collect_id}"; exit 0; fi
    if (( print_only )); then echo; echo "(--print-only: nothing was submitted)"; exit 0; fi
    echo "Submitted ${name}: collect job ${collect_id}${after:+ after jobs ${after}}; results: ${out}/"
    exit 0
fi

if (( dag )); then
    plan="${logs}/plan_$(date +%Y%m%d_%H%M%S)_$$.json"
    say "Planning ${experiment} ..."
    plan_output="$(cd "${POISSON_REPO}" && python3 "${experiment}" --dag-plan "${plan}" \
        --resources "cpus=${cpus},time=${time_limit},mem-per-cpu=${mem_per_cpu}" \
        "${stage_resources[@]}" "${split_by[@]}" "${max_array_size[@]}" "${no_fuse[@]}" "${max_fused_time[@]}" "${selects[@]}")" || {
        echo "submit.sh: '${experiment} --dag-plan' failed; fix the experiment before submitting" >&2
        exit 1
    }
    say "$(grep -v '^dag-group ' <<<"${plan_output}")"
    declare -A job_of=()
    submitted=() leaf_ids=() arrays=0
    while IFS='|' read -r gid size parents gcpus gtime gmem gstage leaf; do
        dep=()
        if [[ "${parents}" != "-" ]]; then
            ids=()
            IFS=',' read -r -a parent_groups <<<"${parents}"
            for pg in "${parent_groups[@]}"; do ids+=("${job_of[${pg}]}"); done
            dep=(--dependency="afterok:$(IFS=:; echo "${ids[*]}")")
        fi
        if ! id="$(submit "array${gid}" "${common[@]}" "${dep[@]}" --job-name="${name}-${gstage}" \
            --array="0-$(( size - 1 ))" --cpus-per-task="${gcpus}" --mem-per-cpu="${gmem}" --time="${gtime}" \
            --output="${logs}/dag_${gstage}_%A_%a.out" \
            "${extra[@]}" "${job}" --repo "${POISSON_REPO}" --dag "${plan}" "${gid}" "${experiment}")"; then
            if (( ${#submitted[@]} )); then
                echo "submit.sh: submitting array ${gid} failed; cancel what was already queued with: scancel ${submitted[*]}" >&2
            fi
            exit 1
        fi
        job_of[${gid}]="${id}"
        submitted+=("${id}")
        arrays=$(( arrays + 1 ))
        if [[ "${leaf}" == 1 ]]; then leaf_ids+=("${id}"); fi
    done < <(sed -n 's/^dag-group //p' <<<"${plan_output}")

    collect_id=
    if (( collect )); then
        out="${out:-${POISSON_REPO}/results/${name}/dag_$(date +%Y%m%d_%H%M%S)}"
        if ! collect_id="$(submit_collect "${leaf_ids[@]}")"; then
            echo "submit.sh: submitting the collect job failed; queued arrays: ${submitted[*]:-none}" >&2
            exit 1
        fi
    fi
    if (( print_id )); then  # the collect job, or the leaf arrays to pass to --after
        if [[ -n "${collect_id}" ]]; then echo "${collect_id}"; else (IFS=,; echo "${leaf_ids[*]}"); fi
        exit 0
    fi
    if (( print_only )); then echo; echo "(--print-only: nothing was submitted)"; exit 0; fi
    cat <<EOF

Submitted ${name} as a DAG: ${arrays} arrays (${submitted[*]:-none, everything is cached})${collect_id:+, collect job ${collect_id}}.
  watch:    squeue --me
  plan:     ${plan}
  logs:     ${logs}/
${collect_id:+  results:  ${out}/  (written by the collect job once every array succeeded)
}If a task fails, the arrays after it and the collect job won't start (DependencyNeverSatisfied).
Cancel them with 'scancel ${submitted[*]:-} ${collect_id}', fix the cause, and run the same command again:
it plans again and submits only the stage runs that are still missing.
EOF
    exit 0
fi

# Dry run on the login node: catches errors in the script before anything waits in the queue.
say "Checking ${experiment} ..."
count_output="$(cd "${POISSON_REPO}" && python3 "${experiment}" --count "${selects[@]}")" || {
    echo "submit.sh: '${experiment} --count' failed; fix the experiment before submitting" >&2
    exit 1
}
say "${count_output}"
branches="$(sed -n 's/^branches: \([0-9][0-9]*\)$/\1/p' <<<"${count_output}" | tail -n 1)"
[[ -n "${branches}" ]] || { echo "submit.sh: could not read the branch count (does the script call experiment_main?)" >&2; exit 1; }
pieces="${pieces:-${branches}}"
if (( pieces < 1 )); then echo "submit.sh: nothing to run (0 branches)" >&2; exit 1; fi
if (( pieces > branches )); then
    say "Only ${branches} branches: using ${branches} pieces instead of ${pieces}"
    pieces="${branches}"
fi

dependency=()
if (( warmup )); then
    warmup_id="$(submit warmup "${common[@]}" --job-name="${name}-warmup" \
        --cpus-per-task="${cpus}" --mem-per-cpu="${mem_per_cpu}" --time="${time_limit}" \
        --output="${logs}/warmup_%j.out" \
        "${extra[@]}" "${job}" --repo "${POISSON_REPO}" "${experiment}" --warmup "${selects[@]}")"
    dependency=(--dependency="afterok:${warmup_id}")
fi

array="${only:-0-$(( pieces - 1 ))}"
pieces_id="$(submit pieces "${common[@]}" "${dependency[@]}" --job-name="${name}" --array="${array}" \
    --cpus-per-task="${cpus}" --mem-per-cpu="${mem_per_cpu}" --time="${time_limit}" \
    --output="${logs}/piece_%A_%a.out" \
    "${extra[@]}" "${job}" --repo "${POISSON_REPO}" --shard-total "${pieces}" "${experiment}" \
    --shard-order "${shard_order}" "${selects[@]}")"

collect_id=
if (( collect )); then
    out="${out:-${POISSON_REPO}/results/${name}/${pieces_id}}"
    collect_id="$(submit_collect "${pieces_id}")"
fi

if (( print_id )); then echo "${pieces_id}"; exit 0; fi
if (( print_only )); then echo; echo "(--print-only: nothing was submitted)"; exit 0; fi
again="bash scripts/delftblue/submit.sh ${experiment} --pieces ${pieces} --shard-order ${shard_order}"
for s in "${selects[@]}"; do [[ "${s}" == --select ]] || again+=" --select ${s}"; done
if (( collect )); then
    cat <<EOF

Submitted ${name}: ${pieces} pieces (array ${array}) as job ${pieces_id}, collect job ${collect_id}${warmup_id:+, warm-up job ${warmup_id}}.
  watch:    squeue --me
  logs:     ${logs}/
  results:  ${out}/  (written by the collect job once every piece succeeded)
If a piece fails, the collect job won't start (squeue shows it waiting on DependencyNeverSatisfied).
Cancel it with 'scancel ${collect_id}', then rerun only the failed pieces, e.g. pieces 3 and 7:
  ${again} --only 3,7 [--time ...]
That queues a new collect job; everything the other pieces finished is reused from the cache.
EOF
else
    cat <<EOF

Submitted ${name}: ${pieces} pieces (array ${array}) as job ${pieces_id}${warmup_id:+, warm-up job ${warmup_id}}; no collect job.
  watch:    squeue --me
  logs:     ${logs}/
Collect everything once all submissions are done, e.g.:
  bash scripts/delftblue/submit.sh ${experiment} --collect-only --after ${pieces_id}[,<other job ids>]
EOF
fi
