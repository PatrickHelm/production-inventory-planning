#!/usr/bin/env bash
set -euo pipefail

usage() {
    local exit_code="${1:-1}"
    cat <<'EOF' >&2
Usage:
  ./run.sh [wrapper options] [script] [script args...]

Examples:
  ./run.sh simulate_rounding_variance.py --policy dynamic_item --scaling 1
  ./run.sh evaluate_base_stock_policy.py --policy dynamic_item --scaling 1 --num_trajectories 100
  ./run.sh -j 5 -d evaluate_optimization_policy.py --policy two_stage --scaling 1 --round --num_trajectories 1000
  ./run.sh run_optimization.py --policy dynamic_item --scaling 1 --stationary_demand

Wrapper options:
  -i, --cpu-start N      First CPU index for --cpuset-cpus (default: 0)
  -j, --num-jobs N       Number of Docker jobs to launch (default: 1)
  -c, --cpus-per-job N   CPUs per Docker job (default: 1)
  -e, --seed-offset N    Job id offset; launched jobs use job_id = seed_offset + job (default: 0)
  -m, --memory S         Docker memory limit, e.g. 10g or 8192m (default: 50g)
  -o, --output PATH      Log file path for one detached job, or prefix for multiple jobs
  -n, --container-name S Container name prefix override
  -d, --detach           Run via nohup in the background
  -h, --help             Show this message

Notes:
  - All arguments after the script name are passed to the Python script unchanged,
    except that this wrapper appends --n_jobs and --job_id.
  - Do not pass --job_id or --n_jobs yourself; the wrapper manages them.
  - Scripts may be given with or without the .py suffix, and with or without the scripts/ prefix.
  - The Docker image is taken from $IMAGE (default: production-inventory-planning).
  - A non-empty gurobi.lic in the repository root is used as the Gurobi license.
EOF
    exit "${exit_code}"
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${IMAGE:-production-inventory-planning}"
script=""
cpu_start=0
num_jobs=1
cpus_per_job=1
seed_offset=0
memory="50g"
detach=0
output_path=""
container_name_override=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--cpu-start)
            [[ $# -ge 2 ]] || usage
            cpu_start="$2"
            shift 2
            ;;
        -j|--num-jobs)
            [[ $# -ge 2 ]] || usage
            num_jobs="$2"
            shift 2
            ;;
        -c|--cpus-per-job)
            [[ $# -ge 2 ]] || usage
            cpus_per_job="$2"
            shift 2
            ;;
        -e|--seed-offset)
            [[ $# -ge 2 ]] || usage
            seed_offset="$2"
            shift 2
            ;;
        -m|--memory)
            [[ $# -ge 2 ]] || usage
            memory="$2"
            shift 2
            ;;
        -o|--output)
            [[ $# -ge 2 ]] || usage
            output_path="$2"
            shift 2
            ;;
        -n|--container-name)
            [[ $# -ge 2 ]] || usage
            container_name_override="$2"
            shift 2
            ;;
        -d|--detach)
            detach=1
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -gt 0 && "$1" != -* ]]; then
    script="$1"
    shift
fi

if [[ -z "$script" ]]; then
    echo "No script provided." >&2
    usage
fi

script="${script#scripts/}"
if [[ "$script" != *.py ]]; then
    script="${script}.py"
fi

script_path="${repo_dir}/scripts/${script}"
if [[ ! -f "$script_path" ]]; then
    echo "Script not found: ${script_path}" >&2
    usage
fi

if ! [[ "$cpu_start" =~ ^[0-9]+$ && "$num_jobs" =~ ^[0-9]+$ && "$cpus_per_job" =~ ^[0-9]+$ && "$seed_offset" =~ ^[0-9]+$ ]]; then
    echo "Wrapper options must be non-negative integers." >&2
    exit 1
fi
if [[ "$num_jobs" -lt 1 || "$cpus_per_job" -lt 1 ]]; then
    echo "num_jobs and cpus_per_job must be at least 1." >&2
    exit 1
fi

for arg in "$@"; do
    if [[ "$arg" == "--job_id" || "$arg" == "--n_jobs" ]]; then
        echo "Do not pass ${arg}; run.sh manages job sharding itself." >&2
        exit 1
    fi
done

make_output_path() {
    local base="$1"
    local job_id="$2"
    local cpu_label="$3"

    if [[ -z "$base" ]]; then
        echo "${repo_dir}/nohup${cpu_label}.out"
        return
    fi
    if [[ "$num_jobs" -eq 1 ]]; then
        echo "$base"
        return
    fi
    local dir name stem ext
    dir="$(dirname "$base")"
    name="$(basename "$base")"
    if [[ "$name" == *.* ]]; then
        stem="${name%.*}"
        ext=".${name##*.}"
    else
        stem="$name"
        ext=""
    fi
    echo "${dir}/${stem}_job${job_id}${ext}"
}

for (( job=1; job<=num_jobs; job++ )); do
    job_id=$((seed_offset + job))
    start=$((cpu_start + (job - 1) * cpus_per_job))
    end=$((start + cpus_per_job - 1))
    cpu_range="${start}-${end}"

    if [[ -n "$container_name_override" ]]; then
        container_name="${container_name_override}_job${job_id}_cpuset${cpu_range}"
    else
        container_name="$(basename "${script}" .py)_job${job_id}_cpuset${cpu_range}_$(date +%Y%m%d-%H%M%S)"
    fi

    docker_cmd=(
        docker run
        --rm
        -v "${repo_dir}:/mnt/mydata/"
        -w /mnt/mydata
        --cpuset-cpus "${cpu_range}"
        -e "NUMBA_NUM_THREADS=${cpus_per_job}"
        --name "${container_name}"
    )
    if [[ -n "$memory" ]]; then
        docker_cmd+=(--memory "${memory}")
    fi
    # An empty gurobi.lic is the placeholder of a fresh clone; Gurobi rejects it,
    # so only point Gurobi at the file once a license has been pasted in
    if [[ -s "${repo_dir}/gurobi.lic" ]]; then
        docker_cmd+=(-e GRB_LICENSE_FILE=/mnt/mydata/gurobi.lic)
    fi
    docker_cmd+=(
        "${image}"
        python -u "scripts/${script}"
    )
    if [[ $# -gt 0 ]]; then
        docker_cmd+=("$@")
    fi
    docker_cmd+=(--n_jobs "${num_jobs}" --job_id "${job_id}")

    if [[ "$detach" -eq 1 ]]; then
        log_path="$(make_output_path "$output_path" "$job_id" "$start")"
        mkdir -p "$(dirname "$log_path")"
        nohup "${docker_cmd[@]}" >> "${log_path}" 2>&1 &
        echo "Started ${container_name}"
        echo "CPUs: ${cpu_range}"
        echo "Log: ${log_path}"
    elif [[ "$num_jobs" -eq 1 ]]; then
        exec "${docker_cmd[@]}"
    else
        "${docker_cmd[@]}" &
        job_pids+=("$!")
    fi
done

if [[ "$detach" -eq 0 && "$num_jobs" -gt 1 ]]; then
    failed=0
    for pid in "${job_pids[@]}"; do
        wait "$pid" || failed=1
    done
    exit "$failed"
fi
