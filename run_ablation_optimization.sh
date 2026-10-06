#!/usr/bin/env bash
# Ablation of the stochastic optimization policies: LP rounding vs MIP,
# planning horizon 20/40, the number of scenarios of the two-stage policy and
# the scenario tree of the multi-stage policy.
# Each configuration changes exactly one factor relative to its policy baseline.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

CPU_START="${1:-0}"
NUM_JOBS="${2:-5}"

SCALING="${SCALING:-1}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-100}"
NUM_SCENARIOS="${NUM_SCENARIOS:-1000}"
CPU_LIMIT="${CPU_LIMIT:-$(nproc)}"
LOG_DIR="${LOG_DIR:-${repo_dir}/logs/ablation_opt/$(date +%Y%m%d-%H%M%S)}"

usage() {
    cat <<'EOF' >&2
Usage:
  ./run_ablation_optimization.sh [CPU_START] [NUM_JOBS]

Arguments:
  CPU_START   First CPU index used by the ablation (default: 0)
  NUM_JOBS    Jobs each configuration is split into, one CPU per job
              (default: 5, must divide NUM_TRAJECTORIES)

Environment overrides:
  SCALING=1            Supply chain scaling factor
  NUM_TRAJECTORIES=100 Evaluation trajectories per configuration
  NUM_SCENARIOS=1000   Scenario count for the two-stage policy
  CPU_LIMIT=$(nproc)   Highest CPU index + 1 the ablation may use
  LOG_DIR=...          Where per-configuration logs are written
  DRY_RUN=1            Print the commands instead of running them

The 12 configurations are, per policy: baseline, MIP (no rounding),
horizon 20 and horizon 40, plus 500 and 2000 scenarios for the two-stage
policy and the trees 6-4-3-2 and 6-4-3-2-2-2 for the multi-stage policy.
Each one uses NUM_JOBS CPUs.
EOF
    exit "${1:-1}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage 0
fi
if [[ $# -gt 2 ]]; then
    usage
fi
for value in "$CPU_START" "$NUM_JOBS" "$NUM_TRAJECTORIES" "$NUM_SCENARIOS" "$SCALING" "$CPU_LIMIT"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "All arguments and numeric overrides must be non-negative integers: got '${value}'" >&2
        exit 1
    fi
done
if [[ "$NUM_JOBS" -lt 1 ]]; then
    echo "NUM_JOBS must be at least 1" >&2
    exit 1
fi
if (( NUM_TRAJECTORIES % NUM_JOBS != 0 )); then
    echo "NUM_TRAJECTORIES (${NUM_TRAJECTORIES}) must be divisible by NUM_JOBS (${NUM_JOBS})" >&2
    exit 1
fi

cpus_per_config="$NUM_JOBS"
if (( CPU_START + cpus_per_config > CPU_LIMIT )); then
    echo "A single configuration needs ${cpus_per_config} CPUs starting at ${CPU_START}, which exceeds CPU_LIMIT=${CPU_LIMIT}" >&2
    exit 1
fi

# label | policy | extra arguments
# The multi-stage MIP runs more than three times longer than any other configuration,
# so it is scheduled first and takes the lowest CPU block. The CPUs above it are
# released as the shorter configurations finish, which leaves the free CPUs
# consecutive while the MIP is still running.
configurations=(
    "multi_stage_mip|multi_stage|"
    "two_stage_baseline|two_stage|--round --num_scenarios ${NUM_SCENARIOS}"
    "two_stage_mip|two_stage|--num_scenarios ${NUM_SCENARIOS}"
    "two_stage_h20|two_stage|--round --num_scenarios ${NUM_SCENARIOS} --planning_horizon 20"
    "two_stage_h40|two_stage|--round --num_scenarios ${NUM_SCENARIOS} --planning_horizon 40"
    "multi_stage_baseline|multi_stage|--round"
    "multi_stage_h20|multi_stage|--round --planning_horizon 20"
    "multi_stage_h40|multi_stage|--round --planning_horizon 40"
    "two_stage_s500|two_stage|--round --num_scenarios 500"
    "two_stage_s2000|two_stage|--round --num_scenarios 2000"
    "multi_stage_tree6-4-3-2|multi_stage|--round --scenario_tree 6-4-3-2"
    "multi_stage_tree6-4-3-2-2-2|multi_stage|--round --scenario_tree 6-4-3-2-2-2"
)

pids=()
labels=()
next_cpu="$CPU_START"

wait_for_running() {
    local failed=0 index
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "Finished ${labels[$index]}"
        else
            echo "Failed ${labels[$index]} (see ${LOG_DIR}/${labels[$index]}.out)" >&2
            failed=1
        fi
    done
    pids=()
    labels=()
    next_cpu="$CPU_START"
    return "$failed"
}

mkdir -p "$LOG_DIR"
echo "Ablation on CPUs ${CPU_START}-$((CPU_LIMIT - 1)), ${#configurations[@]} configurations x ${NUM_JOBS} jobs = $((${#configurations[@]} * NUM_JOBS)) CPUs in total"
echo "Trajectories per configuration: ${NUM_TRAJECTORIES} (scaling ${SCALING})"
echo "Logs: ${LOG_DIR}"

exit_code=0
for configuration in "${configurations[@]}"; do
    IFS='|' read -r label policy extra <<<"$configuration"

    # Start a new wave once the remaining CPUs cannot hold another configuration
    if (( next_cpu + cpus_per_config > CPU_LIMIT )); then
        echo "CPU budget full, waiting for the running configurations"
        wait_for_running || exit_code=1
    fi

    command=(
        "$repo_dir/run.sh" -i "$next_cpu" -j "$NUM_JOBS"
        evaluate_optimization_policy.py
        --policy "$policy" --scaling "$SCALING"
        ${extra} --num_trajectories "$NUM_TRAJECTORIES"
    )

    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "[dry run] CPUs ${next_cpu}-$((next_cpu + cpus_per_config - 1)): ${command[*]}"
    else
        echo "Starting ${label} on CPUs ${next_cpu}-$((next_cpu + cpus_per_config - 1))"
        "${command[@]}" >"${LOG_DIR}/${label}.out" 2>&1 &
        pids+=("$!")
        labels+=("$label")
    fi
    next_cpu=$((next_cpu + cpus_per_config))
done

if [[ -z "${DRY_RUN:-}" ]]; then
    wait_for_running || exit_code=1
fi

if [[ "$exit_code" -ne 0 ]]; then
    echo "Some configurations failed, see ${LOG_DIR}" >&2
    exit 1
fi
echo "All ablation configurations completed. Results under results/optimization/{two_stage,multi_stage}/"
