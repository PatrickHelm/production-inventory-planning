#!/usr/bin/env bash
# Shared configuration for the five settings of the paper: the base case and four variants.

AB_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_SETTINGS=(scaling1 scaling3 scaling1_lower_capacities scaling1_higher_backorder_costs scaling1_stationary_demand)

ab_init() {
    if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
        echo "Usage: ./$(basename "$0") N  (N: CPUs per setting)" >&2
        exit 2
    fi
    AB_CPUS="$1"
    AB_TRAJECTORIES="${NUM_TRAJECTORIES:-1000}"
    AB_CPU_START="${CPU_START:-0}"
    AB_HORIZON=30
    if ! [[ "$AB_TRAJECTORIES" =~ ^[1-9][0-9]*$ && "$AB_CPU_START" =~ ^[0-9]+$ ]]; then
        echo "NUM_TRAJECTORIES and CPU_START must be valid integers" >&2
        exit 2
    fi
    if (( AB_CPUS > AB_TRAJECTORIES )); then
        echo "N cannot exceed NUM_TRAJECTORIES" >&2
        exit 2
    fi
    AB_LOG_DIR="${LOG_DIR:-${AB_REPO_DIR}/logs/ablation/$(basename "$0" .sh)-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$AB_LOG_DIR"
    cd "$AB_REPO_DIR"
}

ab_setting() {
    local setting="$1"
    AB_FLAG=()
    AB_DEMAND_SUFFIX=""
    case "$setting" in
        scaling1) AB_SCALING=1; AB_SUFFIX="" ;;
        scaling3) AB_SCALING=3; AB_SUFFIX="" ;;
        scaling1_lower_capacities)
            AB_SCALING=1; AB_SUFFIX="_lower_capacities"
            AB_FLAG=(--lower_capacities) ;;
        scaling1_higher_backorder_costs)
            AB_SCALING=1; AB_SUFFIX="_higher_backorder_costs"
            AB_FLAG=(--higher_backorder_costs) ;;
        scaling1_stationary_demand)
            AB_SCALING=1; AB_SUFFIX="_stationary_demand"; AB_DEMAND_SUFFIX="_stationary_demand"
            AB_FLAG=(--stationary_demand) ;;
        *) echo "Unknown setting: $setting" >&2; return 2 ;;
    esac
}

ab_cpu_for_setting() {
    echo "$((AB_CPU_START + $1 * AB_CPUS))"
}

ab_run() {
    local label="$1" cpu="$2"
    shift 2
    echo "Running $label on CPUs $cpu-$((cpu + AB_CPUS - 1))"
    "$AB_REPO_DIR/run.sh" -i "$cpu" -c "$AB_CPUS" "$@"
}

ab_run_if_missing() {
    local label="$1" cpu="$2" missing=0
    shift 2
    while [[ "$1" != "--" ]]; do
        [[ -f "$1" ]] || missing=1
        shift
    done
    shift
    if (( missing == 0 )); then
        echo "Skipping $label: outputs exist"
    else
        ab_run "$label" "$cpu" "$@"
    fi
}

ab_run_if_missing_on_cpus() {
    local label="$1" cpu="$2" cpus="$3" missing=0
    shift 3
    while [[ "$1" != "--" ]]; do
        [[ -f "$1" ]] || missing=1
        shift
    done
    shift
    if (( missing == 0 )); then
        echo "Skipping $label: outputs exist"
    else
        echo "Running $label on CPUs $cpu-$((cpu + cpus - 1))"
        "$AB_REPO_DIR/run.sh" -i "$cpu" -c "$cpus" "$@"
    fi
}

ab_run_sharded() {
    local label="$1" cpu="$2"
    shift 2
    echo "Running $label on $AB_CPUS jobs, CPUs $cpu-$((cpu + AB_CPUS - 1))"
    "$AB_REPO_DIR/run.sh" -i "$cpu" -j "$AB_CPUS" "$@"
}

ab_require_states() {
    local setting
    for setting in "${AB_SETTINGS[@]}"; do
        ab_setting "$setting"
        local stem="data/initial_"
        local file
        for file in \
            "data/demand_realizations_scaling${AB_SCALING}${AB_DEMAND_SUFFIX}.npy" \
            "${stem}aggregate_demand_model_history_scaling${AB_SCALING}${AB_SUFFIX}.npy" \
            "${stem}inventory_per_node_scaling${AB_SCALING}${AB_SUFFIX}.npy" \
            "${stem}inventory_in_pipeline_per_node_scaling${AB_SCALING}${AB_SUFFIX}.npy"; do
            if [[ ! -f "$file" ]]; then
                echo "Missing $file; run ./run_setup_base_stock.sh $AB_CPUS first" >&2
                return 1
            fi
        done
        if [[ "$setting" == scaling1_lower_capacities || "$setting" == scaling1_higher_backorder_costs ]]; then
            local marker="data/initial_states_scaling1${AB_SUFFIX}_baseline_demand.txt"
            local marker_count=0
            if [[ -f "$marker" ]]; then
                read -r marker_count < "$marker" || marker_count=0
            fi
            if ! [[ "$marker_count" =~ ^[0-9]+$ ]] || (( marker_count < AB_TRAJECTORIES )); then
                echo "States for $setting may use old demand; run ./run_setup_base_stock.sh $AB_CPUS first" >&2
                return 1
            fi
        fi
    done
}

ab_wait_all() {
    local pid failed=0
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        echo "A setting failed; see $AB_LOG_DIR" >&2
        return 1
    fi
}

ab_run_stochastic() {
    local policy="$1"
    shift
    ab_init "$@"
    ab_require_states
    local num_scenarios="${NUM_SCENARIOS:-1000}"
    if [[ "$policy" == two_stage ]] && ! [[ "$num_scenarios" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_SCENARIOS must be a positive integer" >&2
        return 2
    fi
    local pids=() index setting cpu
    for index in "${!AB_SETTINGS[@]}"; do
        setting="${AB_SETTINGS[$index]}"
        (
            ab_setting "$setting"
            cpu="$(ab_cpu_for_setting "$index")"
            if [[ "$policy" == two_stage ]]; then
                ab_run_sharded "$setting $policy evaluation" "$cpu" \
                    evaluate_optimization_policy.py --policy "$policy" --scaling "$AB_SCALING" \
                    --round --planning_horizon "$AB_HORIZON" --num_scenarios "$num_scenarios" \
                    --num_trajectories "$AB_TRAJECTORIES" "${AB_FLAG[@]}"
            else
                ab_run_sharded "$setting $policy evaluation" "$cpu" \
                    evaluate_optimization_policy.py --policy "$policy" --scaling "$AB_SCALING" \
                    --round --planning_horizon "$AB_HORIZON" \
                    --num_trajectories "$AB_TRAJECTORIES" "${AB_FLAG[@]}"
            fi
        ) >"$AB_LOG_DIR/${setting}_${policy}.out" 2>&1 &
        pids+=("$!")
    done
    ab_wait_all "${pids[@]}"
    echo "$policy evaluations complete. Logs: $AB_LOG_DIR"
}
