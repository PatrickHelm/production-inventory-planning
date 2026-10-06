#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ab_init "$@"
ab_require_states

TUNING_TRAJECTORIES="${TUNING_TRAJECTORIES:-1000}"
if ! [[ "$TUNING_TRAJECTORIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "TUNING_TRAJECTORIES must be positive" >&2
    exit 2
fi

pids=()
for index in "${!AB_SETTINGS[@]}"; do
    setting="${AB_SETTINGS[$index]}"
    (
        ab_setting "$setting"
        cpu="$(ab_cpu_for_setting "$index")"
        grid_dir="results/optimization/deterministic/scaling${AB_SCALING}${AB_SUFFIX}"
        ab_run_if_missing "$setting deterministic hedge grid" "$cpu" \
            "$grid_dir/hedging_parameters.txt" "$grid_dir/all_costs.txt" -- \
            enumerate_deterministic_hedging.py --policy deterministic --scaling "$AB_SCALING" \
            --planning_horizon "$AB_HORIZON" --num_trajectories "$TUNING_TRAJECTORIES" "${AB_FLAG[@]}"
        hedge="$(awk 'NR == 1 {printf "%g", $1; exit}' "$grid_dir/hedging_parameters.txt")"
        if [[ -z "$hedge" ]]; then
            echo "Missing deterministic hedge in $grid_dir/hedging_parameters.txt" >&2
            exit 1
        fi
        ab_run_sharded "$setting deterministic evaluation (hedge $hedge)" "$cpu" \
            evaluate_optimization_policy.py --policy deterministic --scaling "$AB_SCALING" \
            --planning_horizon "$AB_HORIZON" --hedging "$hedge" \
            --num_trajectories "$AB_TRAJECTORIES" "${AB_FLAG[@]}"
    ) >"$AB_LOG_DIR/${setting}_deterministic.out" 2>&1 &
    pids+=("$!")
done
ab_wait_all "${pids[@]}"
echo "Deterministic tuning and evaluations complete. Logs: $AB_LOG_DIR"
