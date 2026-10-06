#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ab_init "$@"

TUNING_TRAJECTORIES="${TUNING_TRAJECTORIES:-1000}"
INITIAL_STATE_TRAJECTORIES="${INITIAL_STATE_TRAJECTORIES:-$AB_TRAJECTORIES}"
if ! [[ "$TUNING_TRAJECTORIES" =~ ^[1-9][0-9]*$ && "$INITIAL_STATE_TRAJECTORIES" =~ ^[1-9][0-9]*$ ]] ||
    (( INITIAL_STATE_TRAJECTORIES < AB_TRAJECTORIES )); then
    echo "TUNING_TRAJECTORIES must be positive and INITIAL_STATE_TRAJECTORIES must cover NUM_TRAJECTORIES" >&2
    exit 2
fi

echo "Step 1/4: shared precomputations"
mkdir -p data/precomputed
ab_run_if_missing "allocation rounding variance" "$AB_CPU_START" \
    data/precomputed/allocation_rounding_variance.txt -- \
    simulate_rounding_variance.py --policy dynamic_item --scaling 1

for scaling in 1 3; do
    ab_run_if_missing "scaling $scaling demand share variance" "$AB_CPU_START" \
        "data/precomputed/variance_product_shares_echelon_lead_time_scaling${scaling}.txt" -- \
        simulate_demand_share.py --policy dynamic_item --scaling "$scaling"
    ab_run_if_missing "scaling $scaling static demand distribution" "$AB_CPU_START" \
        "data/precomputed/static_echelon_lead_time_demand_mean_scaling${scaling}.txt" \
        "data/precomputed/static_echelon_lead_time_demand_std_scaling${scaling}.txt" -- \
        simulate_static_demand_distribution.py --policy dynamic_item --scaling "$scaling"
done
ab_run_if_missing "stationary static demand distribution" "$AB_CPU_START" \
    data/precomputed/static_echelon_lead_time_demand_mean_scaling1_stationary_demand.txt \
    data/precomputed/static_echelon_lead_time_demand_std_scaling1_stationary_demand.txt -- \
    simulate_static_demand_distribution.py --policy dynamic_item --scaling 1 --stationary_demand

echo "Step 2/4: global and item-level base-stock tuning"
pids=()
for index in "${!AB_SETTINGS[@]}"; do
    setting="${AB_SETTINGS[$index]}"
    (
        ab_setting "$setting"
        cpu="$(ab_cpu_for_setting "$index")"
        tune_policy() {
            local policy="$1" tune_cpu="$2" tune_cpus="$3"
            local dir="results/base_stock/${policy}/scaling${AB_SCALING}${AB_SUFFIX}"
            if [[ "$policy" == *_global ]]; then
                ab_run_if_missing_on_cpus "$setting $policy grid" "$tune_cpu" "$tune_cpus" \
                    "$dir/hedging_parameters.txt" "$dir/grid_costs.txt" -- \
                    enumerate_global_hedging.py --policy "$policy" --scaling "$AB_SCALING" \
                    --num_trajectories "$TUNING_TRAJECTORIES" "${AB_FLAG[@]}"
            else
                ab_run_if_missing_on_cpus "$setting $policy tuning" "$tune_cpu" "$tune_cpus" \
                    "$dir/hedging_parameters.txt" "$dir/mean_history.txt" "$dir/sigma_history.txt" -- \
                    run_optimization.py --policy "$policy" --scaling "$AB_SCALING" "${AB_FLAG[@]}"
            fi
        }
        for phase in global item; do
            if (( AB_CPUS == 1 )); then
                tune_policy "dynamic_${phase}" "$cpu" 1
                tune_policy "static_${phase}" "$cpu" 1
            else
                dynamic_cpus="$(((AB_CPUS + 1) / 2))"
                static_cpus="$((AB_CPUS / 2))"
                tune_policy "dynamic_${phase}" "$cpu" "$dynamic_cpus" \
                    >"$AB_LOG_DIR/${setting}_dynamic_${phase}_tuning.out" 2>&1 &
                dynamic_pid="$!"
                tune_policy "static_${phase}" "$((cpu + dynamic_cpus))" "$static_cpus" \
                    >"$AB_LOG_DIR/${setting}_static_${phase}_tuning.out" 2>&1 &
                static_pid="$!"
                ab_wait_all "$dynamic_pid" "$static_pid"
            fi
        done
    ) >"$AB_LOG_DIR/${setting}_tuning_step.out" 2>&1 &
    pids+=("$!")
done
ab_wait_all "${pids[@]}"

# The baseline, capacity, and backorder settings share scaling-1 demand.
# Generate baseline demand before either variant reads it.
echo "Step 3/4: test demand and initial states"
launch_initial_states() {
    local index="$1" setting="${AB_SETTINGS[$1]}"
    (
        ab_setting "$setting"
        cpu="$(ab_cpu_for_setting "$index")"
        state_files=( \
            "data/demand_realizations_scaling${AB_SCALING}${AB_DEMAND_SUFFIX}.npy" \
            "data/demand_realizations_warmup_scaling${AB_SCALING}${AB_DEMAND_SUFFIX}.npy" \
            "data/initial_aggregate_demand_model_history_scaling${AB_SCALING}${AB_SUFFIX}.npy" \
            "data/initial_inventory_per_node_scaling${AB_SCALING}${AB_SUFFIX}.npy" \
            "data/initial_inventory_in_pipeline_per_node_scaling${AB_SCALING}${AB_SUFFIX}.npy" \
        )
        state_command=( \
            precompute_initial_states.py --policy dynamic_item --scaling "$AB_SCALING" \
            --num_trajectories "$INITIAL_STATE_TRAJECTORIES" "${AB_FLAG[@]}" \
        )
        if [[ "$setting" == scaling1_lower_capacities || "$setting" == scaling1_higher_backorder_costs ]]; then
            marker="data/initial_states_scaling1${AB_SUFFIX}_baseline_demand.txt"
            marker_count=0
            if [[ -f "$marker" ]]; then
                read -r marker_count < "$marker" || marker_count=0
            fi
            if [[ "$marker_count" =~ ^[0-9]+$ ]] && (( marker_count >= INITIAL_STATE_TRAJECTORIES )); then
                ab_run_if_missing "$setting initial states" "$cpu" "${state_files[@]}" -- "${state_command[@]}"
            else
                ab_run "$setting initial states" "$cpu" "${state_command[@]}"
                printf '%s\n' "$INITIAL_STATE_TRAJECTORIES" > "$marker"
            fi
        else
            ab_run_if_missing "$setting initial states" "$cpu" "${state_files[@]}" -- "${state_command[@]}"
        fi
    ) >"$AB_LOG_DIR/${setting}_initial_states.out" 2>&1 &
    initial_state_pid="$!"
}
pids=()
for index in 0 1 4; do
    launch_initial_states "$index"
    if (( index == 0 )); then
        baseline_pid="$initial_state_pid"
    else
        pids+=("$initial_state_pid")
    fi
done
if ! wait "$baseline_pid"; then
    ab_wait_all "${pids[@]}" || true
    echo "Scaling-1 baseline initial states failed; see $AB_LOG_DIR" >&2
    exit 1
fi
launch_initial_states 2
pids+=("$initial_state_pid")
launch_initial_states 3
pids+=("$initial_state_pid")
ab_wait_all "${pids[@]}"

echo "Step 4/4: base-stock evaluations"
pids=()
for index in "${!AB_SETTINGS[@]}"; do
    setting="${AB_SETTINGS[$index]}"
    (
        ab_setting "$setting"
        cpu="$(ab_cpu_for_setting "$index")"
        policy_pids=()
        policies=(dynamic_item static_item dynamic_global static_global)
        for policy_index in "${!policies[@]}"; do
            policy="${policies[$policy_index]}"
            policy_cpu="$((cpu + policy_index % AB_CPUS))"
            echo "Starting $setting $policy evaluation on CPU $policy_cpu"
            "$AB_REPO_DIR/run.sh" -i "$policy_cpu" \
                evaluate_base_stock_policy.py --policy "$policy" --scaling "$AB_SCALING" \
                --num_trajectories "$AB_TRAJECTORIES" "${AB_FLAG[@]}" \
                >"$AB_LOG_DIR/${setting}_${policy}_eval.out" 2>&1 &
            policy_pids+=("$!")
            if (( ${#policy_pids[@]} == AB_CPUS )); then
                ab_wait_all "${policy_pids[@]}"
                policy_pids=()
            fi
        done
        if (( ${#policy_pids[@]} > 0 )); then
            ab_wait_all "${policy_pids[@]}"
        fi
    ) >"$AB_LOG_DIR/${setting}_evaluation.out" 2>&1 &
    pids+=("$!")
done
ab_wait_all "${pids[@]}"
echo "Setup and base-stock evaluations complete. Logs: $AB_LOG_DIR"
