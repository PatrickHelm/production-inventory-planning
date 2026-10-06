import sys
sys.path.append(".")

from src.base_stock import StaticBaseStockPolicyShortfallJIT
from src.dynamics import DynamicsJIT, get_demand_from_history
from src.evaluation_utils import get_samples
from src.supply_chain_config import SupplyChainConfig, SupplyChainConfigJIT
from src.parser import get_ablation_kwargs, parse_args

from numba import njit, prange
import numpy as np
import time

@njit(parallel=False)
def simulate_rollouts(
    supply_chain_config: SupplyChainConfigJIT,
    echelon_lead_times_per_product: np.ndarray,
    aggregate_demand_model_history: np.ndarray,
    n_rollouts: int,
    n_logs: int,                 # lead-time demand samples per rollout
):
    # Use the max echelon lead time as block size
    T = np.int64(np.max(echelon_lead_times_per_product))
    total_blocks = n_logs
    total_T = T * total_blocks

    # One lead-time demand sample per block of length T
    lead_time_demand_per_component = np.zeros(
        (n_rollouts, n_logs, supply_chain_config.num_components), dtype=np.int64
    )

    # Generate enough randomness for the whole long trajectory
    normal_noise, random_keys, gamma_samples = get_samples(
        supply_chain_config, n_rollouts, total_T
    )

    for rollout in prange(n_rollouts):
        new_aggregate_demand_model_history = aggregate_demand_model_history.copy()
        # Reuse the same block buffer
        demands = np.zeros((T, supply_chain_config.num_products), dtype=np.int64)

        # Simulate block by block
        for b in range(total_blocks):
            base = b * T

            # Fill one block of length T, carrying the history forward
            for t in range(T):
                demand = get_demand_from_history(
                    supply_chain_config,
                    new_aggregate_demand_model_history,
                    normal_noise[rollout, base + t],
                    random_keys[rollout, base + t],
                    gamma_samples[rollout, base + t],
                )
                new_aggregate_demand_model_history[1:] = new_aggregate_demand_model_history[:-1]
                new_aggregate_demand_model_history[0] = demand[-1]
                demands[t, :] = demand[:-1]   # realized per-product demand

            for j in range(supply_chain_config.num_components):
                component_total_demand = 0
                for i in range(supply_chain_config.num_products):
                    L = echelon_lead_times_per_product[i, j]
                    if L > 0:
                        # lead-time demand: the last L periods of the block
                        component_total_demand += demands[T - L : T, i].sum()
                lead_time_demand_per_component[rollout, b, j] = component_total_demand

    # Flatten back to (samples, components) for convenience, like a distribution sample set
    return lead_time_demand_per_component.reshape(
        n_rollouts * n_logs, supply_chain_config.num_components
    )

args = parse_args()
scaling = args.scaling
supply_chain_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
supply_chain_config_jit = supply_chain_config.get_jit_class(load_precomputed=False)
dynamics_jit = DynamicsJIT(supply_chain_config_jit)
policy_jit = StaticBaseStockPolicyShortfallJIT(
    supply_chain_config_jit, np.zeros(supply_chain_config_jit.num_components),
)


echelon_leadtime = policy_jit.get_echelon_lead_time_per_product()

rollouts = 1000
periods = 10000

samples = simulate_rollouts(
    supply_chain_config_jit,
    echelon_leadtime,
    supply_chain_config_jit.aggregate_demand_model_history,
    2,
    2,
)
start = time.time()
samples = simulate_rollouts(
    supply_chain_config_jit,
    echelon_leadtime,
    supply_chain_config_jit.aggregate_demand_model_history,
    rollouts,
    periods, 
)
end = time.time()
print(f"Simulated {rollouts} rollouts of {periods} periods in {end - start} seconds")
static_mean = policy_jit.get_base_stock_levels()
print("Static Mean", static_mean)
print("Mean", samples.mean(axis=0))
std = samples.std(axis=0)
print("Std", std)
print("correlation", np.cov(samples[:, -3:], rowvar=False))
np.savetxt(f"data/precomputed/static_echelon_lead_time_demand_mean_scaling{supply_chain_config.scaling}{supply_chain_config.static_demand_suffix}.txt", static_mean, fmt='%.6f')
np.savetxt(f"data/precomputed/static_echelon_lead_time_demand_std_scaling{supply_chain_config.scaling}{supply_chain_config.static_demand_suffix}.txt", std, fmt='%.6f')
