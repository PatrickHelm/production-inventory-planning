import sys
import time

sys.path.append(".")

import os
import numpy as np
from numba import njit, prange

from src.base_stock import DynamicBaseStockPolicyShortfallJIT, StaticBaseStockPolicyShortfallJIT
from src.evaluation_utils import (
    evaluate_policy_jit,
    precompute_demand_realizations,
)
from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig, SupplyChainConfigJIT
from src.parser import get_ablation_kwargs, parse_args

@njit(parallel=True)
def enumerate_global_hedging_levels_static(
    config: SupplyChainConfigJIT,
    hedging_values: np.ndarray,
    num_trajectories: int,
    periods_per_trajectory: int,
    warmup_periods: int,
    static_policy: bool = False,
    seed: int = 11,
) -> np.ndarray:

    dynamics = DynamicsJIT(config, False)

    np.random.seed(seed)
    demand_realizations = precompute_demand_realizations(
        config,
        config.aggregate_demand_model_history,
        num_trajectories,
        periods_per_trajectory + warmup_periods,
        seed,
    )

    costs = np.empty(len(hedging_values))
    cost_std = np.empty(len(hedging_values))
    hedging_vectors = np.empty(
        (len(hedging_values), config.num_components)
    )
    for i in range(len(hedging_values)):
        hedging_vectors[i, :] = np.full(
            config.num_components, hedging_values[i]
        )

    for i in prange(len(hedging_vectors)):
        policy = StaticBaseStockPolicyShortfallJIT(
            config, hedging_vectors[i], True
            )
        costs[i], cost_std[i] = evaluate_policy_jit(
            dynamics,
            policy,
            num_trajectories,
            periods_per_trajectory,
            demand_realizations,
            warmup_periods,
        )

    best_config_idx = np.argmin(costs)
    best_cost = costs[best_config_idx]
    best_cost_std = cost_std[best_config_idx]
    best_hedging_levels = hedging_vectors[best_config_idx]
    return best_cost, best_cost_std, best_hedging_levels, costs, cost_std

@njit(parallel=True)
def enumerate_global_hedging_levels_dynamic(
    config: SupplyChainConfigJIT,
    hedging_values: np.ndarray,
    num_trajectories: int,
    periods_per_trajectory: int,
    warmup_periods: int,
    seed: int = 11,
) -> np.ndarray:

    dynamics = DynamicsJIT(config, False)

    np.random.seed(seed)
    demand_realizations = precompute_demand_realizations(
        config,
        config.aggregate_demand_model_history,
        num_trajectories,
        periods_per_trajectory + warmup_periods,
        seed,
    )

    costs = np.empty(len(hedging_values))
    cost_std = np.empty(len(hedging_values))
    hedging_vectors = np.empty(
        (len(hedging_values), config.num_components)
    )
    for i in range(len(hedging_values)):
        hedging_vectors[i, :] = np.full(
            config.num_components, hedging_values[i]
        )

    for i in prange(len(hedging_vectors)):
        policy = DynamicBaseStockPolicyShortfallJIT(
                config, hedging_vectors[i], True
            )
        costs[i], cost_std[i] = evaluate_policy_jit(
            dynamics,
            policy,
            num_trajectories,
            periods_per_trajectory,
            demand_realizations,
            warmup_periods,
        )

    best_config_idx = np.argmin(costs)
    best_cost = costs[best_config_idx]
    best_cost_std = cost_std[best_config_idx]
    best_hedging_levels = hedging_vectors[best_config_idx]
    return best_cost, best_cost_std, best_hedging_levels, costs, cost_std

if __name__ == "__main__":
    args = parse_args()
    scaling = args.scaling
    config = SupplyChainConfig(
        scaling=scaling,
        **get_ablation_kwargs(args),
    )
    supply_chain_config = config.get_jit_class()
    PATH = f"results/base_stock/{args.policy}/scaling{scaling}{config.ablation_suffix}/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    num_trajectories = args.num_trajectories
    periods_per_trajectory = 60
    warmup_periods = 30
    hedging_levels_values = np.linspace(1.5, 3.5, 201)
    print(f"Enumerating global hedging levels for {args.policy} base stock policy...")
    print("Hedging levels to evaluate:", hedging_levels_values)
    start_time =  time.time()
    if args.policy.startswith("dynamic"):
        (best_cost, 
         best_cost_std, 
         best_hedging_levels, 
         all_costs, 
         all_cost_stds) = enumerate_global_hedging_levels_dynamic(
            supply_chain_config,
            hedging_levels_values,
            num_trajectories,
            periods_per_trajectory,
            warmup_periods,
            seed=args.seed,
        )
    else:
        (best_cost, 
        best_cost_std, 
        best_hedging_levels, 
        all_costs, 
        all_cost_stds) = enumerate_global_hedging_levels_static(
            supply_chain_config,
            hedging_levels_values,
            num_trajectories,
            periods_per_trajectory,
            warmup_periods,
            seed=args.seed,
        )
    print(f"Optimization took {time.time() - start_time} seconds")
    print("Best hedging levels:", best_hedging_levels)
    print("Best cost:", best_cost)
    print("Best cost std:", best_cost_std)
    np.savetxt(PATH + "hedging_parameters.txt", np.around(best_hedging_levels, decimals=4), fmt="%.4f", )
    np.savetxt(PATH + "grid_costs.txt", np.vstack((hedging_levels_values, all_costs, all_cost_stds)).T, fmt="%.4f")
