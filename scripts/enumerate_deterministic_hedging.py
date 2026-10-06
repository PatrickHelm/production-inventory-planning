import sys
import time
import os
import random

sys.path.append(".")

import numpy as np
from numba import prange

from src.optimization_policies import DeterministicOptimizationPolicy
from src.evaluation_utils import (
    evaluate_policy_jit,
    precompute_demand_realizations,
)
from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig
from src.parser import get_ablation_kwargs, parse_args


def enumerate_global_hedging_levels(
    config: SupplyChainConfig,
    hedging_values: np.ndarray,
    num_trajectories: int,
    periods_per_trajectory: int,
    warmup_periods: int,
    seed: int = 11,
    planning_horizon_length: int = 30,
) -> np.ndarray:
    config_jit = config.get_jit_class()
    dynamics = DynamicsJIT(config_jit, False)

    np.random.seed(seed)
    random.seed(seed)
    demand_realizations = precompute_demand_realizations(
        config_jit,
        config_jit.aggregate_demand_model_history,
        num_trajectories,
        periods_per_trajectory + warmup_periods,
        seed,
    )

    costs = np.empty(len(hedging_values))
    cost_std = np.empty(len(hedging_values))

    for i in prange(len(hedging_values)):
        policy = DeterministicOptimizationPolicy(
            config, hedging_values[i], planning_horizon_length=planning_horizon_length
        )
        costs[i], cost_std[i] = evaluate_policy_jit.py_func(
            dynamics,
            policy,
            num_trajectories,
            periods_per_trajectory,
            demand_realizations,
            warmup_periods,
        )
        print(f"Hedging level {hedging_values[i]}: Cost {costs[i]}, Std {cost_std[i]}")

    best_config_idx = np.argmin(costs)
    best_cost = costs[best_config_idx]
    best_cost_std = cost_std[best_config_idx]
    best_hedging_levels = hedging_values[best_config_idx]
    return best_cost, best_cost_std, best_hedging_levels, costs, cost_std

if __name__ == "__main__":
    args = parse_args()
    scaling = args.scaling
    supply_chain_config = SupplyChainConfig(
        scaling=scaling,
        **get_ablation_kwargs(args),
    )
    PATH = f"results/optimization/deterministic/scaling{scaling}{supply_chain_config.ablation_suffix}/"
    os.makedirs(PATH, exist_ok=True)
    num_trajectories = args.num_trajectories
    periods_per_trajectory = 60
    warmup_periods = 30
    hedging_levels_values = np.linspace(0.2, 0.32, 13)
    print(f"Enumerating global hedging levels for {args.policy} optimization policy for scaling{scaling} and {num_trajectories} trajectories...")
    print("Hedging levels to evaluate:", hedging_levels_values)
    start_time =  time.time()
    (best_cost, 
     best_cost_std, 
     best_hedging_levels, 
     all_costs, 
     all_cost_stds) = enumerate_global_hedging_levels(
        supply_chain_config,
        hedging_levels_values,
        num_trajectories,
        periods_per_trajectory,
        warmup_periods,
        seed=args.seed,
        planning_horizon_length=args.planning_horizon or 30,
    )
    print(f"Optimization took {time.time() - start_time} seconds")
    print("Best hedging level:", best_hedging_levels)
    print("Best cost:", best_cost)
    print("Best cost std:", best_cost_std)
    np.savetxt(PATH + "hedging_parameters.txt", np.around([best_hedging_levels], decimals=4), fmt="%.4f")
    np.savetxt(PATH + "all_costs.txt", np.vstack((hedging_levels_values, all_costs, all_cost_stds)).T, fmt="%.4f")
