import sys
import os

sys.path.append(".")

import numpy as np

from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig
from src.hedging_optimizer import HedgingOptimizer, optimize_static, optimize_dynamic
from src.parser import get_ablation_kwargs, parse_args

args = parse_args()

seed = args.seed
scaling = args.scaling
num_generations = 100
num_trajectories = 1_000
num_trajectories_increment = 100
warmup_periods = 30
periods_per_trajectory = 60
initial_sigma = 0.5
population_size = 0  # 0 means default CMA-ES population size

seed_sequence = np.random.SeedSequence(seed)
candidate_seed_sequence, demand_seed_sequence = seed_sequence.spawn(2)
candidate_seeds = candidate_seed_sequence.generate_state(num_generations)
demand_seeds = demand_seed_sequence.generate_state(num_generations)

supply_chain_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
supply_chain_config_jit = supply_chain_config.get_jit_class()
ablation_suffix = supply_chain_config.ablation_suffix
# Seed suffix: only appended for non-default seeds so existing result folders stay valid
seed_add = f"_seed{seed}" if seed != 42 else ""
LOAD_PATH = f"results/base_stock/{args.policy.replace('item', 'global')}/scaling{scaling}{ablation_suffix}/"
SAVE_PATH = f"results/base_stock/{args.policy}/scaling{scaling}{ablation_suffix}{seed_add}/"
os.makedirs(SAVE_PATH, exist_ok=True)
hedging_parameter_global = np.loadtxt(LOAD_PATH + "hedging_parameters.txt")

cma_es = HedgingOptimizer(hedging_parameter_global, initial_sigma, population_size, seed)
dynamics = DynamicsJIT(supply_chain_config_jit)

use_static_policy = args.policy.startswith("static")

if use_static_policy:
    print("Optimizing hedging parameters for static base stock policy...")
    mean_history, sigma_history = optimize_static(
        cma_es, dynamics, num_generations, candidate_seeds, demand_seeds,
        num_trajectories, periods_per_trajectory, warmup_periods, num_trajectories_increment
    )
else:
    print("Optimizing hedging parameters for dynamic base stock policy...")
    mean_history, sigma_history = optimize_dynamic(
        cma_es, dynamics, num_generations, candidate_seeds, demand_seeds,
        num_trajectories, periods_per_trajectory, warmup_periods, num_trajectories_increment
    )
np.savetxt(SAVE_PATH + "mean_history.txt", mean_history, fmt="%.6f")
np.savetxt(SAVE_PATH + "sigma_history.txt", sigma_history, fmt="%.6f")
np.savetxt(SAVE_PATH + "hedging_parameters.txt", mean_history[-1, :], fmt="%.6f")
