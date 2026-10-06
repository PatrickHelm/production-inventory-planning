import sys
import os

sys.path.append(".")

import numpy as np

from src.base_stock import DynamicBaseStockPolicyShortfallJIT
from src.evaluation_utils import (
    precompute_demand_realizations,
    precompute_initial_states,
)
from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig
from src.parser import get_ablation_kwargs, parse_args

SEED = 2**32 // 2

args = parse_args()

num_trajectories = args.num_trajectories
periods_per_trajectory = 60
warmup_periods = 30
scaling = args.scaling

demand_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
supply_chain_config_jit = demand_config.get_jit_class()
demand_path = f"data/demand_realizations_scaling{scaling}{demand_config.test_demand_suffix}.npy"
warmup_demand_path = f"data/demand_realizations_warmup_scaling{scaling}{demand_config.test_demand_suffix}.npy"

if os.path.exists(demand_path) and os.path.exists(warmup_demand_path):
    demand_realizations_eval = np.load(demand_path)[:num_trajectories]
    demand_realizations_warmup = np.load(warmup_demand_path)[:num_trajectories]
    if len(demand_realizations_eval) < num_trajectories or len(demand_realizations_warmup) < num_trajectories:
        raise ValueError("Existing demand files contain fewer trajectories than requested")
else:
    print("Precomputing demand realizations for scaling", scaling)
    demand_realizations = precompute_demand_realizations(
        supply_chain_config_jit,
        supply_chain_config_jit.aggregate_demand_model_history,
        num_trajectories,
        periods_per_trajectory + warmup_periods,
        SEED,
    )
    demand_realizations_warmup = demand_realizations[:, :warmup_periods]
    demand_realizations_eval = demand_realizations[:, warmup_periods:]
    if os.path.exists(demand_path):
        existing_demand = np.load(demand_path, mmap_mode="r")[:num_trajectories]
        if existing_demand.shape != demand_realizations_eval.shape or not np.array_equal(existing_demand, demand_realizations_eval):
            raise ValueError(f"Generated test demand does not match existing file: {demand_path}")
    else:
        np.save(demand_path, demand_realizations_eval)
    np.save(warmup_demand_path, demand_realizations_warmup)

print("Precomputing initial states for scaling", scaling)
supply_chain_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
supply_chain_config_jit = supply_chain_config.get_jit_class()
dynamics_jit = DynamicsJIT(supply_chain_config_jit)
PATH = f"results/base_stock/dynamic_item/scaling{scaling}{supply_chain_config.ablation_suffix}/"
hedging_parameters = np.loadtxt(PATH + "hedging_parameters.txt")
policy_jit = DynamicBaseStockPolicyShortfallJIT(
    supply_chain_config_jit,
    hedging_parameters,
    True,
)
costs, std, aggregate_demand_model_history, inventory_per_node, inventory_in_pipeline_per_node = precompute_initial_states(
    dynamics_jit,
    policy_jit,
    num_trajectories,
    warmup_periods,
    demand_realizations_warmup,
)
print(f"Precomputed initial states for scaling{scaling}: Average Cost {costs}, Std {std}")
np.save("data/initial_aggregate_demand_model_history_scaling"+str(scaling)+supply_chain_config.ablation_suffix+".npy", aggregate_demand_model_history)
np.save("data/initial_inventory_per_node_scaling"+str(scaling)+supply_chain_config.ablation_suffix+".npy", inventory_per_node)
np.save("data/initial_inventory_in_pipeline_per_node_scaling"+str(scaling)+supply_chain_config.ablation_suffix+".npy", inventory_in_pipeline_per_node)
