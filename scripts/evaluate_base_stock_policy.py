import sys
import time

sys.path.append(".")

import numpy as np

from src.base_stock import DynamicBaseStockPolicyShortfallJIT, StaticBaseStockPolicyShortfallJIT
from src.evaluation_utils import (
    evaluate_policy_and_track_performance,
)
from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig
from src.parser import get_ablation_kwargs, parse_args

args = parse_args()

job_id = args.job_id
n_jobs = args.n_jobs
scaling = args.scaling
if n_jobs != 1:
    raise ValueError("Base-stock evaluation must run as one job per policy")
num_trajectories = args.num_trajectories
start_index = 0
end_index = num_trajectories
periods_per_trajectory = 60

supply_chain_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
ablation_suffix = supply_chain_config.ablation_suffix
demand_suffix = supply_chain_config.test_demand_suffix
seed_add = f"_seed{args.seed}" if args.seed != 42 else ""
PATH = f"results/base_stock/{args.policy}/scaling{scaling}{ablation_suffix}{seed_add}/"
hedging_parameters = np.loadtxt(PATH + "hedging_parameters.txt")

supply_chain_config_jit = supply_chain_config.get_jit_class()
dynamics_jit = DynamicsJIT(supply_chain_config_jit)

if args.policy.startswith("dynamic"):
    policy_jit = DynamicBaseStockPolicyShortfallJIT(
        supply_chain_config_jit,
        hedging_parameters,
        True,
    )
elif args.policy.startswith("static"):
    policy_jit = StaticBaseStockPolicyShortfallJIT(
        supply_chain_config_jit,
        hedging_parameters,
        True,
    )
else:
    raise ValueError(f"Unknown policy type: {args.policy}")

add = "" # "_final"

demand_realizations = np.load("data/demand_realizations_scaling"+str(scaling)+demand_suffix+add+".npy")[start_index:end_index]
assert demand_realizations.shape[0] == num_trajectories, f"Expected {num_trajectories} trajectories, but got {demand_realizations.shape[0]}"

aggregate_demand_model_history = np.load("data/initial_aggregate_demand_model_history_scaling"+str(scaling)+ablation_suffix+add+".npy")[start_index:end_index]
inventory_per_node = np.load("data/initial_inventory_per_node_scaling"+str(scaling)+ablation_suffix+add+".npy")[start_index:end_index]
inventory_in_pipeline_per_node = np.load("data/initial_inventory_in_pipeline_per_node_scaling"+str(scaling)+ablation_suffix+add+".npy")[start_index:end_index]

print(f"Evaluating {args.policy} base stock policy for scaling{scaling} for {num_trajectories} trajectories for {job_id}/{n_jobs} with parameters:")
print(hedging_parameters)
timer_start = time.time()
cost_path = PATH + "all_costs.txt"
open(cost_path, "w").close()
cost, std, holding_cost, backorder_cost, service_level, inventory_distribution = evaluate_policy_and_track_performance(
    dynamics_jit, 
    policy_jit, 
    num_trajectories, 
    periods_per_trajectory, 
    demand_realizations, 
    inventory_per_node, 
    inventory_in_pipeline_per_node, 
    aggregate_demand_model_history,
    cost_path,
)
print(f"{num_trajectories} trajectories in {time.time() - timer_start} seconds")
print(f"Average Policy cost: {cost}")
print(f"Policy cost std: {std}")
np.savetxt(PATH + "costs_mean_std.txt", np.array([cost, std]), fmt="%.6f")
np.savetxt(PATH + "fill_rate.txt", np.array([service_level]), fmt="%.6f")
np.savetxt(PATH + "holding_cost.txt", np.array([holding_cost]), fmt="%.6f")
np.savetxt(PATH + "backorder_cost.txt", np.array([backorder_cost]), fmt="%.6f")
np.savetxt(PATH + "inventory_distribution.txt", inventory_distribution, fmt="%.6f")
