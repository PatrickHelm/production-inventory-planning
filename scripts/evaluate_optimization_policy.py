import sys
import time
import os
import random

sys.path.append(".")

import numpy as np

from src.optimization_policies import TwoStageStochasticOptimizationPolicy, DeterministicOptimizationPolicy, MultiStageStochasticOptimizationPolicy
from src.evaluation_utils import evaluate_policy_and_track_performance
from src.dynamics import DynamicsJIT as DynamicsJIT
from src.supply_chain_config import SupplyChainConfig
from src.parser import get_ablation_kwargs, parse_args

args = parse_args()
job_id = args.job_id
n_jobs = args.n_jobs
scaling = args.scaling
total_num_trajectories = args.num_trajectories
num_trajectories = total_num_trajectories // n_jobs
start_index = (job_id - 1) * num_trajectories
end_index = start_index + num_trajectories

periods_per_trajectory = 60
seed = num_trajectories * periods_per_trajectory * job_id

supply_chain_config = SupplyChainConfig(
    scaling=scaling,
    **get_ablation_kwargs(args),
)
ablation_suffix = supply_chain_config.ablation_suffix
demand_suffix = supply_chain_config.test_demand_suffix
round_add = "_round" if args.round else ""
# Branching structure of the multi-stage scenario tree; the remaining periods are deterministic
DEFAULT_BRANCHING = [6, 4, 3, 2, 2]
if args.scenario_tree is None:
    branching = DEFAULT_BRANCHING
else:
    branching = [int(b) for b in args.scenario_tree.split("-")]
    assert all(b >= 1 for b in branching), f"Branching factors must be at least 1, got {branching}"
# Ablation suffixes: only appended for non-default settings so existing result folders stay valid
horizon_add = f"_h{args.planning_horizon}" if args.planning_horizon is not None else ""
tree_add = (
    "_tree" + "-".join(str(b) for b in branching)
    if args.policy == "multi_stage" and branching != DEFAULT_BRANCHING
    else ""
)
scenario_add = f"_s{args.num_scenarios}" if args.policy == "two_stage" and args.num_scenarios != 1000 else ""
trajectory_add = f"_n{total_num_trajectories}" if total_num_trajectories != 1000 else ""
PATH = f"results/optimization/{args.policy}/scaling{scaling}{ablation_suffix}{round_add}{horizon_add}{tree_add}{scenario_add}{trajectory_add}/"
if not os.path.exists(PATH):
    os.makedirs(PATH, exist_ok=True)
supply_chain_config_jit = supply_chain_config.get_jit_class()
dynamics_jit = DynamicsJIT(supply_chain_config_jit)

demand_realizations = np.load("data/demand_realizations_scaling"+str(scaling)+demand_suffix+".npy")[start_index:end_index]
assert demand_realizations.shape[0] == num_trajectories, f"Expected {num_trajectories} trajectories, but got {demand_realizations.shape[0]}"

aggregate_demand_model_history = np.load("data/initial_aggregate_demand_model_history_scaling"+str(scaling)+ablation_suffix+".npy")[start_index:end_index]
inventory_per_node = np.load("data/initial_inventory_per_node_scaling"+str(scaling)+ablation_suffix+".npy")[start_index:end_index]
inventory_in_pipeline_per_node = np.load("data/initial_inventory_in_pipeline_per_node_scaling"+str(scaling)+ablation_suffix+".npy")[start_index:end_index]

if args.policy == "two_stage":
    policy_jit = TwoStageStochasticOptimizationPolicy(
        supply_chain_config,
        args.num_scenarios,
        seed + job_id,
        lp_rounding=args.round,
        planning_horizon_length=args.planning_horizon,
    )
    add = str(args.num_scenarios) + " scenarios"
elif args.policy == "deterministic":
    policy_jit = DeterministicOptimizationPolicy(
        supply_chain_config,
        args.hedging,
        planning_horizon_length=args.planning_horizon or 30,
    )
    PATH = f"results/optimization/{args.policy}/scaling{scaling}{ablation_suffix}/"
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
    add = str(args.hedging) + " hedging" 
elif args.policy == "multi_stage":
    if args.planning_horizon is not None:
        tree_length = args.planning_horizon
    else:
        tree_length = 30
    assert tree_length >= len(branching), f"Planning horizon must be at least {len(branching)}"
    tree = branching + [1] * (tree_length - len(branching))
    policy_jit = MultiStageStochasticOptimizationPolicy(
        supply_chain_config,
        tree,
        seed + job_id,
        lp_rounding=args.round,
    )
    add = "scenario tree " + "-".join([str(t) for t in tree])
else:
    raise NotImplementedError(f"Policy {args.policy} not implemented.")

print(f"Evaluating {args.policy} optimization policy with {add} for {num_trajectories} trajectories for {job_id}/{n_jobs} and scaling{scaling}")
np.random.seed(42 + job_id)
random.seed(42 + job_id)
timer_start = time.time()
cost_path = PATH + f"all_costs_{job_id}.txt"
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
    cost_path
)
print(f"{num_trajectories} trajectories in {time.time() - timer_start} seconds")
print(f"Average Policy cost: {cost}")
print(f"Policy cost std: {std}")
np.savetxt(PATH + f"costs_mean_std_{job_id}.txt", np.array([cost, std]), fmt="%.6f")
np.savetxt(PATH + f"fill_rate_{job_id}.txt", np.array([service_level]), fmt="%.6f")
np.savetxt(PATH + f"holding_cost_{job_id}.txt", np.array([holding_cost]), fmt="%.6f")
np.savetxt(PATH + f"backorder_cost_{job_id}.txt", np.array([backorder_cost]), fmt="%.6f")
np.savetxt(PATH + f"inventory_distribution_{job_id}.txt", inventory_distribution, fmt="%.6f")
