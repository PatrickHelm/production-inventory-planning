from __future__ import annotations

from numba import njit, prange
from src.dynamics import get_demand_from_history
from src.dynamics import DynamicsJIT
from src.supply_chain_config import SupplyChainConfigJIT
from src.state import StateJIT
import numpy as np


@njit(parallel=False)
def evaluate_policy_jit(
    dynamics: DynamicsJIT,
    policy, # any policy with set_action(state) method
    num_trajectories: int,
    periods_per_trajectory: int,
    demand_realizations: np.ndarray,
    warm_up_periods: int = 0,
) -> tuple[float, float]:
    policy_cost = np.zeros(num_trajectories)
    for i in prange(num_trajectories):
        state = dynamics.get_initial_state()
        total_costs = 0
        for j in range(periods_per_trajectory + warm_up_periods):
            action = policy.set_action(state)
            state, holding_cost, backorder_cost, _ = dynamics.transition_dynamics(state, action, demand_realizations[i, j])
            if j >= warm_up_periods:
                total_costs += holding_cost + backorder_cost
        policy_cost[i] = total_costs / periods_per_trajectory
    avg_costs = np.sum(policy_cost) / num_trajectories
    std_costs = np.std(policy_cost)
    return avg_costs, std_costs

def evaluate_policy_and_track_performance(
    dynamics: DynamicsJIT,
    policy, # any policy with set_action(state) method
    num_trajectories: int,
    periods_per_trajectory: int,
    demand_realizations: np.ndarray,
    initial_inventory_per_node: np.ndarray,
    initial_inventory_in_pipeline_per_node: np.ndarray,
    initial_aggregate_demand_model_history: np.ndarray,
    path: str = "",
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    total_holding_costs = 0.
    total_backorder_costs = 0.
    total_demand = np.zeros(dynamics.supply_chain_config.num_products, dtype=np.float64)
    total_satisfied_demand = np.zeros(dynamics.supply_chain_config.num_products, dtype=np.float64)
    inventory_distribution = np.zeros(dynamics.supply_chain_config.num_components, dtype=np.int64)
    policy_cost = np.zeros(num_trajectories)
    for i in prange(num_trajectories):
        state = StateJIT(
            initial_inventory_per_node[i].copy(),
            initial_inventory_in_pipeline_per_node[i].copy(),
            initial_aggregate_demand_model_history[i].copy(),
            )
        total_costs = 0
        for j in range(periods_per_trajectory):
            action = policy.set_action(state)
            (state, 
             holding_cost, 
             backorder_cost, 
             satisfied_demand) = dynamics.transition_dynamics(state, action, demand_realizations[i, j])
            total_costs += holding_cost + backorder_cost
            total_holding_costs += holding_cost
            total_backorder_costs += backorder_cost
            total_demand += demand_realizations[i, j, :-1]
            total_satisfied_demand += satisfied_demand
            for k in range(dynamics.supply_chain_config.num_components):
                inventory_distribution[k] += state.inventory_per_node[k]
        policy_cost[i] = total_costs / periods_per_trajectory
        if path: 
            with open(f"{path}", "a") as f:
                f.write(f"{policy_cost[i]}\n")
    avg_costs = np.sum(policy_cost) / num_trajectories
    std_costs = np.std(policy_cost)
    avg_holding_costs = total_holding_costs / (num_trajectories * periods_per_trajectory)
    avg_backorder_costs = total_backorder_costs / (num_trajectories * periods_per_trajectory)
    service_level = total_satisfied_demand / total_demand
    inventory_distribution = inventory_distribution / (num_trajectories * periods_per_trajectory)
    return avg_costs, std_costs, avg_holding_costs, avg_backorder_costs, service_level, inventory_distribution

@njit
def precompute_initial_states(
    dynamics: DynamicsJIT,
    policy, # any policy with set_action(state) method
    num_trajectories: int,
    periods_per_trajectory: int,
    demand_realizations: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    policy_cost = np.zeros(num_trajectories)
    aggregate_demand_model_history = np.zeros((num_trajectories, dynamics.supply_chain_config.aggregate_demand_model_history.size), dtype=np.float64)
    inventory_per_node = np.zeros((num_trajectories, dynamics.supply_chain_config.num_components), dtype=np.int64)
    inventory_in_pipeline_per_node = np.zeros(
            (num_trajectories, dynamics.supply_chain_config.num_components, np.max(dynamics.supply_chain_config.lead_times)), dtype=np.int64,
        )
    for i in prange(num_trajectories):
        state = dynamics.get_initial_state()
        total_costs = 0
        for j in range(periods_per_trajectory):
            action = policy.set_action(state)
            state, holding_cost, backorder_cost, _ = dynamics.transition_dynamics(state, action, demand_realizations[i, j])
            total_costs += holding_cost + backorder_cost
        aggregate_demand_model_history[i, :] = state.aggregate_demand_model_history
        inventory_per_node[i, :] = state.inventory_per_node
        inventory_in_pipeline_per_node[i, :, :] = state.inventory_in_pipeline_per_node
        policy_cost[i] = total_costs / periods_per_trajectory
    avg_costs = np.sum(policy_cost) / num_trajectories
    std_costs = np.std(policy_cost)
    return avg_costs, std_costs, aggregate_demand_model_history, inventory_per_node, inventory_in_pipeline_per_node

@njit(parallel=False)
def precompute_demand_realizations(
    supply_chain_config: SupplyChainConfigJIT,
    aggregate_demand_model_history: np.ndarray,
    num_trajectories: int,
    periods_per_trajectory: int,
    seed: int = -1,
) -> np.ndarray:
    # Generate enough randomness for the whole long trajectory
    normal_noise, random_keys, gamma_samples = get_samples(
        supply_chain_config, num_trajectories, periods_per_trajectory, seed
    )
    demands = np.zeros((num_trajectories, 
                        periods_per_trajectory, 
                        supply_chain_config.num_products + 1), dtype=np.float64)
    for k in range(num_trajectories):
        new_aggregate_demand_model_history = aggregate_demand_model_history.copy()
        for t in range(periods_per_trajectory):
            demand = get_demand_from_history(
                supply_chain_config,
                new_aggregate_demand_model_history,
                normal_noise[k, t],
                random_keys[k, t],
                gamma_samples[k, t],
            )
            new_aggregate_demand_model_history[1:] = new_aggregate_demand_model_history[:-1]
            new_aggregate_demand_model_history[0] = demand[-1]
            demands[k, t, :] = demand 
    return demands

@njit(parallel=False)
def get_samples(
    supply_chain_config: SupplyChainConfigJIT,
    num_trajectories: int,
    periods_per_trajectory: int,
    seed: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed >= 0:
        np.random.seed(seed)
    normal_samples = supply_chain_config.aggregate_demand_error_sigma * np.random.standard_normal(
        size=(num_trajectories, periods_per_trajectory),
    )
    uniform_random_keys = np.random.random(
        size=(
            num_trajectories,
            periods_per_trajectory,
            supply_chain_config.num_products + 1,
        )
    )
    # numba's np.random.gamma requires scalar shape/scale, so draw per product
    gamma_samples = np.empty(
        (num_trajectories, periods_per_trajectory, supply_chain_config.num_products),
        dtype=np.float64,
    )
    for n in range(supply_chain_config.num_products):
        gamma_samples[:, :, n] = np.random.gamma(
            supply_chain_config.product_demand_split_kappa[n],
            supply_chain_config.product_demand_split_theta[n],
            (num_trajectories, periods_per_trajectory),
        )
    return normal_samples, uniform_random_keys, gamma_samples