import numpy as np
from numba.experimental import jitclass
from numba import njit, boolean

from src.state import StateJIT
from src.supply_chain_config import SupplyChainConfigJIT


spec = [
    ("supply_chain_config", SupplyChainConfigJIT.class_type.instance_type),
    ("check_feasible", boolean),
]

@jitclass(spec)
class DynamicsJIT:
    def __init__(self, supply_chain_config: SupplyChainConfigJIT, check_feasible: bool = False):
        self.supply_chain_config = supply_chain_config
        self.check_feasible = check_feasible
        
    # One time step of the simulation
    def transition_dynamics(
        self,
        state: StateJIT,
        action: np.ndarray,
        event: np.ndarray,
    ) -> tuple[StateJIT, float, float, np.ndarray]:
        # Check if action is feasible
        if self.check_feasible:
            if not self.feasible_action(state, action):
                raise Exception("Action is not feasible")
        # assume action is feasible
        self.update_state_with_action(state, action)

        # satisfied demand for fill rate calculation
        satisfied_demand = self.update_state_with_demand(state, event)

        # Calculate the costs
        holding_costs, backorder_costs = self.get_costs(state)

        # Transition to new time period
        self.update_state_with_transition(state)

        return state, holding_costs, backorder_costs, satisfied_demand

    def update_state_with_action(self, state: StateJIT, action: np.ndarray) -> None:
        for i in range(self.supply_chain_config.num_components):
            # Implement the production decision
            state.inventory_in_pipeline_per_node[i][
                self.supply_chain_config.lead_times[i] - 1
            ] += action[i]
            #  Remove the production decision from the on-hand inventory at predecessors
            for j in self.supply_chain_config.predecessors[
                i, : self.supply_chain_config.n_predecessors[i]
            ]:
                state.inventory_per_node[j] -= action[i]

    def update_state_with_demand(self, state: StateJIT, demand: np.ndarray) -> np.ndarray:
        # Demand served from on-hand inventory (for the fill rate); the rest is backordered
        satisfied_demand = np.minimum(
            np.maximum(0, state.inventory_per_node[
                self.supply_chain_config.num_sub_components :
            ]), demand[:-1]
        )
        state.inventory_per_node[-self.supply_chain_config.num_products :] -= demand[
            :-1
        ].astype(np.int64)
        # Update aggregate demand history
        state.aggregate_demand_model_history[1:] = state.aggregate_demand_model_history[
            :-1
        ]
        state.aggregate_demand_model_history[0] = demand[-1]
        return satisfied_demand

    def get_costs(self, state: StateJIT) -> tuple[float, float]:
        # Calculate the holding costs for the components
        holding_cost = (
            np.maximum(0, state.inventory_per_node) * self.supply_chain_config.h
        ).sum()
        # Calculate the backorder costs of the end products
        backorder_cost = (
            np.maximum(
                0,
                -state.inventory_per_node[
                    self.supply_chain_config.num_sub_components :
                ],
            )
            * self.supply_chain_config.p
        ).sum()
        return holding_cost, backorder_cost

    def update_state_with_transition(self, state: StateJIT) -> None:
        # Remove the oldest production decision from the inventory in pipeline
        state.inventory_per_node += state.inventory_in_pipeline_per_node[:, 0]
        state.inventory_in_pipeline_per_node[:, :-1] = (
            state.inventory_in_pipeline_per_node[:, 1:].copy()
        )
        state.inventory_in_pipeline_per_node[:, -1] = 0

    # Get the fixed initial state to start simulating the trajectories
    def get_initial_state(self) -> StateJIT:

        inventory_per_node = np.ceil(
            self.supply_chain_config.avg_demand_per_comp * 2.0
        ).astype(np.int64)
        inventory_in_pipeline_per_node = np.zeros(
            (
                self.supply_chain_config.num_components,
                np.max(self.supply_chain_config.lead_times),
            ),
            dtype=np.int64,
        )
        for i in range(self.supply_chain_config.num_components):
            L = int(self.supply_chain_config.lead_times[i])
            inventory_in_pipeline_per_node[i, : L - 1] = np.ceil(
                self.supply_chain_config.avg_demand_per_comp
            ).astype(np.int64)[i]

        # Aggregate demand history (first position is the most recent observation)
        aggregate_demand_model_history = (
            self.supply_chain_config.aggregate_demand_model_history.copy()
        )

        state = StateJIT(
            inventory_per_node,
            inventory_in_pipeline_per_node,
            aggregate_demand_model_history,
        )

        return state
    
    # Checks if the action is feasible in the current state
    # Returns false if the action is not feasible
    def feasible_action(self, state: StateJIT, action: np.ndarray) -> bool:
        feasible = True
        # Check if action does not exceed capacity group
        for i in range(self.supply_chain_config.num_nodes):
            capacity_usage = 0
            capacity_group_size = self.supply_chain_config.capacity_groups_size[i]
            capacity_group_components = self.supply_chain_config.capacity_groups_indices[i, : capacity_group_size]
            for j in capacity_group_components:
                capacity_usage += action[j]
            if self.supply_chain_config.capacities[i] < capacity_usage:
                feasible = False
                break

        # Check material availability
        for i in range(self.supply_chain_config.num_sub_components):
            material_usage = 0
            for j in range(self.supply_chain_config.n_successors[i]):
                succ_idx = self.supply_chain_config.successors[i, j]
                material_usage += action[succ_idx]
            if state.inventory_per_node[i] < material_usage:
                feasible = False
                break
            
        for i in range(self.supply_chain_config.num_components):
            if action[i] < 0:
                feasible = False
                break

        return feasible

# Returns the demand for this period: an array with the demand of each product,
# followed by the (fractional) aggregate demand, which enters the history that
# generates the demand of future periods
@njit
def get_demand_from_history(
    supply_chain_config: SupplyChainConfigJIT,
    aggregate_demand_model_history: np.ndarray,
    normal_noise: float,
    random_keys: np.ndarray,
    gamma_samples: np.ndarray,
):
    # Generate next period aggregate demand based on AR model
    aggregate_demand = (
        np.sum(
            supply_chain_config.aggregate_demand_ar_params
            * (
                aggregate_demand_model_history
                - supply_chain_config.aggregate_demand_mean
            )
        )
        + normal_noise
    ) + supply_chain_config.aggregate_demand_mean

    # Censor at zero
    demand = np.zeros(supply_chain_config.num_products + 1, dtype=np.float64)
    if aggregate_demand <= 0.0:
        return demand
    
    # ---- Stochastic rounding to integer ----
    base = np.floor(aggregate_demand)
    fractional_part = aggregate_demand - base
    if random_keys[0] < fractional_part:
        aggregate_demand_int = np.ceil(aggregate_demand)
    else:
        aggregate_demand_int = base

    # Generate the stochastic split parameters
    weights = gamma_samples / gamma_samples.sum()

    # ---- Deterministic floor allocation ----
    demand_decimals = np.empty(supply_chain_config.num_products, dtype=np.float64)
    for i in range(supply_chain_config.num_products):
        d_i = weights[i] * aggregate_demand_int
        demand_decimals[i] = d_i
        demand[i] = int(np.floor(d_i))

    # ---- Remainder R ----
    total_floor = demand.sum()
    remainder = aggregate_demand_int - total_floor

    if remainder > 0:
        # fractional parts
        fractional_parts = demand_decimals - demand[:-1]
        frac_sum = fractional_parts.sum()

        # build CDF of probabilities (normalized)
        cum_probs = fractional_parts.cumsum() / frac_sum

        # sample with replacement R times 
        for r in range(remainder):
            idx = np.searchsorted(cum_probs, random_keys[r + 1])
            demand[idx] += 1

    demand[-1] = aggregate_demand

    return demand
