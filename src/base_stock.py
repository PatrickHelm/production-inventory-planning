import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba import types

from src.state import StateJIT
from src.supply_chain_config import SupplyChainConfigJIT
from src.ar_utils import build_aggregate_rounding_variance, build_echelon_horizon_weight_matrix, build_ar_transition_matrix, impulse_response_row, precompute_mean_path_operator, mean_path_ARp_from_R

# Rules for jit classes: https://numba.readthedocs.io/en/stable/user/jitclass.html
# - All attributes must be typed in spec
# - Types cannot be ambiguous or changed after initialization
# - Lists are not supported, use numpy arrays instead

spec = [
    ("supply_chain_config", SupplyChainConfigJIT.class_type.instance_type),
    ("hedging_parameter", float64[:]),
    ("base_stock_levels", int64[:]),
    ("unique_divergent_components", int64[:]),
    ("is_divergent_node", types.boolean[:]),
    ("num_components", int64),
    ("use_std", types.boolean)
]


@jitclass(spec)
class StaticBaseStockPolicyShortfallJIT:
    def __init__(
        self, supply_chain_config: SupplyChainConfigJIT, hedging_parameter: np.ndarray, use_std: bool = False
    ):
        self.supply_chain_config = supply_chain_config
        self.num_components = self.supply_chain_config.num_components
        self.use_std = use_std
        self.set_hedging_parameter(hedging_parameter)

        # Determine the parts after a divergent allocation decision
        self.is_divergent_node = supply_chain_config.capacity_groups_size > 1
        n_unique_divergent_components = supply_chain_config.capacity_groups_size[
            self.is_divergent_node
        ].sum()
        self.unique_divergent_components = np.zeros(
            n_unique_divergent_components, dtype=np.int64
        )
        idx = 0
        for i in range(supply_chain_config.num_components):
            if self.is_divergent_node[self.supply_chain_config.capacity_groups[i]]:
                self.unique_divergent_components[idx] = i
                idx += 1

    def set_hedging_parameter(self, hedging_parameter: np.ndarray):
        self.hedging_parameter = hedging_parameter
        self.base_stock_levels = self.get_base_stock_levels()

    def set_action(self, state: StateJIT):
        action_per_component = np.zeros(
            self.num_components, dtype=np.int64
        )

        # Decide the components in a fixed order
        indices = np.arange(self.num_components)

        # Compute echelon inventory
        echelon_inventory = self.compute_echelon_inventory_position(state)

        # Decide production for each node
        remaining = np.ones(self.num_components, dtype=np.int64)
        while np.sum(remaining) > 0:
            # index i is the current node to decide for
            i = indices[np.argmax(remaining)]
            capacity_group_i = self.supply_chain_config.capacity_groups[i]

            # If part of allocation rule
            if i in self.unique_divergent_components:
                action_per_component = self.action_divergent_parts(
                    state, echelon_inventory, i, action_per_component
                )
                for j in self.supply_chain_config.capacity_groups_indices[
                    capacity_group_i,
                    : self.supply_chain_config.capacity_groups_size[capacity_group_i],
                ]:
                    remaining[j] = 0
            else:
                # Raise the echelon inventory to its base-stock level if possible
                action_per_component[i] += max(
                    self.base_stock_levels[i] - echelon_inventory[i], 0
                )
                capacity = self.supply_chain_config.capacities[capacity_group_i]
                used_capacity = 0
                for j in self.supply_chain_config.capacity_groups_indices[
                    capacity_group_i,
                    : self.supply_chain_config.capacity_groups_size[capacity_group_i],
                ]:
                    used_capacity += action_per_component[j]
                # Check if capacity group is not exceeded
                if used_capacity > capacity:
                    action_per_component[i] -= used_capacity - capacity

                # Check if material availability is not exceeded
                for j in self.supply_chain_config.predecessors[
                    i, : self.supply_chain_config.n_predecessors[i]
                ]:
                    successor_action = 0
                    for k in self.supply_chain_config.successors[
                        j, : self.supply_chain_config.n_successors[j]
                    ]:
                        successor_action += action_per_component[k]
                    if state.inventory_per_node[j] < successor_action:
                        action_per_component[i] -= (
                            successor_action - state.inventory_per_node[j]
                        )

            remaining[i] = 0

        return action_per_component

    # Function to determine action for the allocation based on minimizing shortfall
    def action_divergent_parts(
        self, state, echelon_inventory, component, action_per_component
    ):
        remaining = np.zeros(self.num_components, dtype=np.int64)
        # Find other nodes to decide for in the divergent part
        capacity_group = self.supply_chain_config.capacity_groups[component]
        n_components = self.supply_chain_config.capacity_groups_size[capacity_group]
        nodes_allocation = np.zeros(n_components, dtype=np.int64)
        idx = 0
        for j in self.supply_chain_config.capacity_groups_indices[
            capacity_group, :n_components
        ]:
            nodes_allocation[idx] = j
            remaining[j] = 1
            idx += 1
        while True:
            # Compute the shortfall of those nodes
            current_allocation = np.zeros(np.sum(remaining), dtype=np.int64)
            idx = 0
            for j in nodes_allocation:
                if remaining[j] == 1:
                    current_allocation[idx] = j
                    idx += 1
            expected_shortfall = self.compute_expected_shortfall(
                current_allocation, echelon_inventory
            )

            idx_highest_shortfall = np.argmax(expected_shortfall)
            component_highest_shortfall = current_allocation[idx_highest_shortfall]

            # Add one unit to the node with the highest expected shortfall
            # This will later be removed if the action is infeasible
            action_per_component[component_highest_shortfall] += 1
            echelon_inventory[component_highest_shortfall] += 1

            # If even the highest shortfall is zero, all base-stock levels are reached
            if np.max(expected_shortfall) == 0:
                # Then stop iterating and return action as all nodes are at base-stock level
                action_per_component[component_highest_shortfall] -= 1
                echelon_inventory[component_highest_shortfall] -= 1
                break

            # If action is not feasible, try producing for other nodes
            if not self.check_action_feasibility(
                state, action_per_component, component_highest_shortfall
            ):
                action_per_component[component_highest_shortfall] -= 1
                echelon_inventory[component_highest_shortfall] -= 1
                remaining[component_highest_shortfall] = 0

            # If there is no feasible action for all nodes, stop
            if np.sum(remaining) == 0:
                break

        return action_per_component

    def check_action_feasibility(self, state: StateJIT, action_per_node, node):
        feasible = True
        # Check if materials are available for all predecessors of this node
        for j in self.supply_chain_config.predecessors[
            node, : self.supply_chain_config.n_predecessors[node]
        ]:
            successor_action = 0
            for k in self.supply_chain_config.successors[
                j, : self.supply_chain_config.n_successors[j]
            ]:
                successor_action += action_per_node[k]
            if state.inventory_per_node[j] < successor_action:
                feasible = False
                break

        # Check if capacity group is not exceeded
        used_capacity = np.zeros(self.supply_chain_config.num_nodes, dtype=np.int64)
        for i in range(self.num_components):
            capacity_group_i = self.supply_chain_config.capacity_groups[i]
            used_capacity[capacity_group_i] += action_per_node[i]
        if (used_capacity > self.supply_chain_config.capacities).any():
            feasible = False
        return feasible

    def compute_expected_shortfall(self, components, echelon_inventory):
        expected_shortfall = np.zeros(len(components))
        for idx, i in enumerate(components):
            expected_shortfall[idx] = max(
                0,
                (self.base_stock_levels[i] - echelon_inventory[i])
                / self.base_stock_levels[i],
            )
        return expected_shortfall

    def compute_echelon_inventory_position(self, state: StateJIT):
        # Compute echelon inventory
        total_inventory_per_components = np.zeros(
            self.num_components, dtype=np.int64
        )
        echelon_inventory = np.zeros(
            self.num_components, dtype=np.int64
        )
        # Inventory on-hand and in pipeline of own node
        for i in range(self.num_components):
            total_inventory_per_components[i] = state.inventory_per_node[i] + np.sum(
                state.inventory_in_pipeline_per_node[i]
            )
            echelon_inventory[i] = total_inventory_per_components[i]
        # Add the inventory position of the successors to compute echelon inventory position
        for i in range(self.num_components - 1, -1, -1):
            for j in self.supply_chain_config.predecessors[
                i, : self.supply_chain_config.n_predecessors[i]
            ]:
                echelon_inventory[j] += echelon_inventory[i]

        return echelon_inventory

    def get_base_stock_levels(self):
        base_stock_levels = np.zeros(
            self.num_components, dtype=np.int64
        )

        # Compute the lead time per product per component, lead time is zero if that component is not part of the product
        echelon_lead_times_per_product = self.get_echelon_lead_time_per_product()
        lead_time_demand_per_product = np.zeros(
            (
                self.supply_chain_config.num_products,
                self.num_components,
            ),
            dtype=np.int64,
        )
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.num_components):
                lead_time_demand_per_product[i, j] = (
                    self.supply_chain_config.avg_demand_per_product[i]
                    * echelon_lead_times_per_product[i, j]
                )

        # Compute the base-stock levels by summing the lead time demand per component
        for i in range(self.num_components):
            for j in range(self.supply_chain_config.num_products):
                base_stock_levels[i] += lead_time_demand_per_product[j, i]

        # Perform hedging for the base-stock policy
        for i in range(self.num_components):
            if self.use_std:
                base_stock_levels[i] = np.ceil(
                    base_stock_levels[i] + self.hedging_parameter[i] * self.supply_chain_config.echelon_lead_time_demand_std[i]
                )
            else:
                base_stock_levels[i] = np.ceil(
                    base_stock_levels[i] * (1 + self.hedging_parameter[i])
                )

        return base_stock_levels

    def get_echelon_lead_time_per_product(self):
        # compute echelon lead times per product
        echelon_lead_times_component_per_product = np.zeros(
            (
                self.supply_chain_config.num_products,
                self.num_components,
            ),
            dtype=np.int64,
        )
        for i in range(self.supply_chain_config.num_products):
            predecessors_i = self.supply_chain_config.predecessors[
                self.supply_chain_config.num_sub_components + i,
                : self.supply_chain_config.n_predecessors[
                    self.supply_chain_config.num_sub_components + i
                ],
            ]
            echelon_lead_times_component_per_product[
                i, self.supply_chain_config.num_sub_components + i
            ] = self.supply_chain_config.lead_times[
                self.supply_chain_config.num_sub_components + i
            ]
            for j in predecessors_i:
                echelon_lead_times_component_per_product[i, j] = (
                    echelon_lead_times_component_per_product[
                        i, self.supply_chain_config.num_sub_components + i
                    ]
                )
            # For suppliers
            # Here we assume that the supply chain has no more than 3 tiers
            n_pred_of_pred_nodes = 0
            for pred in predecessors_i:
                n_pred_of_pred_nodes += self.supply_chain_config.n_predecessors[pred]
            pred_of_pred_nodes = np.zeros(n_pred_of_pred_nodes, dtype=np.int64)
            idx = 0
            for pred in predecessors_i:
                echelon_lead_times_component_per_product[
                    i, pred
                ] += self.supply_chain_config.lead_times[pred]
                for j in self.supply_chain_config.predecessors[
                    pred, : self.supply_chain_config.n_predecessors[pred]
                ]:
                    pred_of_pred_nodes[idx] = j
                    echelon_lead_times_component_per_product[i, j] = (
                        echelon_lead_times_component_per_product[i, pred]
                    )
                    idx += 1

            for pred in pred_of_pred_nodes:
                echelon_lead_times_component_per_product[
                    i, pred
                ] += self.supply_chain_config.lead_times[pred]

        # Add one period to echelon lead time due to periodic review
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.num_components):
                if echelon_lead_times_component_per_product[i, j] > 0:
                    echelon_lead_times_component_per_product[i, j] += 1
        return echelon_lead_times_component_per_product


spec = [
    ("supply_chain_config", SupplyChainConfigJIT.class_type.instance_type),
    ("hedging_parameter", float64[:]),
    ("base_stock_levels", int64[:]),
    ("echelon_lead_times_per_product", int64[:, :]),
    ("expected_echelon_lead_time_demand", float64[:]),
    ("unique_divergent_components", int64[:]),
    ("is_divergent_node", types.boolean[:]),
    ("use_std", types.boolean),
    ("maximum_echelon_lead_time", int64),
    ("ar_transition_matrix", float64[:, :]),
    ("ar_innovation_loading", float64[:]),
    ("mean_path_operator", float64[:, :]),
    ("echelon_horizon_weights", float64[:, :]),
    ("num_components", int64),
    ("innovation_variance", float64),
    ("impulse_response", float64[:]),
    ("aggregate_rounding_variance", float64[:]),
    ("allocation_rounding_variance", float64[:]),
]


@jitclass(spec)
class DynamicBaseStockPolicyShortfallJIT:
    def __init__(
        self, supply_chain_config: SupplyChainConfigJIT, hedging_parameter: np.ndarray, use_std: bool = False,
    ):
        self.supply_chain_config = supply_chain_config
        self.num_components = self.supply_chain_config.num_components
        self.use_std = use_std
        self.set_hedging_parameter(hedging_parameter)
        self.echelon_lead_times_per_product = self.get_echelon_lead_time_per_product()
        self.maximum_echelon_lead_time = np.max(self.echelon_lead_times_per_product)
        self.expected_echelon_lead_time_demand = self.get_expected_echelon_lead_time_demand(self.echelon_lead_times_per_product)

        # Determine the parts after a divergent allocation decision
        self.is_divergent_node = supply_chain_config.capacity_groups_size > 1
        n_unique_divergent_components = supply_chain_config.capacity_groups_size[
            self.is_divergent_node
        ].sum()
        self.unique_divergent_components = np.zeros(
            n_unique_divergent_components, dtype=np.int64
        )
        idx = 0
        for i in range(supply_chain_config.num_components):
            if self.is_divergent_node[self.supply_chain_config.capacity_groups[i]]:
                self.unique_divergent_components[idx] = i
                idx += 1
        self.ar_transition_matrix = build_ar_transition_matrix(self.supply_chain_config.aggregate_demand_ar_params)
        self.ar_innovation_loading = np.zeros(self.ar_transition_matrix.shape[0], dtype=np.float64)
        self.ar_innovation_loading[0] = 1.0
        self.mean_path_operator = precompute_mean_path_operator(self.ar_transition_matrix, self.maximum_echelon_lead_time)
        self.echelon_horizon_weights = build_echelon_horizon_weight_matrix(self.echelon_lead_times_per_product, self.maximum_echelon_lead_time, self.supply_chain_config.expected_product_shares)
        self.innovation_variance = self.supply_chain_config.aggregate_demand_error_sigma ** 2
        self.impulse_response = impulse_response_row(self.ar_transition_matrix, self.ar_innovation_loading, self.maximum_echelon_lead_time)
        self.aggregate_rounding_variance = build_aggregate_rounding_variance(self.echelon_horizon_weights)
        
    def set_hedging_parameter(self, hedging_parameter: np.ndarray):
        self.hedging_parameter = hedging_parameter

    def set_action(self, state: StateJIT):
        base_stock_levels = self.get_base_stock_levels(state)
        action_per_component = np.zeros(
            self.supply_chain_config.num_components, dtype=np.int64
        )

        # Decide the components in a fixed order
        indices = np.arange(self.num_components)

        # Compute echelon inventory
        echelon_inventory = self.compute_echelon_inventory_position(state)

        # Decide production for each node
        remaining = np.ones(self.num_components, dtype=np.int64)
        while np.sum(remaining) > 0:
            # index i is the current node to decide for
            i = indices[np.argmax(remaining)]
            capacity_group_i = self.supply_chain_config.capacity_groups[i]

            # If part of allocation rule
            if i in self.unique_divergent_components:
                action_per_component = self.action_divergent_parts(
                    state, echelon_inventory, i, action_per_component, base_stock_levels
                )
                for j in self.supply_chain_config.capacity_groups_indices[
                    capacity_group_i,
                    : self.supply_chain_config.capacity_groups_size[capacity_group_i],
                ]:
                    remaining[j] = 0
            else:
                # Raise the echelon inventory to its base-stock level if possible
                action_per_component[i] += max(
                    base_stock_levels[i] - echelon_inventory[i], 0
                )
                capacity = self.supply_chain_config.capacities[capacity_group_i]
                used_capacity = 0
                for j in self.supply_chain_config.capacity_groups_indices[
                    capacity_group_i,
                    : self.supply_chain_config.capacity_groups_size[capacity_group_i],
                ]:
                    used_capacity += action_per_component[j]
                # Check if capacity group is not exceeded
                if used_capacity > capacity:
                    action_per_component[i] -= used_capacity - capacity

                # Check if material availability is not exceeded
                for j in self.supply_chain_config.predecessors[
                    i, : self.supply_chain_config.n_predecessors[i]
                ]:
                    successor_action = 0
                    for k in self.supply_chain_config.successors[
                        j, : self.supply_chain_config.n_successors[j]
                    ]:
                        successor_action += action_per_component[k]
                    if state.inventory_per_node[j] < successor_action:
                        action_per_component[i] -= (
                            successor_action - state.inventory_per_node[j]
                        )

            remaining[i] = 0

        return action_per_component

    # Function to determine action for the allocation based on minimizing shortfall
    def action_divergent_parts(
        self, state, echelon_inventory, component, action_per_component, base_stock_levels
    ):
        remaining = np.zeros(self.num_components, dtype=np.int64)
        # Find other nodes to decide for in the divergent part
        capacity_group = self.supply_chain_config.capacity_groups[component]
        n_components = self.supply_chain_config.capacity_groups_size[capacity_group]
        nodes_allocation = np.zeros(n_components, dtype=np.int64)
        idx = 0
        for j in self.supply_chain_config.capacity_groups_indices[
            capacity_group, :n_components
        ]:
            nodes_allocation[idx] = j
            remaining[j] = 1
            idx += 1
        while True:
            # Compute the shortfall of those nodes
            current_allocation = np.zeros(np.sum(remaining), dtype=np.int64)
            idx = 0
            for j in nodes_allocation:
                if remaining[j] == 1:
                    current_allocation[idx] = j
                    idx += 1
            expected_shortfall = self.compute_expected_shortfall(
                current_allocation, echelon_inventory, base_stock_levels
            )

            idx_highest_shortfall = np.argmax(expected_shortfall)
            component_highest_shortfall = current_allocation[idx_highest_shortfall]

            # Add one unit to the node with the highest expected shortfall
            # This will later be removed if the action is infeasible
            action_per_component[component_highest_shortfall] += 1
            echelon_inventory[component_highest_shortfall] += 1

            # If even the highest shortfall is zero, all base-stock levels are reached
            if np.max(expected_shortfall) == 0:
                # Then stop iterating and return action as all nodes are at base-stock level
                action_per_component[component_highest_shortfall] -= 1
                echelon_inventory[component_highest_shortfall] -= 1
                break

            # If action is not feasible, try producing for other nodes
            if not self.check_action_feasibility(
                state, action_per_component, component_highest_shortfall
            ):
                action_per_component[component_highest_shortfall] -= 1
                echelon_inventory[component_highest_shortfall] -= 1
                remaining[component_highest_shortfall] = 0

            # If there is no feasible action for all nodes, stop
            if np.sum(remaining) == 0:
                break

        return action_per_component

    def check_action_feasibility(self, state: StateJIT, action_per_node, node):
        action = True
        # Check if materials are available for all predecessors of this node
        for j in self.supply_chain_config.predecessors[
            node, : self.supply_chain_config.n_predecessors[node]
        ]:
            successor_action = 0
            for k in self.supply_chain_config.successors[
                j, : self.supply_chain_config.n_successors[j]
            ]:
                successor_action += action_per_node[k]
            if state.inventory_per_node[j] < successor_action:
                action = False
                break

        # Check if capacity group is not exceeded
        used_capacity = np.zeros(self.supply_chain_config.num_nodes, dtype=np.int64)
        for i in range(self.supply_chain_config.num_components):
            capacity_group_i = self.supply_chain_config.capacity_groups[i]
            used_capacity[capacity_group_i] += action_per_node[i]
        if (used_capacity > self.supply_chain_config.capacities).any():
            action = False
        return action

    def compute_expected_shortfall(self, components, echelon_inventory, base_stock_levels):
        expected_shortfall = np.zeros(len(components))
        for idx, i in enumerate(components):
            expected_shortfall[idx] = max(
                0,
                (base_stock_levels[i] - echelon_inventory[i])
                / base_stock_levels[i],
            )
        return expected_shortfall

    def compute_echelon_inventory_position(self, state: StateJIT):
        # Compute echelon inventory
        echelon_inventory = np.zeros(
            self.supply_chain_config.num_components, dtype=np.int64
        )
        # Inventory on-hand and in pipeline of own node
        for i in range(self.supply_chain_config.num_components):
            echelon_inventory[i] = state.inventory_per_node[i] + np.sum(
                state.inventory_in_pipeline_per_node[i]
            )
        # Add the inventory position of the successors to compute echelon inventory position
        for i in range(self.num_components - 1, -1, -1):
            for j in self.supply_chain_config.predecessors[
                i, : self.supply_chain_config.n_predecessors[i]
            ]:
                echelon_inventory[j] += echelon_inventory[i]

        return echelon_inventory

    def get_base_stock_levels(self, state: StateJIT):
        base_stock_levels = np.zeros(
            self.supply_chain_config.num_components, dtype=np.int64
        )
        mean_lead_time_demand_per_component = self.compute_means(state.aggregate_demand_model_history)
        if self.use_std:
            std_lead_time_demand_per_component = np.sqrt(self.compute_variances(state.aggregate_demand_model_history))
            
        # Perform hedging for the base-stock policy
        for i in range(self.num_components): 
            if self.use_std:
                base_stock_levels[i] = np.ceil(mean_lead_time_demand_per_component[i] + self.hedging_parameter[i] * std_lead_time_demand_per_component[i])
            else:
                safety_stock_mean = self.expected_echelon_lead_time_demand[i]
                base_stock_levels[i] = np.ceil(
                    mean_lead_time_demand_per_component[i]
                    + safety_stock_mean * self.hedging_parameter[i]
                )
        return base_stock_levels

    def get_echelon_lead_time_per_product(self):
        # compute echelon lead times per product
        echelon_lead_times_component_per_product = np.zeros(
            (
                self.supply_chain_config.num_products,
                self.num_components,
            ),
            dtype=np.int64,
        )
        for i in range(self.supply_chain_config.num_products):
            predecessors_i = self.supply_chain_config.predecessors[
                self.supply_chain_config.num_sub_components + i,
                : self.supply_chain_config.n_predecessors[
                    self.supply_chain_config.num_sub_components + i
                ],
            ]
            echelon_lead_times_component_per_product[
                i, self.supply_chain_config.num_sub_components + i
            ] = self.supply_chain_config.lead_times[
                self.supply_chain_config.num_sub_components + i
            ]
            for j in predecessors_i:
                echelon_lead_times_component_per_product[i, j] = (
                    echelon_lead_times_component_per_product[
                        i, self.supply_chain_config.num_sub_components + i
                    ]
                )
            # For suppliers
            # Here we assume that the supply chain has no more than 3 tiers
            n_pred_of_pred_nodes = 0
            for pred in predecessors_i:
                n_pred_of_pred_nodes += self.supply_chain_config.n_predecessors[pred]
            pred_of_pred_nodes = np.zeros(n_pred_of_pred_nodes, dtype=np.int64)
            idx = 0
            for pred in predecessors_i:
                echelon_lead_times_component_per_product[
                    i, pred
                ] += self.supply_chain_config.lead_times[pred]
                for j in self.supply_chain_config.predecessors[
                    pred, : self.supply_chain_config.n_predecessors[pred]
                ]:
                    pred_of_pred_nodes[idx] = j
                    echelon_lead_times_component_per_product[i, j] = (
                        echelon_lead_times_component_per_product[i, pred]
                    )
                    idx += 1

            for pred in pred_of_pred_nodes:
                echelon_lead_times_component_per_product[
                    i, pred
                ] += self.supply_chain_config.lead_times[pred]

        # Add one period to echelon lead time due to periodic review
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.num_components):
                if echelon_lead_times_component_per_product[i, j] > 0:
                    echelon_lead_times_component_per_product[i, j] += 1
        return echelon_lead_times_component_per_product
    
    def get_expected_echelon_lead_time_demand(self, echelon_lead_times_per_product: np.ndarray):
        expected_echelon_lead_time_demand = np.zeros(
            self.num_components, dtype=np.float64
        )

        # Compute the lead time per product per component, lead time is zero if that component is not part of the product
        lead_time_demand_per_product = np.zeros(
            (
                self.supply_chain_config.num_products,
                self.num_components,
            ),
            dtype=np.float64,
        )
        for i in range(self.supply_chain_config.num_products):
            for j in range(self.num_components):
                lead_time_demand_per_product[i, j] = (
                    self.supply_chain_config.avg_demand_per_product[i]
                    * echelon_lead_times_per_product[i, j]
                )

        # Compute the base-stock levels by summing the lead time demand per component
        for i in range(self.num_components):
            for j in range(self.supply_chain_config.num_products):
                expected_echelon_lead_time_demand[i] += lead_time_demand_per_product[j, i]
        return expected_echelon_lead_time_demand

    def compute_means(self, state_aggregate_demand_model_history):
        """ Online: returns E[LTD_i | x_t] for all components (i=0..num_components-1). """
        x_t = state_aggregate_demand_model_history
        m = mean_path_ARp_from_R(self.supply_chain_config.aggregate_demand_mean, self.mean_path_operator, x_t)
        # means = W @ m
        out = np.empty(self.num_components, dtype=np.float64)
        for i in range(self.num_components):
            s = 0.0
            for k in range(self.maximum_echelon_lead_time):
                s += self.echelon_horizon_weights[i, k] * m[k]
            out[i] = s
        return out
    
    def compute_variances(
        self,
        aggregate_demand_model_history: np.ndarray,
    ) -> np.ndarray:
        """
        Var(S_i | x_t) = sigma^2 * sum_n [ sum_{k>=n} w_i[k] h[k-n] ]^2
                    + sigma^2 * sum_n sum_{k>=n} h[k-n]^2 Var_c[i,k]
                    + sum_k m[k]^2 Var_c[i,k]
                    + allocation rounding variance of component i
                    + aggregate rounding variance of component i
        """
        nC, L_max = self.echelon_horizon_weights.shape
        m = mean_path_ARp_from_R(self.supply_chain_config.aggregate_demand_mean, self.mean_path_operator, aggregate_demand_model_history)
        # rounding of the product split, precomputed by scripts/simulate_rounding_variance.py
        var_rounding = self.supply_chain_config.allocation_rounding_variance
        
        out = np.empty(nC, dtype=np.float64)
        for i in range(nC):
            w = self.echelon_horizon_weights[i, :]
            Var_c_row = self.supply_chain_config.variance_product_shares[i, :]

            # process variance using mean weights
            var_proc = 0.0
            for n in range(L_max):
                b = 0.0
                for k in range(n, L_max):
                    b += w[k] * self.impulse_response[k - n]
                var_proc += b * b
            var_proc *= self.innovation_variance

            # extra innovation variance from share randomness
            var_share_innov = 0.0
            for n in range(L_max):
                for k in range(n, L_max):
                    var_share_innov += (self.impulse_response[k - n] * self.impulse_response[k - n]) * Var_c_row[k]
            var_share_innov *= self.innovation_variance

            # share randomness on the mean path
            var_share_mean = 0.0
            for k in range(L_max):
                var_share_mean += m[k] * m[k] * Var_c_row[k]

            out[i] = var_proc + var_share_innov + var_share_mean + var_rounding[i] + self.aggregate_rounding_variance[i]
        return out
