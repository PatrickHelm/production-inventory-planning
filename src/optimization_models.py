from src.supply_chain_config import SupplyChainConfig, SupplyChainConfigJIT
import time
from itertools import product

import gurobipy as gp
from gurobipy import GRB

import numpy as np     
from numba import njit

class DeterministicOptimizationModel:
    def __init__(
            self,
            deterministic_optimization_policy,
            supply_chain_config: SupplyChainConfig
    ):
        self.deterministic_optimization_policy = deterministic_optimization_policy
        self.supply_chain_config = supply_chain_config
        self.logging = self.deterministic_optimization_policy.logging
        self.num_components = len(self.supply_chain_config.lead_times) - len(self.supply_chain_config.p)
        self.num_end_items = len(self.supply_chain_config.p)
        self.components = list(range(self.num_components))
        self.end_items = list(range(self.num_components, self.num_components + self.num_end_items))
        self.items = self.components + self.end_items
        self.planning_horizon = list(range(self.deterministic_optimization_policy.planning_horizon_length))


        self.model: gp.Model = None
        self.production = {}
        self.inventory = {}
        self.backlog = {}

        self.end_item_balance_constraints = {}
        self.component_balance_constraints = {}


    def create_model(self):
        self.model = gp.Model("model")
        if not self.logging:
            self.model.setParam("LogToConsole", 0)

        # Create variables
        for i in self.items:
            for t in self.planning_horizon:
                self.production[i, t] = self.model.addVar(vtype=GRB.INTEGER, lb=0,
                                                          name=f"prod_{i}_{t}")
                self.inventory[i, t] = self.model.addVar(vtype=GRB.INTEGER, lb=0,
                                                         name=f"inv_{i}_{t}")

        for i in self.end_items:
            for t in self.planning_horizon:
                self.backlog[i, t] = self.model.addVar(vtype=GRB.INTEGER,
                                                       name=f"backlog_{i}_{t}")

        self.model.update()

        # Create constraints
        # Balance constraints for end items
        for end_item in self.end_items:
            L = self.supply_chain_config.lead_times[end_item]

            for time_slot in self.planning_horizon:
                prod_window = [
                    self.production[end_item, τ]
                    for τ in self.planning_horizon[0:max(time_slot - L + 1, 0)]
                ]

                ct = self.model.addConstr(
                    gp.quicksum(prod_window)
                    - self.inventory[end_item, time_slot]
                    + self.backlog[end_item, time_slot]
                    == 0,
                    name=f"bal_end_{end_item}_{time_slot}"
                )
                self.end_item_balance_constraints[(end_item, time_slot)] = ct

        # Balance constraints for components
        for component in self.components:
            L = self.supply_chain_config.lead_times[component]

            for time_slot in self.planning_horizon:
                prod_window = [
                    self.production[component, τ]
                    for τ in self.planning_horizon[0:max(time_slot - L + 1, 0)]
                ]

                usage = [
                    self.production[s, τ]
                    for s in self.supply_chain_config.successors[component]
                    for τ in self.planning_horizon[0:time_slot + 1]
                ]

                ct = self.model.addConstr(
                    gp.quicksum(prod_window)
                    - gp.quicksum(usage)
                    - self.inventory[component, time_slot]
                    == 0,
                    name=f"bal_comp_{component}_{time_slot}"
                )
                self.component_balance_constraints[(component, time_slot)] = ct

        # Capacity constraints
        for g, cap in enumerate(self.supply_chain_config.capacities):
            items_in_group = self.supply_chain_config.capacity_groups_indices[g]

            for t in self.planning_horizon:
                self.model.addConstr(
                    gp.quicksum(self.production[i, t] for i in items_in_group) <= cap,
                    name=f"capacity_{g}_{t}"
                )

        # Create objective function
        holding_costs = gp.quicksum(
            self.inventory[i, t] * self.supply_chain_config.h[i]
            for i in self.items
            for t in self.planning_horizon
        )

        backlog_costs = gp.quicksum(
            self.backlog[i, t] * self.supply_chain_config.p[i - self.num_components]
            for i in self.end_items
            for t in self.planning_horizon
        )

        self.model.setObjective(holding_costs + backlog_costs, GRB.MINIMIZE)

        self.model.update()


    def solve_model(self):
        self.model.optimize()

        if self.logging:
            print(f"Objective value: {self.model.objVal}")


    def retrieve_production_quantities(self):
        return [self.production[i, 0].X for i in self.items]

    def update_model(self, state, demand_forecasts):
        # End items
        for (i, t), ct in self.end_item_balance_constraints.items():
            rhs = (
                    sum(demand_forecasts[i - self.num_components][0:t + 1])
                    - sum(state.inventory_in_pipeline_per_node[i][:t])
                    - state.inventory_per_node[i]
            )
            ct.rhs = rhs

        # Components
        for (i, t), ct in self.component_balance_constraints.items():
            rhs = (
                -state.inventory_per_node[i]
                - sum(state.inventory_in_pipeline_per_node[i][:t])
            )
            ct.rhs = rhs

class TwoStageStochasticOptimizationModel:
    def __init__(
            self,
            two_stage_stochastic_optimization_policy,
            supply_chain_config: SupplyChainConfig,
            lp_rounding: bool = False,
    ):
        self.two_stage_stochastic_optimization_policy = two_stage_stochastic_optimization_policy
        self.supply_chain_config = supply_chain_config
        self.supply_chain_config_jit = supply_chain_config.get_jit_class()
        self.lp_rounding = lp_rounding
        self.logging = self.two_stage_stochastic_optimization_policy.logging
        self.num_components = len(self.supply_chain_config.lead_times) - len(self.supply_chain_config.p)
        self.num_end_items = len(self.supply_chain_config.p)
        self.components = list(range(self.num_components))
        self.end_items = list(range(self.num_components, self.num_components + self.num_end_items))
        self.items = self.components + self.end_items
        self.planning_horizon = list(range(self.two_stage_stochastic_optimization_policy.planning_horizon_length))
        self.scenarios = self.two_stage_stochastic_optimization_policy.scenarios
        self.probabilities_scenarios = self.two_stage_stochastic_optimization_policy.probabilities_scenarios

        self.model: gp.Model = None

        # variable containers (dicts keyed by tuples)
        self.production = {}  # (item, t) -> var (first-stage)
        self.inventory = {}  # (item, t, scenario) -> var (second-stage)
        self.backlog = {}  # (end_item, t, scenario) -> var (second-stage)

        # constraint dictionaries to allow RHS updates
        self.end_item_balance_constraints = {}  # (end_item, t, scenario) -> constraint
        self.component_balance_constraints = {}  # (component, t, scenario) -> constraint

        # Contains the previous action for warm starting
        self.prev_solution = {}

    def create_model(self):

        self.model = gp.Model("model")
        self.model.setParam('TimeLimit', 60 * 60)
        self.model.setParam('Method', 1)
        self.model.setParam('NodeMethod', 1)
        # suppress console logging if user disabled logging
        if not self.logging:
            # newer gurobi versions often use LogToConsole / OutputFlag
            try:
                self.model.setParam("LogToConsole", 0)
            except Exception:
                self.model.setParam("OutputFlag", 0)

        vtype = GRB.CONTINUOUS if self.lp_rounding else GRB.INTEGER
        
        # Create variables
        for i in self.items:
            for t in self.planning_horizon:
                self.production[i, t] = self.model.addVar(vtype=vtype, lb=0, name=f"prod_{i}_{t}")

        # second-stage inventory/backlog
        # - components: inventory is scenario-independent
        for i in self.components:
            for t in self.planning_horizon:
                self.inventory[i, t] = self.model.addVar(vtype=vtype, lb=0, name=f"inv_{i}_{t}")

        # - end items: inventory and backlog remain scenario-dependent
        for i in self.end_items:
            for t in self.planning_horizon:
                for s in self.scenarios:
                    self.inventory[i, t, s] = self.model.addVar(vtype=vtype, lb=0, name=f"inv_{i}_{t}_s{s}")

        for i in self.end_items:
            for t in self.planning_horizon:
                for s in self.scenarios:
                    self.backlog[i, t, s] = self.model.addVar(vtype=vtype, lb=0, name=f"back_{i}_{t}_s{s}")

            # push added vars into model
        self.model.update()

        # Create constraints
        # Balance constraints for end items (scenario-dependent)
        for end_item in self.end_items:
            L = self.supply_chain_config.lead_times[end_item]
            for time_slot in self.planning_horizon:
                # production started at or before time_slot - L has arrived at time_slot
                prod_index_slice_end = max(time_slot - L + 1, 0)
                prod_time_indices = self.planning_horizon[0:prod_index_slice_end]
                for scenario in self.scenarios:
                    prod_terms = [self.production[end_item, τ] for τ in prod_time_indices]
                    # sum(empty) -> 0 works naturally
                    ct = self.model.addConstr(
                        gp.quicksum(prod_terms)
                        - self.inventory[end_item, time_slot, scenario]
                        + self.backlog[end_item, time_slot, scenario]
                        == 0,
                        name=f"bal_end_{end_item}_{time_slot}_s{scenario}"
                    )
                    self.end_item_balance_constraints[(end_item, time_slot, scenario)] = ct


        # Balance constraints for components
        for component in self.components:
            L = self.supply_chain_config.lead_times[component]
            for time_slot in self.planning_horizon:
                prod_index_slice_end = max(time_slot - L + 1, 0)
                prod_time_indices = self.planning_horizon[0:prod_index_slice_end]

                prod_terms = [self.production[component, τ] for τ in prod_time_indices]

                usage_terms = []
                # successors could be an empty list
                for successor in self.supply_chain_config.successors[component]:
                    usage_terms.extend([self.production[successor, τ] for τ in self.planning_horizon[0:time_slot + 1]])

                ct = self.model.addConstr(
                    gp.quicksum(prod_terms)
                    - gp.quicksum(usage_terms)
                    - self.inventory[component, time_slot]
                    == 0,
                    name=f"bal_comp_{component}_{time_slot}"
                )
                self.component_balance_constraints[(component, time_slot)] = ct

        # Capacity constraints
        for capacity_group in range(len(self.supply_chain_config.capacities)):
            cap = self.supply_chain_config.capacities[capacity_group]
            items_in_group = self.supply_chain_config.capacity_groups_indices[capacity_group]
            for time_slot in self.planning_horizon:
                self.model.addConstr(
                    gp.quicksum(self.production[item, time_slot] for item in items_in_group) <= cap,
                    name=f"constraint_capacity_{capacity_group}_{time_slot}"
                )

        # Create objective function
        start_time = time.time()

        # holding cost: components are scenario-independent; end items and backlog are scenario-dependent
        comp_inv_cost = gp.quicksum(
            self.inventory[component, t] * self.supply_chain_config.h[component]
            for component in self.components
            for t in self.planning_horizon
        )

        scenario_terms = []
        for s in self.scenarios:
            inv_cost_end = gp.quicksum(
                self.inventory[end_item, t, s] * self.supply_chain_config.h[end_item]
                for end_item in self.end_items
                for t in self.planning_horizon
            )
            back_cost = gp.quicksum(
                self.backlog[end_item, t, s] * self.supply_chain_config.p[end_item - self.num_components]
                for end_item in self.end_items
                for t in self.planning_horizon
            )
            scenario_terms.append(self.probabilities_scenarios[s] * (inv_cost_end + back_cost))

        expected_cost = comp_inv_cost + gp.quicksum(scenario_terms)

        self.model.setObjective(expected_cost, GRB.MINIMIZE)

        # finalize model
        self.model.update()

        if self.logging:
            print("Objective function created")
            print(f"Time elapsed: {(time.time() - start_time):.2f} s")


    def solve_model(self):
        # solve
        self.model.optimize()
        if self.logging:
            # .objVal available after optimize for feasible results
            if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.TIME_LIMIT or self.model.Status == GRB.SUBOPTIMAL:
                print(f"Objective value: {self.model.objVal}")
            else:
                print(f"Model status: {self.model.Status}")


        # Store warm start values of first stage variables (if model is feasible)
        if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self.prev_solution = {}
            # store all production, inventory, backlog
            for key, var in self.production.items():
                self.prev_solution[("production", key)] = var.X

    def retrieve_production_quantities(self, inventory_per_node: np.ndarray):
        # return production decisions for t=0 (first-stage)
        prod0 = []
        for item in self.items:
            var = self.production[item, 0]
            # if not solved or var has no X, .X may raise — but optimize() is expected to be called first
            if var.X is not None:
                prod0.append(var.X)
            else:
                prod0.append(0.0)
                
        if self.lp_rounding:
            prod0 = production_quantity_rounding(np.array(prod0), inventory_per_node, self.supply_chain_config_jit)
        return prod0

    def update_model(self, state, demand_forecasts_scenarios):
        # End item balance constraints (scenario dependent)
        for (end_item, time_slot, scenario), ct in self.end_item_balance_constraints.items():
            rhs = (
                    sum(demand_forecasts_scenarios[scenario][end_item - self.num_components][0:time_slot + 1])
                    - sum(state.inventory_in_pipeline_per_node[end_item][:time_slot])
                    - state.inventory_per_node[end_item]
            )
            # set right-hand side
            ct.rhs = rhs

        # Component balance constraints (scenario dependent)
        for (component, time_slot), ct in self.component_balance_constraints.items():
            rhs = (
                - state.inventory_per_node[component]
                - sum(state.inventory_in_pipeline_per_node[component][:time_slot])
            )
            ct.rhs = rhs

        if self.prev_solution:
            for (item, t), var in self.production.items():
                # previous solution stored values at (item, old_t)
                old_t = t + 1  # shift left: to fill slot t we take previous t+1

                prev_key = ("production", (item, old_t))
                start_val = self.prev_solution.get(prev_key)

                if start_val is not None:
                    # enforce non-negativity; Gurobi accepts Start=0
                    var.Start = max(0.0, float(start_val))
                    
class MultiStageStochasticOptimizationModel:
    def __init__(
            self,
            multi_stage_stochastic_optimization_policy,
            supply_chain_config: SupplyChainConfig,
            lp_rounding: bool = False,
            immediate_integer_only: bool = False,
    ):
        self.multi_stage_stochastic_optimization_policy = multi_stage_stochastic_optimization_policy
        self.supply_chain_config = supply_chain_config
        self.supply_chain_config_jit = supply_chain_config.get_jit_class()
        self.lp_rounding = lp_rounding
        # If True, only production quantities at the immediate (root/time-0)
        # nodes are forced to be integer; all other variables are continuous.
        self.immediate_integer_only = immediate_integer_only
        self.logging = self.multi_stage_stochastic_optimization_policy.logging

        self.num_components = len(self.supply_chain_config.lead_times) - len(self.supply_chain_config.p)
        self.num_end_items = len(self.supply_chain_config.p)
        self.components = list(range(self.num_components))
        self.end_items = list(range(self.num_components, self.num_components + self.num_end_items))
        self.items = self.components + self.end_items

        self.planning_horizon = self.multi_stage_stochastic_optimization_policy.planning_horizon
        self.scenario_tree = self.multi_stage_stochastic_optimization_policy.scenario_tree

        self.ancestor_map = {}
        self.generate_ancestor_map()

        self.model: gp.Model = None
        self.production_quantities = None

        # var containers
        self.production = {}  # (item, tree_node_id)
        self.inventory = {}  # (item, tree_node_id)
        self.backlog = {}  # (end_item, tree_node_id)

        # constraint containers for RHS updates
        self.end_item_balance_constraints = {}
        self.component_balance_constraints = {}

        # precalc relevant ancestors for demand accumulation
        self.relevant_ancestors_demand = {}
        self.generate_relevant_ancestors_demand()

        self.prev_solution = {}

    def generate_ancestor_map(self):
        for tree_node_id in self.scenario_tree:
            ancestors = set()
            current_id = tree_node_id
            while current_id is not None:
                ancestors.add(current_id)
                current_id = self.scenario_tree[current_id]['parent']
            self.ancestor_map[tree_node_id] = ancestors

    def generate_relevant_ancestors_demand(self):
        # for each node and time slot: the node's ancestors (and the node) at or before that time slot
        self.relevant_ancestors_demand = {
            (tree_node_id, time_slot): [
                ancestor_id
                for ancestor_id, ancestor_node in self.scenario_tree.items()
                if ancestor_node['time_slot'] <= time_slot and ancestor_id in self.ancestor_map[tree_node_id]
            ]
            for tree_node_id in self.scenario_tree
            for time_slot in self.planning_horizon
        }


    def create_model(self):
        self.model = gp.Model(name='model')
        # Dual simplex for the LP and for the node relaxations of the MIP
        self.model.setParam('Method', 1)
        self.model.setParam('NodeMethod', 1)
        # turn off console logging if requested
        if not self.logging:
            try:
                self.model.setParam("LogToConsole", 0)
            except Exception:
                self.model.setParam("OutputFlag", 0)
        self.model.setParam('TimeLimit', 60 * 60)
        
        # Default vtype based on lp_rounding
        vtype_default = GRB.CONTINUOUS if self.lp_rounding else GRB.INTEGER

        for item, tree_node_id in product(self.items, list(self.scenario_tree.keys())):
            node = self.scenario_tree[tree_node_id]
            time_slot = node.get('time_slot', 0)

            # Production variable type: integer only for immediate (root/time-0)
            # when `immediate_integer_only` is set; otherwise follow vtype_default.
            if self.immediate_integer_only and time_slot == 0:
                prod_vtype = GRB.INTEGER
            else:
                prod_vtype = vtype_default

            # Inventory variables: if immediate_integer_only we keep inventories
            # continuous (they are not required to be integer), otherwise follow default.
            inv_vtype = GRB.CONTINUOUS if self.immediate_integer_only else vtype_default

            self.production[item, tree_node_id] = self.model.addVar(vtype=prod_vtype, lb=0,
                                                                    name=f"prod_{item}_n{tree_node_id}")
            self.inventory[item, tree_node_id] = self.model.addVar(vtype=inv_vtype, lb=0,
                                                                   name=f"inv_{item}_n{tree_node_id}")

        for end_item, tree_node_id in product(self.end_items, list(self.scenario_tree.keys())):
            # backlog variables are continuous when immediate_integer_only is True,
            # otherwise follow the default vtype
            backlog_vtype = GRB.CONTINUOUS if self.immediate_integer_only else vtype_default
            self.backlog[end_item, tree_node_id] = self.model.addVar(vtype=backlog_vtype, lb=0,
                                                                     name=f"back_{end_item}_n{tree_node_id}")

        self.model.update()

        # Create constraints
        current_time = time.time()

        for tree_node_id, tree_node in self.scenario_tree.items():
            time_slot = tree_node['time_slot']

            # End item balance constraints
            for end_item in self.end_items:
                lead_time = self.supply_chain_config.lead_times[end_item]

                # production_sum: production at ancestor nodes that occurred at times <= time_slot - lead_time
                prod_terms = [
                    self.production[end_item, ancestor_id]
                    for ancestor_id, ancestor_node in self.scenario_tree.items()
                    if ancestor_node['time_slot'] <= time_slot - lead_time
                       and ancestor_id in self.ancestor_map[tree_node_id]
                ]

                inventory = self.inventory[end_item, tree_node_id]
                backlog = self.backlog[end_item, tree_node_id]

                ct = self.model.addConstr(
                    gp.quicksum(prod_terms) - inventory + backlog == 0,
                    name=f"bal_end_{end_item}_n{tree_node_id}"
                )
                self.end_item_balance_constraints[(end_item, tree_node_id)] = ct

            # Component balance constraints
            for component in self.components:
                lead_time = self.supply_chain_config.lead_times[component]
                prod_terms = [
                    self.production[component, ancestor_id]
                    for ancestor_id, ancestor_node in self.scenario_tree.items()
                    if ancestor_node['time_slot'] <= time_slot - lead_time
                       and ancestor_id in self.ancestor_map[tree_node_id]
                ]

                usage_terms = []
                for successor in self.supply_chain_config.successors[component]:
                    usage_terms.extend([
                        self.production[successor, ancestor_id]
                        for ancestor_id, ancestor_node in self.scenario_tree.items()
                        if ancestor_node['time_slot'] <= time_slot
                           and ancestor_id in self.ancestor_map[tree_node_id]
                    ])

                inventory = self.inventory[component, tree_node_id]

                ct = self.model.addConstr(
                    gp.quicksum(prod_terms) - gp.quicksum(usage_terms) - inventory == 0,
                    name=f"bal_comp_{component}_n{tree_node_id}"
                )
                self.component_balance_constraints[(component, tree_node_id)] = ct

            # Capacity constraints (per tree node)
            for capacity_group in range(len(self.supply_chain_config.capacities)):
                items_in_group = self.supply_chain_config.capacity_groups_indices[capacity_group]
                cap = self.supply_chain_config.capacities[capacity_group]

                self.model.addConstr(
                    gp.quicksum(self.production[item, tree_node_id] for item in items_in_group) <= cap,
                    name=f"constraint_capacity_{capacity_group}_n{tree_node_id}"
                )

        if self.logging:
            print('Constraints created')
            print(f'Time elapsed: {(time.time() - current_time):.2f} s')

        # Non-anticipativity constraints
        current_time = time.time()

        # time 0 group: all nodes at time 0 must have equal production (if multiple)
        group_time0 = [nid for nid, node in self.scenario_tree.items() if node['time_slot'] == 0]
        if len(group_time0) > 1:
            base_node = group_time0[0]
            for item in self.items:
                for other_node in group_time0[1:]:
                    self.model.addConstr(
                        self.production[item, base_node] == self.production[item, other_node],
                        name=f"constraint_non_anticipativity_{item}_n{base_node}_n{other_node}"
                    )

        # For t > 0, group nodes by shared parent
        for t in self.planning_horizon:
            if t == 0:
                continue
            # parents of nodes at time t (excluding None)
            parent_ids = set(
                node['parent'] for nid, node in self.scenario_tree.items()
                if node['time_slot'] == t and node['parent'] is not None
            )
            for parent_id in parent_ids:
                group = [
                    nid for nid, node in self.scenario_tree.items()
                    if node['time_slot'] == t and node['parent'] == parent_id
                ]
                if len(group) > 1:
                    base_node = group[0]
                    for item in self.items:
                        for other_node in group[1:]:
                            self.model.addConstr(
                                self.production[item, base_node] == self.production[item, other_node],
                                name=f"constraint_non_anticipativity_{item}_n{base_node}_n{other_node}"
                            )

        if self.logging:
            print('Non-anticipativity constraints created')
            print(f'Time elapsed: {(time.time() - current_time):.2f} s')

        # Objective: expected holding + backlog cost over scenario tree
        current_time = time.time()

        obj_terms = []
        for tree_node_id, tree_node in self.scenario_tree.items():
            prob = tree_node.get('probability', 1.0)
            inv_cost = gp.quicksum(
                self.inventory[item, tree_node_id] * self.supply_chain_config.h[item]
                for item in self.items
            )
            back_cost = gp.quicksum(
                self.backlog[end_item, tree_node_id] * self.supply_chain_config.p[end_item - self.num_components]
                for end_item in self.end_items
            )
            obj_terms.append(prob * (inv_cost + back_cost))

        self.model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

        if self.logging:
            print('Objective function created')
            print(f'Time elapsed: {(time.time() - current_time):.2f} s')

        self.model.update()


    def solve_model(self):
        self.model.optimize()
        if self.logging:
            if self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                print(f'Objective value: {self.model.objVal}')
            else:
                print(f'Model status: {self.model.Status}')

        # Save previous solution after solve (for warm start)
        if self.model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            self.prev_solution = {}
            for (item, tree_node_id), var in self.production.items():
                if var.X is not None:
                    self.prev_solution[("production", (item, tree_node_id))] = float(var.X)

    def retrieve_production_quantities(self, inventory_per_node: np.ndarray):
        # the decision to implement is the production at the root node (id 0)
        root_node_id = 0
        prod0 = []
        for item in self.items:
            var = self.production.get((item, root_node_id))
            if var is None:
                # fallback: try any node at time 0
                nodes_time0 = [nid for nid, n in self.scenario_tree.items() if n['time_slot'] == 0]
                if len(nodes_time0) > 0:
                    var = self.production[item, nodes_time0[0]]
                else:
                    prod0.append(0.0)
                    continue
            if var.X is None:
                print(f"Warning: production variable for item {item} at root node has no solution value.")
            prod0.append(var.X if var.X is not None else 0.0)
            
        if self.lp_rounding and not self.immediate_integer_only:
            prod0 = production_quantity_rounding(np.array(prod0), inventory_per_node, self.supply_chain_config_jit)
        return prod0

    def update_model(self, state, scenario_tree_demand):
        # update end item balance RHS
        for (end_item, tree_node_id), ct in self.end_item_balance_constraints.items():
            tree_node = scenario_tree_demand[tree_node_id]
            time_slot = tree_node['time_slot']
            initial_inventory = state.inventory_per_node[end_item]
            pipeline_inventory = sum(state.inventory_in_pipeline_per_node[end_item][:time_slot])

            demand_sum = sum(
                scenario_tree_demand[ancestor_id]['demand'][end_item - self.num_components]
                for ancestor_id in self.relevant_ancestors_demand[(tree_node_id, time_slot)]
            )

            ct.rhs = demand_sum - pipeline_inventory - initial_inventory

        # update component balance RHS
        for (component, tree_node_id), ct in self.component_balance_constraints.items():
            tree_node = scenario_tree_demand[tree_node_id]
            time_slot = tree_node['time_slot']
            initial_inventory = state.inventory_per_node[component]
            pipeline_inventory = sum(state.inventory_in_pipeline_per_node[component][:time_slot])
            ct.rhs = -initial_inventory - pipeline_inventory

        # Apply warm start for production variables using previous solution
        if self.prev_solution:
            for (item, tree_node_id), var in self.production.items():
                start_key = ("production", (item, tree_node_id))
                start_val = self.prev_solution.get(start_key)
                if start_val is not None:
                    var.Start = max(0.0, float(start_val))

@njit           
def production_quantity_rounding(production_quantities: np.ndarray, inventory_per_node: np.ndarray, supply_chain_config: SupplyChainConfigJIT):
    production_floor = np.floor(np.around(production_quantities, 2))
    fractional_parts = production_quantities - production_floor
    total_to_allocate = int(round(production_quantities.sum() - production_floor.sum()))
    if total_to_allocate <= 0:
        return production_floor.astype(np.int64)
    
    # Precompute used capacity per capacity group based on floor solution
    num_capacity_groups = supply_chain_config.capacities.shape[0]
    used_capacity = np.zeros(num_capacity_groups, dtype=np.int64)
    for i in range(supply_chain_config.num_components):
        g = supply_chain_config.capacity_groups[i]
        used_capacity[g] += production_floor[i]
        
    # ---- Precompute material usage:
    # material_usage[j] = sum_{k in successors(j)} production_floor[k] ----
    material_usage = np.zeros(supply_chain_config.num_components, dtype=np.int64)
    for j in range(supply_chain_config.num_components):
        total = 0
        num_succ = supply_chain_config.n_successors[j]
        for s in range(num_succ):
            k = supply_chain_config.successors[j, s]
            total += production_floor[k]
        material_usage[j] = total
    
    allocated = 0
    indices_sorted = np.argsort(-fractional_parts)
    for t in range(indices_sorted.shape[0]):
        if allocated >= total_to_allocate:
            break
        idx = indices_sorted[t]
        # Try to add 1 at idx if feasible
        if _can_add_one(idx, used_capacity, material_usage, inventory_per_node, supply_chain_config):
            # Update production
            production_floor[idx] += 1
            # Update capacity usage
            g = supply_chain_config.capacity_groups[idx]
            used_capacity[g] += 1
            # Update material usage for all predecessors of idx
            num_preds = supply_chain_config.n_predecessors[idx]
            for p in range(num_preds):
                j = supply_chain_config.predecessors[idx, p]
                material_usage[j] += 1
            # Update allocated count
            allocated += 1
    return production_floor.astype(np.int64)

@njit
def _can_add_one(
    node_idx: int,
    used_capacity: np.ndarray,
    material_usage: np.ndarray,
    inventory_per_node: np.ndarray,
    supply_chain_config: SupplyChainConfigJIT,
) -> bool:
    """
    Check whether increasing production at `node_idx` by 1 keeps
    material and capacity constraints feasible, given precomputed
    used_capacity and material_usage arrays.
    """

    # ---- Capacity check ----
    group = supply_chain_config.capacity_groups[node_idx]
    if used_capacity[group] + 1 > supply_chain_config.capacities[group]:
        return False

    # ---- Material check: for each predecessor j of node_idx,
    #      ensure inventory_per_node[j] >= material_usage[j] + 1 ----
    num_preds = supply_chain_config.n_predecessors[node_idx]
    for p in range(num_preds):
        j = supply_chain_config.predecessors[node_idx, p]
        # adding 1 unit at node_idx consumes 1 more unit of material j
        if inventory_per_node[j] < material_usage[j] + 1:
            return False

    return True