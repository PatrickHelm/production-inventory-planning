import random
import math
import numpy as np
from src.optimization_models import (DeterministicOptimizationModel, 
                                     TwoStageStochasticOptimizationModel,
                                     MultiStageStochasticOptimizationModel)
from src.evaluation_utils import precompute_demand_realizations

# Deterministic optimization policy
# This policy runs a deterministic MILP with a timeseries point forecast including hedging
class DeterministicOptimizationPolicy:
    def __init__(self, supply_chain_config, hedging_parameter, planning_horizon_length=None):
        self.supply_chain_config = supply_chain_config
        self.supply_chain_config_jit = supply_chain_config.get_jit_class()
        self.hedging_parameter = hedging_parameter
        if planning_horizon_length is None:
            self.planning_horizon_length = 30
        else:
            self.planning_horizon_length = planning_horizon_length
        self.logging = False
        self.math_model = DeterministicOptimizationModel(self, supply_chain_config=self.supply_chain_config)
        self.math_model.create_model()

    def set_action(self, state):
        demand_forecasts = self.generate_demand_forecasts(state=state)

        self.math_model.update_model(state=state, demand_forecasts=demand_forecasts)
        self.math_model.solve_model()
        action_per_node = self.math_model.retrieve_production_quantities()
        action_per_node = np.around(action_per_node)

        return action_per_node

    def generate_demand_forecasts(self, state):
        # Get demand vector for AR model, i.e., most recent observations are first elements
        aggregate_demand_history_ar_model = state.aggregate_demand_model_history[
                                            -self.supply_chain_config.aggregate_demand_num_lags:]

        # Generate demand forecasts over the planning horizon based on AR model
        demand_forecasts = [[] for _ in range(self.supply_chain_config.num_products)]

        for forecast_period in range(self.planning_horizon_length):
            next_aggregate_demand = (np.sum(self.supply_chain_config.aggregate_demand_ar_params *
                                           (aggregate_demand_history_ar_model -
                                            self.supply_chain_config.aggregate_demand_mean)) +
                                     self.supply_chain_config.aggregate_demand_mean)
            next_aggregate_demand = max(next_aggregate_demand, 0)
            forecast_aggregate_demand = next_aggregate_demand * (1 + self.hedging_parameter)

            aggregate_demand_history_ar_model = np.insert(aggregate_demand_history_ar_model[:-1], 0,
                                                          next_aggregate_demand)

            fractional_part = forecast_aggregate_demand - math.floor(forecast_aggregate_demand)
            if random.random() < fractional_part:
                aggregate_demand_integer = math.ceil(forecast_aggregate_demand)
            else:
                aggregate_demand_integer = math.floor(forecast_aggregate_demand)

            # Generate the stochastic split parameters
            samples = list(range(self.supply_chain_config.num_products))
            for i in range(self.supply_chain_config.num_products):
                samples[i] = self.supply_chain_config.product_demand_split_kappa[i] * self.supply_chain_config.product_demand_split_theta[i]

            # Compute proportions of total demand
            proportion = np.array([samples[i] / sum(samples) for i in range(self.supply_chain_config.num_products)])

            # Compute the demand per product
            # Deterministic floor allocation
            demand_decimals = proportion * aggregate_demand_integer
            demand = np.floor(demand_decimals).astype(int)

            # Remainder of demand
            R = aggregate_demand_integer - demand.sum()

            # If there are remainders, allocate them randomly based on the fractional parts
            if R != 0:
                # Randomly allocate the remainder based on the fractional parts
                fractional_parts = demand_decimals - demand
                if fractional_parts.sum() == 0:
                    # all fractional parts zero, randomly assign leftovers equally
                    fractional_parts = np.ones_like(fractional_parts)
                fractional_parts = fractional_parts / fractional_parts.sum()

                # Stochastically distribute the leftovers
                leftovers = np.random.choice(len(fractional_parts), size=R, p=fractional_parts)
                # Increment counts for chosen indices
                for i in leftovers:
                    demand[i] += 1

            for end_item in range(self.supply_chain_config.num_products):
                demand_forecasts[end_item].append(demand[end_item])


        return demand_forecasts


# Two-stage stochastic optimization policy
# This policy runs a two-stage stochastic program over demand scenarios, as a MILP or as its LP relaxation with rounding
class TwoStageStochasticOptimizationPolicy:
    def __init__(self, supply_chain_config, num_scenarios, seed, lp_rounding=False, planning_horizon_length=None):
        self.supply_chain_config = supply_chain_config
        self.supply_chain_config_jit = supply_chain_config.get_jit_class()
        self.num_scenarios = num_scenarios
        self.seed = seed
        self.scenarios = list(range(num_scenarios))
        self.probabilities_scenarios = self.num_scenarios * [1 / self.num_scenarios]
        if planning_horizon_length is None:
            self.planning_horizon_length = 30
        else:
            self.planning_horizon_length = planning_horizon_length
        self.logging = False
        self.math_model = TwoStageStochasticOptimizationModel(self, supply_chain_config=self.supply_chain_config, lp_rounding=lp_rounding)
        self.math_model.create_model()

    def set_action(self, state):
        demand_forecasts_scenarios = precompute_demand_realizations(
            self.supply_chain_config_jit,
            state.aggregate_demand_model_history,
            self.num_scenarios,
            self.planning_horizon_length,
            self.seed
        )
        self.seed += 1
        demand_forecasts_scenarios = demand_forecasts_scenarios[:, :, :-1].transpose(0, 2, 1)

        self.math_model.update_model(state=state, demand_forecasts_scenarios=demand_forecasts_scenarios)
        self.math_model.solve_model()
        action_per_node = self.math_model.retrieve_production_quantities(state.inventory_per_node)
        action_per_node = (np.around(action_per_node)).astype(int)

        return action_per_node

# Multi-stage stochastic optimization policy
# This policy runs a multi-stage stochastic program over a demand scenario tree, as a MILP or as its LP relaxation with rounding
class MultiStageStochasticOptimizationPolicy:
    def __init__(self, supply_chain_config, branching_structure, seed, lp_rounding=False):
        self.supply_chain_config = supply_chain_config
        self.branching_structure = branching_structure
        self.planning_horizon_length = len(branching_structure)
        self.planning_horizon = list(range(self.planning_horizon_length))
        self.scenario_tree = self.generate_scenario_tree()
        self.logging = False
        self.seed = seed
        self.math_model = MultiStageStochasticOptimizationModel(self, supply_chain_config=self.supply_chain_config, lp_rounding=lp_rounding, immediate_integer_only=False)
        self.math_model.create_model()

    def set_action(self, state):
        scenario_tree_demand = self.generate_scenario_tree_demand(state=state, seed=self.seed)
        self.seed += 1
        self.math_model.update_model(state=state, scenario_tree_demand=scenario_tree_demand)
        self.math_model.solve_model()
        action_per_node = self.math_model.retrieve_production_quantities(state.inventory_per_node)
        action_per_node = np.around(action_per_node).astype(np.int64)
        return action_per_node

    def generate_demand(self, aggregate_demand_history_ar_model):
        aggregate_demand = ((np.sum(self.supply_chain_config.aggregate_demand_ar_params *
                                    (aggregate_demand_history_ar_model -
                                     self.supply_chain_config.aggregate_demand_mean)) +
                             np.random.normal(loc=0, scale=self.supply_chain_config.aggregate_demand_error_sigma)) +
                            self.supply_chain_config.aggregate_demand_mean)

        # Censor at zero
        aggregate_demand = max(aggregate_demand, 0)

        # Stochastically round the aggregated demand based on its fractional part
        fractional_part = aggregate_demand - math.floor(aggregate_demand)
        if np.random.random() < fractional_part:
            aggregate_demand_integer = math.ceil(aggregate_demand)
        else:
            aggregate_demand_integer = math.floor(aggregate_demand)

        # Generate the stochastic split parameters
        samples = list(range(self.supply_chain_config.num_products))
        for i in range(self.supply_chain_config.num_products):
            samples[i] = np.random.gamma(shape=self.supply_chain_config.product_demand_split_kappa[i],
                                         scale=self.supply_chain_config.product_demand_split_theta[i])

        # Compute proportions of total demand
        proportion = np.array(
            [samples[i] / sum(samples) for i in range(self.supply_chain_config.num_products)])

        # Compute the demand per product
        # Deterministic floor allocation
        demand_decimals = proportion * aggregate_demand_integer
        demand = np.floor(demand_decimals).astype(int)

        # Remainder of demand
        R = aggregate_demand_integer - demand.sum()

        # If there are remainders, allocate them randomly based on the fractional parts
        if R != 0:
            # Randomly allocate the remainder based on the fractional parts
            fractional_parts = demand_decimals - demand
            if fractional_parts.sum() == 0:
                # all fractional parts zero, randomly assign leftovers equally
                fractional_parts = np.ones_like(fractional_parts)
            fractional_parts = fractional_parts / fractional_parts.sum()

            # Stochastically distribute the leftovers
            leftovers = np.random.choice(len(fractional_parts), size=R, p=fractional_parts)
            # Increment counts for chosen indices
            for i in leftovers:
                demand[i] += 1

        demand = np.append(demand, aggregate_demand)

        return demand

    def generate_scenario_tree(self):
        scenario_tree = {}
        tree_node_id = 0

        # Branching structure
        current_level = [0]
        for time_slot in self.planning_horizon:
            next_level = []
            if time_slot == 0:
                for b in range(self.branching_structure[time_slot]):
                    prob = 1 / self.branching_structure[time_slot]
                    scenario_tree[tree_node_id] = {}
                    scenario_tree[tree_node_id]['parent'] = None
                    scenario_tree[tree_node_id]['time_slot'] = time_slot
                    scenario_tree[tree_node_id]['probability'] = prob
                    next_level.append(tree_node_id)
                    tree_node_id += 1
            else:
                for parent_id in current_level:
                    for b in range(self.branching_structure[time_slot]):
                        prob = scenario_tree[parent_id]['probability'] / self.branching_structure[time_slot]
                        scenario_tree[tree_node_id] = {}
                        scenario_tree[tree_node_id]['parent'] = parent_id
                        scenario_tree[tree_node_id]['time_slot'] = time_slot
                        scenario_tree[tree_node_id]['probability'] = prob
                        next_level.append(tree_node_id)
                        tree_node_id += 1

            current_level = next_level

        return scenario_tree

    def generate_scenario_tree_demand(self, state, seed):
        scenario_tree_demand = {}
        tree_node_id = 0
        np.random.seed(seed)
        # Branching structure
        current_level = [0]
        for time_slot in self.planning_horizon:
            next_level = []
            if time_slot == 0:
                for b in range(self.branching_structure[time_slot]):
                    prob = 1 / self.branching_structure[time_slot]

                    aggregate_demand_history_ar_model = state.aggregate_demand_model_history[
                                                        -self.supply_chain_config.aggregate_demand_num_lags:]
                    demand = self.generate_demand(aggregate_demand_history_ar_model=aggregate_demand_history_ar_model)
                    new_history = np.insert(aggregate_demand_history_ar_model[:-1], 0,
                                            demand[self.supply_chain_config.num_products])
                    
                    scenario_tree_demand[tree_node_id] = {}
                    scenario_tree_demand[tree_node_id]['parent'] = None
                    scenario_tree_demand[tree_node_id]['time_slot'] = time_slot
                    scenario_tree_demand[tree_node_id]['probability'] = prob
                    scenario_tree_demand[tree_node_id]['demand'] = demand[0:self.supply_chain_config.num_products]
                    scenario_tree_demand[tree_node_id]['aggregate_demand_history'] = new_history
                    next_level.append(tree_node_id)
                    tree_node_id += 1
            else:
                for parent_id in current_level:
                    for b in range(self.branching_structure[time_slot]):
                        prob = scenario_tree_demand[parent_id]['probability'] / self.branching_structure[time_slot]

                        aggregate_demand_history_ar_model = scenario_tree_demand[parent_id]['aggregate_demand_history']
                        demand = self.generate_demand(
                            aggregate_demand_history_ar_model=aggregate_demand_history_ar_model)
                        new_history = np.insert(aggregate_demand_history_ar_model[:-1], 0,
                                                demand[self.supply_chain_config.num_products])

                        scenario_tree_demand[tree_node_id] = {}
                        scenario_tree_demand[tree_node_id]['parent'] = parent_id
                        scenario_tree_demand[tree_node_id]['time_slot'] = time_slot
                        scenario_tree_demand[tree_node_id]['probability'] = prob
                        scenario_tree_demand[tree_node_id]['demand'] = demand[0:self.supply_chain_config.num_products]
                        scenario_tree_demand[tree_node_id]['aggregate_demand_history'] = new_history
                        next_level.append(tree_node_id)
                        tree_node_id += 1

            current_level = next_level

        return scenario_tree_demand
