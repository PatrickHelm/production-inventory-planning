import numpy as np
import pandas as pd
from numba import float64, int64
from numba.experimental import jitclass


class SupplyChainConfig:
    def __init__(
        self,
        scaling: int = 1,
        stationary_demand: bool = False,
        higher_backorder_costs: bool = False,
        lower_capacities: bool = False,
    ):
        self.stationary_demand = stationary_demand
        self.higher_backorder_costs = higher_backorder_costs
        self.lower_capacities = lower_capacities
        self.static_demand_suffix = "_stationary_demand" if stationary_demand else ""
        self.test_demand_suffix = "_stationary_demand" if stationary_demand else ""
        self.ablation_suffix = "".join(
            suffix
            for enabled, suffix in (
                (stationary_demand, "_stationary_demand"),
                (higher_backorder_costs, "_higher_backorder_costs"),
                (lower_capacities, "_lower_capacities"),
            )
            if enabled
        )
        path_data = 'data/supply_chain_configuration.xlsx' # Path to the data file
        self.bill_of_materials = pd.read_excel(path_data, sheet_name='BillOfMaterials')
        component_names = self.bill_of_materials.iloc[:, 0].to_list()
        self.successors, self.predecessors = self.get_bill_of_materials()
        self.lead_times = pd.read_excel(path_data,
                                        sheet_name='LeadTimes').iloc[:, 1].to_list()
        # Capacities per capacity group
        self.capacities = pd.read_excel(path_data,
                                        sheet_name='Capacities', index_col=None).iloc[:, 1].to_list()
        if lower_capacities:
            self.capacities = [capacity - 1 for capacity in self.capacities]
        df_capacities_groups = pd.read_excel(path_data,
                                             sheet_name='Capacities', index_col=None).iloc[:, 2:]
        # Define the capacity groups
        self.capacity_groups_indices = self.get_capacity_groups(df_capacities_groups, component_names)

        self.num_components = len(self.lead_times)
        self.num_sub_components = sum(1 for successors in self.successors if successors)
        self.num_products = self.num_components - self.num_sub_components
        self.product_indices = [self.num_components + i for i in range(self.num_products)]
        self.num_nodes = len(self.capacity_groups_indices)

        # Find the capacity group for each component/product
        self.capacity_groups = self.find_capacity_groups()
        self.h = pd.read_excel(path_data,
                               sheet_name='HoldingCosts').iloc[:, 1].to_list()
        self.p = pd.read_excel(path_data,
                               sheet_name='BacklogPenalties').iloc[:self.num_products, 1].to_list()
        if higher_backorder_costs:
            # set p = 0.95 / (1-0.95) * h for each product (95% service level)
            self.p = [
                19 * self.h[self.num_sub_components + i]
                for i in range(self.num_products)
            ]

        # Aggregate demand model parameters
        self.aggregate_demand_ar_params = pd.read_excel(path_data,
                                                        sheet_name='AggregateDemandModel',
                                                        header=None).iloc[0, 1:].to_list()
        self.aggregate_demand_num_lags = len(self.aggregate_demand_ar_params)
        self.aggregate_demand_error_sigma = pd.read_excel(path_data,
                                                          sheet_name='AggregateDemandModel',
                                                          header=None).iloc[1, 1]
        if stationary_demand:
            self.aggregate_demand_ar_params = [
                0.0 for _ in self.aggregate_demand_ar_params
            ]
            self.aggregate_demand_error_sigma = np.sqrt(13.5)

        # Mean demand per product and split parameters
        self.product_demand_mean = pd.read_excel(path_data,
                                                 sheet_name='ProductDemandMean').iloc[:self.num_products, 1].to_list()
        self.scaling = scaling
        self.product_demand_split_kappa = (pd.read_excel(
            path_data, sheet_name='ProductDemandMean').iloc[:self.num_products, 2] * self.scaling).to_list()
        self.product_demand_split_theta = (pd.read_excel(
            path_data, sheet_name='ProductDemandMean').iloc[:self.num_products, 3] / self.scaling).to_list()

        self.aggregate_demand_mean = sum(self.product_demand_mean)

        self.avg_demand_per_product = self.product_demand_mean

        # Average demand per component for initial state
        self.avg_demand_per_comp = self.get_avg_demand()

        # Get the added values for each component
        self.added_value = self.get_added_values()

        self.echelon_lead_times_component_per_product = self.get_echelon_lead_time_per_product()

    def get_bill_of_materials(self):
        # Vector of vectors for rows
        successors = []
        for index, row in self.bill_of_materials.iterrows():
            successor = [self.bill_of_materials.columns.get_loc(col) - 1
                         for col in self.bill_of_materials.columns if row[col] == 1]
            successors.append(successor)

        # Vector of vectors for columns
        predecessors = []
        for col in self.bill_of_materials.columns:
            # skip the first column, which holds the component names
            if col == 'Unnamed: 0':
                continue
            predecessor = [index for index, row in self.bill_of_materials.iterrows() if row[col] == 1]
            predecessors.append(predecessor)

        return successors, predecessors

    def get_capacity_groups(self, df_capacities_groups, component_names):
        capacity_groups = [[] for _ in range(len(df_capacities_groups))]
        for i in range(len(df_capacities_groups)):
            for j in range(len(df_capacities_groups.iloc[i])):
                if pd.isna(df_capacities_groups.iloc[i, j]):
                    continue
                else:
                    capacity_groups[i].append(component_names.index(df_capacities_groups.iloc[i, j]))

        return capacity_groups

    def find_capacity_groups(self):
        find_capacity_group = []
        for i in range(self.num_components):
            find_capacity_group.append([j for j in range(self.num_nodes) if i in self.capacity_groups_indices[j]][0])

        return find_capacity_group

    def get_avg_demand(self):
        avg_demand_per_product = self.product_demand_mean

        avg_demand_per_component = [0 for _ in range(self.num_sub_components)]
        for i in range(self.num_sub_components - 1, -1, -1):
            for suc in self.successors[i]:
                if suc > self.num_sub_components - 1:
                    avg_demand_per_component[i] += avg_demand_per_product[suc - self.num_sub_components]
                else:
                    avg_demand_per_component[i] += avg_demand_per_component[suc]

        avg_demand = avg_demand_per_component + avg_demand_per_product

        return avg_demand

    def get_echelon_lead_time_per_product(self):
        # compute echelon lead times per product
        echelon_lead_times_component_per_product = [[0 for _ in range(self.num_components)] for _ in
                                                    range(self.num_products)]
        for i in range(self.num_products):
            predecessors_nodes = []
            echelon_lead_times_component_per_product[i][self.num_sub_components + i] = (
                self.lead_times)[self.num_sub_components + i]
            for j in self.predecessors[self.num_sub_components + i]:
                predecessors_nodes.append(j)
                echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][
                    self.num_sub_components + i]

            # For suppliers
            # Here we assume that the supply chain has no more than 3 tiers
            pred_of_pred_nodes = []
            for pred in predecessors_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.lead_times[pred]
                for j in self.predecessors[pred]:
                    pred_of_pred_nodes.append(j)
                    echelon_lead_times_component_per_product[i][j] = echelon_lead_times_component_per_product[i][pred]

            for pred in pred_of_pred_nodes:
                echelon_lead_times_component_per_product[i][pred] += self.lead_times[pred]

        # Add one period to echelon lead time due to periodic review
        for i in range(self.num_products):
            for j in range(self.num_components):
                if echelon_lead_times_component_per_product[i][j] > 0:
                    echelon_lead_times_component_per_product[i][j] += 1

        return echelon_lead_times_component_per_product

    def get_added_values(self):
        added_value = self.h.copy()
        for i in range(self.num_components):
            for j in range(len(self.predecessors[i])):
                added_value[i] -= self.h[self.predecessors[i][j]]
        return added_value

    def get_jit_class(self, load_precomputed: bool = True):
        # Aggregate demand history (first position is the most recent observation)
        aggregate_demand_model_history = np.loadtxt("data/aggregate_demand_model_history.txt")

        expected_product_shares = self.product_demand_split_kappa / np.sum(self.product_demand_split_kappa)
        if load_precomputed:
            variance_product_shares = np.loadtxt(
                f"data/precomputed/variance_product_shares_echelon_lead_time_scaling{self.scaling}.txt"
            )
            echelon_lead_time_demand_std = np.loadtxt(
                f"data/precomputed/static_echelon_lead_time_demand_std_scaling{self.scaling}{self.static_demand_suffix}.txt"
            )
            allocation_rounding_variance = np.loadtxt(
                "data/precomputed/allocation_rounding_variance.txt"
            )
        else:
            # Bootstrap path for the scripts that generate these files.
            max_echelon_lead_time = int(
                np.max(np.asarray(self.echelon_lead_times_component_per_product))
            )
            variance_product_shares = np.zeros(
                (self.num_components, max_echelon_lead_time), dtype=np.float64
            )
            echelon_lead_time_demand_std = np.zeros(
                self.num_components, dtype=np.float64
            )
            allocation_rounding_variance = np.zeros(
                self.num_components, dtype=np.float64
            )
        
        predecessor_matrix = np.zeros(
            (self.num_components, self.num_components), dtype=np.int64
        )
        for i in range(self.num_components):
            for pre in self.predecessors[i]:
                predecessor_matrix[i, pre] = 1

        successor_matrix = np.zeros(
            (self.num_components, self.num_components), dtype=np.int64
        )
        for i in range(self.num_components):
            for suc in self.successors[i]:
                successor_matrix[i, suc] = 1

        capacity_group_matrix = np.zeros(
            (self.num_nodes, self.num_components), dtype=np.int64
        )
        for i in range(self.num_nodes):
            for comp in self.capacity_groups_indices[i]:
                capacity_group_matrix[i, comp] = 1

        capacity_groups = np.argmax(capacity_group_matrix, axis=0)
        capacity_groups_size = capacity_group_matrix.sum(axis=1)
        capacity_groups_indices = np.full(
            (self.num_nodes, np.max(capacity_groups_size)), -1, np.int64
        )
        for i in range(self.num_nodes):
            row = capacity_group_matrix[i]
            f = 0
            for j in range(self.num_components):
                if row[j] == 1:
                    capacity_groups_indices[i, f] = j
                    f += 1

        n_predecessors = np.sum(predecessor_matrix, axis=1)
        predecessors = np.full(
            (self.num_components, np.max(n_predecessors)), -1, np.int64
        )
        for i in range(self.num_components):
            row = predecessor_matrix[i]
            f = 0
            for j in range(self.num_components):
                if row[j] == 1:
                    predecessors[i, f] = j
                    f += 1

        n_successors = np.sum(successor_matrix, axis=1)
        successors = np.full((self.num_components, np.max(n_successors)), -1, np.int64)
        for i in range(self.num_components):
            row = successor_matrix[i]
            f = 0
            for j in range(self.num_components):
                if row[j] == 1:
                    successors[i, f] = j
                    f += 1

        return SupplyChainConfigJIT(
            num_components=self.num_components,
            n_predecessors=n_predecessors,
            n_successors=n_successors,
            predecessors=predecessors,
            successors=successors,
            capacity_groups=capacity_groups,
            capacity_groups_size=capacity_groups_size,
            capacity_groups_indices=capacity_groups_indices,
            num_sub_components=self.num_sub_components,
            num_products=self.num_products,
            num_nodes=self.num_nodes,
            lead_times=np.array(self.lead_times, dtype=np.int64),
            capacities=np.array(self.capacities, dtype=np.int64),
            h=np.array(self.h, dtype=np.float64),
            p=np.array(self.p, dtype=np.float64),
            aggregate_demand_ar_params=np.array(
                self.aggregate_demand_ar_params, dtype=np.float64
            ),
            aggregate_demand_num_lags=self.aggregate_demand_num_lags,
            aggregate_demand_error_sigma=self.aggregate_demand_error_sigma,
            product_demand_mean=np.array(self.product_demand_mean, dtype=np.float64),
            product_demand_split_kappa=np.array(
                self.product_demand_split_kappa, dtype=np.float64
            ),
            product_demand_split_theta=np.array(
                self.product_demand_split_theta, dtype=np.float64
            ),
            aggregate_demand_mean=np.array(
                self.aggregate_demand_mean, dtype=np.float64
            ),
            avg_demand_per_product=np.array(
                self.avg_demand_per_product, dtype=np.float64
            ),
            avg_demand_per_comp=np.array(self.avg_demand_per_comp, dtype=np.float64),
            aggregate_demand_model_history=np.array(
                aggregate_demand_model_history, dtype=np.float64
            ),
            expected_product_shares=expected_product_shares,
            variance_product_shares=variance_product_shares,
            echelon_lead_time_demand_std=echelon_lead_time_demand_std,
            allocation_rounding_variance=allocation_rounding_variance,
        )


cfg_spec = [
    ("n_predecessors", int64[:]),
    ("n_successors", int64[:]),
    ("predecessors", int64[:, :]),
    ("successors", int64[:, :]),
    ("capacity_groups", int64[:]),
    ("capacity_groups_size", int64[:]),
    ("capacity_groups_indices", int64[:, :]),
    ("num_components", int64),
    ("num_sub_components", int64),
    ("num_products", int64),
    ("num_nodes", int64),
    ("lead_times", int64[:]),
    ("capacities", int64[:]),
    ("h", float64[:]),
    ("p", float64[:]),
    ("aggregate_demand_ar_params", float64[:]),
    ("aggregate_demand_num_lags", int64),
    ("aggregate_demand_error_sigma", float64),
    ("product_demand_mean", float64[:]),
    ("product_demand_split_kappa", float64[:]),
    ("product_demand_split_theta", float64[:]),
    ("aggregate_demand_mean", float64),
    ("avg_demand_per_product", float64[:]),
    ("avg_demand_per_comp", float64[:]),
    ("aggregate_demand_model_history", float64[:]),
    ("expected_product_shares", float64[:]),
    ("variance_product_shares", float64[:, :]),
    ("echelon_lead_time_demand_std", float64[:]),
    ("allocation_rounding_variance", float64[:])
]


@jitclass(cfg_spec)
class SupplyChainConfigJIT:
    def __init__(
        self,
        n_predecessors,
        n_successors,
        predecessors,
        successors,
        capacity_groups,
        capacity_groups_size,
        capacity_groups_indices,
        num_components,
        num_sub_components,
        num_products,
        num_nodes,
        lead_times,
        capacities,
        h,
        p,
        aggregate_demand_ar_params,
        aggregate_demand_num_lags,
        aggregate_demand_error_sigma,
        product_demand_mean,
        product_demand_split_kappa,
        product_demand_split_theta,
        aggregate_demand_mean,
        avg_demand_per_product,
        avg_demand_per_comp,
        aggregate_demand_model_history,
        expected_product_shares,
        variance_product_shares,
        echelon_lead_time_demand_std,
        allocation_rounding_variance,
    ):

        self.n_predecessors = n_predecessors
        self.n_successors = n_successors
        self.predecessors = predecessors
        self.successors = successors
        self.capacity_groups = capacity_groups
        self.capacity_groups_size = capacity_groups_size
        self.capacity_groups_indices = capacity_groups_indices
        self.num_components = num_components
        self.num_sub_components = num_sub_components
        self.num_products = num_products
        self.num_nodes = num_nodes
        self.lead_times = lead_times
        self.capacities = capacities
        self.h = h
        self.p = p
        self.aggregate_demand_ar_params = aggregate_demand_ar_params
        self.aggregate_demand_num_lags = aggregate_demand_num_lags
        self.aggregate_demand_error_sigma = aggregate_demand_error_sigma
        self.product_demand_mean = product_demand_mean
        self.product_demand_split_kappa = product_demand_split_kappa
        self.product_demand_split_theta = product_demand_split_theta
        self.aggregate_demand_mean = aggregate_demand_mean
        self.avg_demand_per_product = avg_demand_per_product
        self.avg_demand_per_comp = avg_demand_per_comp
        self.aggregate_demand_model_history = aggregate_demand_model_history
        self.expected_product_shares = expected_product_shares
        self.variance_product_shares = variance_product_shares
        self.echelon_lead_time_demand_std = echelon_lead_time_demand_std
        self.allocation_rounding_variance = allocation_rounding_variance
