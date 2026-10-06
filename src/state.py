import numpy as np
from numba import float64, int64
from numba.experimental import jitclass


state_spec = [
    ("inventory_per_node", int64[:]),
    ("inventory_in_pipeline_per_node", int64[:, :]),
    ("aggregate_demand_model_history", float64[:]),
]

@jitclass(state_spec)
class StateJIT:
    def __init__(
        self,
        inventory_per_node: np.ndarray,
        inventory_in_pipeline_per_node: np.ndarray,
        aggregate_demand_model_history: np.ndarray,
    ):
        self.inventory_per_node = inventory_per_node
        self.inventory_in_pipeline_per_node = inventory_in_pipeline_per_node
        self.aggregate_demand_model_history = aggregate_demand_model_history
