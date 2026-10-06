import sys
sys.path.append(".")

import numpy as np
from src.supply_chain_config import SupplyChainConfig
from src.base_stock import StaticBaseStockPolicyShortfallJIT
from src.parser import get_ablation_kwargs, parse_args
from numba import njit

@njit
def mc_alloc_variance_uniform(Lij: np.ndarray,  # (P, C) int; lead time of end item j on component i
                              Lmax: int,        # horizon: typically Lij.max()
                              n_draws: int,
                              seed: int = 0):
    """
    Uniform-only proxy:
      f_j ~ Unif(0,1)  (fractional parts)
      R = sum_j f_j
      p_j = f_j / R
      p_{i,k} = sum_{j: Lij[j,i] >= k+1} p_j
      V_alloc[i,k] = E[ R * p_{i,k} * (1 - p_{i,k}) ]  (MC estimate)

    Returns:
      V_alloc: (C, Lmax) per-offset contributions
      V_alloc_sum: (C,)   per-component sum over k
    """
    P, C = Lij.shape
    V_alloc = np.zeros((C, Lmax), dtype=np.float64)
    np.random.seed(seed)

    # buffers
    f = np.empty(P, dtype=np.float64)
    p = np.empty(P, dtype=np.float64)
    diff = np.empty(Lmax + 1, dtype=np.float64)

    for s in range(n_draws):
        # draw uniforms for fractional parts
        for j in range(P):
            f[j] = np.random.random()
        R = 0.0
        for j in range(P):
            R += f[j]
        if R <= 1e-15:
            continue  # essentially impossible, but safe

        invR = 1.0 / R
        for j in range(P):
            p[j] = f[j] * invR

        # for each component, build subset sum of p_j over k via a difference array
        for i in range(C):
            # reset diff
            for idx in range(Lmax + 1):
                diff[idx] = 0.0
            # add p_j to all k < Lij[j,i]
            for j in range(P):
                L = Lij[j, i]
                if L > 0:
                    v = p[j]
                    diff[0] += v
                    if L < Lmax:
                        diff[L] -= v
            # cumulative to get p_{i,k}, then accumulate R * p_{i,k} * (1 - p_{i,k})
            run = 0.0
            for k in range(Lmax):
                run += diff[k]      # this is p_{i,k}
                V_alloc[i, k] += R * run * (1.0 - run)

    # average
    inv = 1.0 / n_draws
    for i in range(C):
        for k in range(Lmax):
            V_alloc[i, k] *= inv

    # per-component sum over k
    V_alloc_sum = np.zeros(C, dtype=np.float64)
    for i in range(C):
        ssum = 0.0
        for k in range(Lmax):
            ssum += V_alloc[i, k]
        V_alloc_sum[i] = ssum

    return V_alloc_sum

args = parse_args()
# This precompute depends only on the network structure / lead times.
# The demand scaling parameter does not affect the result.
config = SupplyChainConfig(
    **get_ablation_kwargs(args),
)
base = StaticBaseStockPolicyShortfallJIT(
    config.get_jit_class(load_precomputed=False),
    hedging_parameter=np.zeros(config.num_components, dtype=np.float64),
)
Lij = base.get_echelon_lead_time_per_product()
Lmax = np.max(Lij)
V_alloc_sum = mc_alloc_variance_uniform(Lij, Lmax, n_draws=10_000_000)
np.savetxt('data/precomputed/allocation_rounding_variance.txt', np.around(V_alloc_sum, decimals=6))
