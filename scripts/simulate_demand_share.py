import sys
sys.path.append(".")

import numpy as np
from src.supply_chain_config import SupplyChainConfig
from src.base_stock import StaticBaseStockPolicyShortfallJIT
from src.parser import get_ablation_kwargs, parse_args
from numba import njit

@njit
def build_share_variance_gamma_mc(
    echelon_lead_time: np.ndarray,  # (num_products, num_components) = Lij (int)
    gamma_shape: np.ndarray,        # (num_products,) = α_j
    gamma_scale: np.ndarray,         # (num_products,) = β_j  (NumPy uses scale=1/β)
    maximum_echelon_lead_time: int, # Lmax
    n_draws: int,
    seed: int = 0
) -> np.ndarray:
    """
    Estimates Var_c[i,k] by Monte Carlo:
      draw G_j ~ Gamma(shape=α_j, scale=1/β_j), π_j = G_j / sum G
      c[i,k] = sum_{j: k+1 <= L_ij} π_j
      Var_c[i,k] = Var_s[c[i,k]]
    """
    np.random.seed(seed)
    num_products, num_components = echelon_lead_time.shape
    L = maximum_echelon_lead_time
    if L <= 0 or n_draws <= 0:
        return np.zeros((num_components, L), dtype=np.float64)

    sum_c   = np.zeros((num_components, L), dtype=np.float64)
    sumsq_c = np.zeros((num_components, L), dtype=np.float64)

    # Work buffers
    G = np.empty(num_products, dtype=np.float64)
    pi = np.empty(num_products, dtype=np.float64)

    for s in range(n_draws):
        # 1) Draw Gammas and normalize to shares
        tot = 0.0
        for j in range(num_products):
            g = np.random.gamma(gamma_shape[j], gamma_scale[j])
            G[j] = g
            tot += g
        if tot <= 0.0:
            # extremely unlikely unless shapes tiny; skip
            continue
        inv_tot = 1.0 / tot
        for j in range(num_products):
            pi[j] = G[j] * inv_tot

        # 2) For each component build c[i, :] via difference array
        for i in range(num_components):
            diff = np.zeros(L + 1, dtype=np.float64)
            # add π_j to all k < L_ij
            for j in range(num_products):
                Lij = echelon_lead_time[j, i]
                if Lij > 0:
                    v = pi[j]
                    diff[0] += v
                    if Lij < L:
                        diff[Lij] -= v
            run = 0.0
            for k in range(L):
                run += diff[k]
                c = run
                sum_c[i, k]   += c
                sumsq_c[i, k] += c * c

    # 3) Unbiased sample variance (population variance if you prefer: divide by n_draws)
    Var_c = np.empty((num_components, L), dtype=np.float64)
    n = float(n_draws)
    for i in range(num_components):
        for k in range(L):
            mean = sum_c[i, k] / n
            Var_c[i, k] = max(sumsq_c[i, k] / n - mean * mean, 0.0)
    return Var_c

args = parse_args()
config = SupplyChainConfig(
    scaling=args.scaling,
    **get_ablation_kwargs(args),
)
base = StaticBaseStockPolicyShortfallJIT(
    config.get_jit_class(load_precomputed=False),
    hedging_parameter=np.zeros(config.num_components, dtype=np.float64),
)
Lij = base.get_echelon_lead_time_per_product()
Lmax = np.max(Lij)
Var_c = build_share_variance_gamma_mc(Lij, np.array(config.product_demand_split_kappa), np.array(config.product_demand_split_theta), Lmax, n_draws=10_000_000)
np.savetxt(f'data/precomputed/variance_product_shares_echelon_lead_time_scaling{config.scaling}.txt', np.around(Var_c, decimals=6))
