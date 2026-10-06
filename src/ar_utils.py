import numpy as np
from numba import njit

@njit
def build_ar_transition_matrix(ar_coefficients: np.ndarray) -> np.ndarray:
    """
    ar_coefficients: (p,) AR coefficients [phi_1, ..., phi_p]; the model uses the zero-mean state x_t.
    Returns the companion matrix A: (p, p).
    """
    num_lags = ar_coefficients.shape[0]
    A = np.zeros((num_lags, num_lags), dtype=np.float64)
    # first row
    for j in range(num_lags):
        A[0, j] = ar_coefficients[j]
    # subdiagonal 1's
    for r in range(1, num_lags):
        A[r, r-1] = 1.0
    return A

@njit
def build_echelon_horizon_weight_matrix(echelon_lead_time: np.ndarray, Lmax: int, demand_shares: np.ndarray) -> np.ndarray:
    """
    echelon_lead_time: (num_products, num_components); L[j, i] is the echelon lead time of product j on component i (0 => not used)
    demand_shares: (num_products,) expected shares q (sum ~ 1)
    Returns W: (num_components, Lmax) with w_i[k] = sum_j 1_{k+1 <= L[j, i]} * q[j], k=0..Lmax-1
    """
    num_products, num_components = echelon_lead_time.shape

    echelon_horizon_weights = np.zeros((num_components, Lmax), dtype=np.float64)
    # difference-array trick per component
    for i in range(num_components):
        diff = np.zeros(Lmax + 1, dtype=np.float64)
        for j in range(num_products):
            L = echelon_lead_time[j, i]
            if L <= 0:
                continue
            v = demand_shares[j]
            diff[0] += v
            if L < Lmax:
                diff[L] -= v
        run = 0.0
        for k in range(Lmax):
            run += diff[k]
            echelon_horizon_weights[i, k] = run
    return echelon_horizon_weights

@njit
def mean_path_ARp_from_R(demand_mean: float, R: np.ndarray, demand_history: np.ndarray) -> np.ndarray:
    """
    Given R[k,:] = e1^T A^{k+1}, returns m[k] = mu + R[k,:] @ x_t for k=0..Lmax-1,
    with mu = demand_mean and x_t = demand_history - demand_mean.
    """
    demand_history_zero_mean = demand_history - demand_mean
    maximum_echelon_lead_time, num_lags = R.shape
    m = np.empty(maximum_echelon_lead_time, dtype=np.float64)
    for k in range(maximum_echelon_lead_time):
        s = 0.0
        for j in range(num_lags):
            s += R[k, j] * demand_history_zero_mean[j]
        m[k] = demand_mean + s
    return m

@njit
def precompute_mean_path_operator(ar_transition_matrix: np.ndarray, maximum_echelon_lead_time: int) -> np.ndarray:
    """
    R[k,:] = e1^T A^{k+1}, k=0..Lmax-1, with A = ar_transition_matrix and Lmax = maximum_echelon_lead_time
    (row block to get the mean path fast).
    """
    num_lags = ar_transition_matrix.shape[0]
    mean_path_operator = np.empty((maximum_echelon_lead_time, num_lags), dtype=np.float64)
    # r is row vector; start with e1^T, then multiply by A each step
    r = np.zeros(num_lags, dtype=np.float64)
    r[0] = 1.0
    for k in range(maximum_echelon_lead_time):
        # r <- r @ A
        r_next = np.zeros(num_lags, dtype=np.float64)
        for a in range(num_lags):
            s = 0.0
            for b in range(num_lags):
                s += r[b] * ar_transition_matrix[b, a]
            r_next[a] = s
        mean_path_operator[k, :] = r_next
        r = r_next
    return mean_path_operator

@njit
def impulse_response_row(A: np.ndarray, B: np.ndarray, L: int) -> np.ndarray:
    """h[n] = e1^T A^n B, n=0..L-1"""
    p = A.shape[0]
    h = np.empty(L, dtype=np.float64)
    v = B.copy()  # A^0 B
    for n in range(L):
        h[n] = v[0]
        vv = np.zeros(p, dtype=np.float64)
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += A[i, j] * v[j]
            vv[i] = s
        v = vv
    return h

@njit
def build_aggregate_rounding_variance(W, vbar=1.0/6.0):
    C, L = W.shape
    out = np.zeros(C, np.float64)
    for i in range(C):
        s = 0.0
        for k in range(L):
            s += W[i, k] * W[i, k]
        out[i] = vbar * s
    return out

