import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import float64, int64, prange

from src.base_stock import StaticBaseStockPolicyShortfallJIT, DynamicBaseStockPolicyShortfallJIT
from src.evaluation_utils import evaluate_policy_jit, precompute_demand_realizations
from src.dynamics import DynamicsJIT

# ============================================================
# CMA-ES (vanilla rank-µ + rank-1 with CSA step-size control)
# ============================================================

cma_es_spec = [
    ("dimension", int64),
    ("population_size", int64),
    ("num_parents", int64),
    ("global_step_size", float64),
    ("mean_parameters", float64[:]),  # current distribution mean (m)
    ("covariance_matrix", float64[:, :]),  # C
    ("eigenvectors", float64[:, :]),  # B
    ("eigenvalues_sqrt", float64[:]),  # D (sqrt of eigenvalues)
    ("inv_sqrt_covariance", float64[:, :]),  # C^{-1/2} = B * diag(1/D) * B^T
    ("evolution_path_c", float64[:]),  # pc (for covariance)
    ("evolution_path_sigma", float64[:]),  # ps (for step-size)
    ("recombination_weights", float64[:]),  # w_i (log weights)
    ("mu_effective", float64),  # mueff
    ("c_cov", float64),  # cc (cumulation for C)
    ("c_sigma", float64),  # cs (cumulation for sigma)
    ("c_rank_one", float64),  # c1 (rank-1 update coeff)
    ("c_rank_mu", float64),  # cmu (rank-µ update coeff)
    ("damping_sigma", float64),  # damping factor for sigma
    ("num_func_evaluations", int64),  # total evaluations so far
    ("last_eigensystem_eval_at", int64),  # when eigensystem last updated
    ("expected_norm_gaussian", float64),  # E||N(0,I)|| in R^n
    ("random_seed", int64),
]


@jitclass(cma_es_spec)
class HedgingOptimizer:
    """
    A clear, Numba-jittable implementation of the CMA-ES algorithm.

    Notes
    -----
    - Samples are drawn as: x_k = mean_parameters + global_step_size * (B @ (D * z_k)),
      where z_k ~ N(0, I), B are eigenvectors, and D are sqrt(eigenvalues).
    - Step-size (global_step_size) is adapted by CSA using evolution_path_sigma.
    - Covariance matrix is adapted using a rank-1 term (from evolution_path_c)
      and a rank-µ term (from the top µ weighted samples).
    - This is the "vanilla" CMA-ES (no active covariance, no restarts).
    """

    def __init__(self, initial_mean, initial_sigma=0.2, population_size=0, seed=0):
        """
        Parameters
        ----------
        initial_mean : np.ndarray (shape: [n])
            Initial center of the search distribution.
        initial_sigma : float
            Initial global step-size (typical per-coordinate std at start).
        population_size : int
            Number of candidates per generation (lambda).
            If <= 0, uses default: 4 + floor(3 * ln(n)).
        seed : int
            RNG seed for reproducibility.
        """
        # --- Core dimensions and RNG ---
        self.dimension = initial_mean.shape[0]
        self.global_step_size = initial_sigma
        self.mean_parameters = initial_mean.copy()
        self.random_seed = seed
        np.random.seed(seed)

        # --- Population configuration ---
        if population_size <= 0:
            self.population_size = 4 + int(3.0 * np.log(self.dimension))
        else:
            self.population_size = population_size
        self.num_parents = self.population_size // 2

        # --- Recombination weights (logarithmic) ---
        weights = np.zeros(self.num_parents)
        for i in range(self.num_parents):
            weights[i] = np.log(self.num_parents + 0.5) - np.log(i + 1.0)
        weights = weights / np.sum(weights)
        self.recombination_weights = weights
        self.mu_effective = 1.0 / np.sum(weights * weights)

        # --- Strategy parameter settings (vanilla defaults) ---
        # Cumulation for covariance and step-size:
        self.c_cov = (4.0 + self.mu_effective / self.dimension) / (
            self.dimension + 4.0 + 2.0 * self.mu_effective / self.dimension
        )
        self.c_sigma = (self.mu_effective + 2.0) / (
            self.dimension + self.mu_effective + 5.0
        )

        # Learning rates for covariance updates:
        self.c_rank_one = 2.0 / ((self.dimension + 1.3) ** 2 + self.mu_effective)
        self.c_rank_mu = min(
            1.0 - self.c_rank_one,
            2.0
            * (self.mu_effective - 2.0 + 1.0 / self.mu_effective)
            / ((self.dimension + 2.0) ** 2 + self.mu_effective),
        )

        # Damping for step-size adaptation:
        self.damping_sigma = (
            1.0
            + 2.0
            * max(
                0.0, np.sqrt((self.mu_effective - 1.0) / (self.dimension + 1.0)) - 1.0
            )
            + self.c_sigma
        )

        # --- State initialization ---
        self.covariance_matrix = np.eye(self.dimension)
        self.eigenvectors = np.eye(self.dimension)
        self.eigenvalues_sqrt = np.ones(self.dimension)
        self.inv_sqrt_covariance = np.eye(self.dimension)
        self.evolution_path_c = np.zeros(self.dimension)
        self.evolution_path_sigma = np.zeros(self.dimension)

        self.num_func_evaluations = 0
        self.last_eigensystem_eval_at = 0

        # Expected norm of a standard normal vector in R^n (good approximation):
        n = self.dimension
        self.expected_norm_gaussian = np.sqrt(n) * (
            1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)
        )

    def _update_eigensystem(self):
        """
        Recompute eigen-decomposition of the covariance matrix C:
        C = B * diag(D^2) * B^T, and inv_sqrt_covariance = B * diag(1/D) * B^T.
        Guard against tiny negative/zero eigenvalues.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        # Numerical guard to keep positive semidefiniteness
        for i in range(self.dimension):
            if eigenvalues[i] < 1e-10:
                eigenvalues[i] = 1e-10

        self.eigenvalues_sqrt = np.sqrt(eigenvalues)
        self.eigenvectors = eigenvectors

        # inv_sqrt_covariance = B * diag(1/D) * B^T
        # Compute via column-wise division to avoid forming explicit diag matrices.
        temp = (
            self.eigenvectors / self.eigenvalues_sqrt
        )  # broadcast divide columns by D
        self.inv_sqrt_covariance = np.dot(temp, self.eigenvectors.T)

    def ask(self):
        """
        Sample a new population of candidate solutions.

        Returns
        -------
        candidates_parameters : (population_size, dimension) ndarray
            The parameter vectors to evaluate.
        candidates_standard_normals : (population_size, dimension) ndarray
            The underlying z ~ N(0, I) vectors used to produce each sample
            (needed for the covariance update in tell()).
        """
        candidates_parameters = np.zeros((self.population_size, self.dimension))
        candidates_standard_normals = np.zeros((self.population_size, self.dimension))

        # Draw z ~ N(0, I), then map through current shape and scale: x = m + σ * (B @ (D * z))
        for k in range(self.population_size):
            z = np.random.normal(0.0, 1.0, self.dimension)
            Dz = self.eigenvalues_sqrt * z
            shape_transformed = np.dot(self.eigenvectors, Dz)
            x = self.mean_parameters + self.global_step_size * shape_transformed

            candidates_parameters[k, :] = x
            candidates_standard_normals[k, :] = z

        return candidates_parameters, candidates_standard_normals

    def tell(
        self, candidates_parameters, candidates_standard_normals, objective_values
    ):
        """
        Update CMA-ES state from evaluated candidates.

        Parameters
        ----------
        candidates_parameters : (population_size, dimension) ndarray
            Candidate parameter vectors returned by ask().
        candidates_standard_normals : (population_size, dimension) ndarray
            The corresponding z ~ N(0, I) vectors from ask().
        objective_values : (population_size,) ndarray
            Objective function values for each candidate (lower is better).
        """
        # --- Sort candidates by fitness (ascending) ---
        sorted_indices = np.argsort(objective_values)

        # --- Save old mean and prepare weighted sums ---
        previous_mean = self.mean_parameters.copy()
        weighted_mean_step_in_coords = np.zeros(
            self.dimension
        )  # y_w = Σ w_i * ((x_i - m_old)/σ)
        weighted_mean_z = np.zeros(self.dimension)  # z_w = Σ w_i * z_i

        # --- Recombination: compute new mean and weighted steps ---
        for i in range(self.num_parents):
            idx = sorted_indices[i]
            x_i = candidates_parameters[idx, :]
            z_i = candidates_standard_normals[idx, :]
            weight_i = self.recombination_weights[i]

            step_i = (
                x_i - previous_mean
            ) / self.global_step_size  # step in "phenotype" coords
            weighted_mean_step_in_coords += weight_i * step_i
            weighted_mean_z += weight_i * z_i

        # Update mean
        self.mean_parameters = (
            previous_mean + self.global_step_size * weighted_mean_step_in_coords
        )

        # --- Evolution path for step-size (ps) ---
        # ps ← (1 - c_sigma) ps + sqrt(c_sigma(2-c_sigma) * mueff) * C^{-1/2} * y_w
        scalar_ps = np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_effective)
        self.evolution_path_sigma = (
            1.0 - self.c_sigma
        ) * self.evolution_path_sigma + scalar_ps * np.dot(
            self.inv_sqrt_covariance, weighted_mean_step_in_coords
        )

        # --- Heuristic for rank-1 covariance update activity (h_sigma) ---
        self.num_func_evaluations += self.population_size
        ps_norm = np.linalg.norm(self.evolution_path_sigma)
        one_minus_term = 1.0 - (1.0 - self.c_sigma) ** (
            2.0 * (self.num_func_evaluations / self.population_size)
        )
        h_sigma_condition = ps_norm / (
            np.sqrt(max(1e-12, one_minus_term)) * self.expected_norm_gaussian
        )
        h_sigma = (
            1.0 if (h_sigma_condition < (1.4 + 2.0 / (self.dimension + 1.0))) else 0.0
        )

        # --- Evolution path for covariance (pc) ---
        # pc ← (1 - c_cov) pc + h_sigma * sqrt(c_cov(2-c_cov) * mueff) * y_w
        scalar_pc = np.sqrt(self.c_cov * (2.0 - self.c_cov) * self.mu_effective)
        self.evolution_path_c = (
            (1.0 - self.c_cov) * self.evolution_path_c
            + h_sigma * scalar_pc * weighted_mean_step_in_coords
        )

        # --- Rank-µ covariance contribution from top µ candidates ---
        rank_mu_update = np.zeros((self.dimension, self.dimension))
        for i in range(self.num_parents):
            idx = sorted_indices[i]
            step_i = (
                candidates_parameters[idx, :] - previous_mean
            ) / self.global_step_size
            weight_i = self.recombination_weights[i]
            rank_mu_update += weight_i * np.outer(step_i, step_i)

        # --- Full covariance update ---
        # C ← (1 - c1 - cmu) C + c1(pc pc^T + (1 - h_sigma) * c_cov(2-c_cov) C) + cmu * rank_mu_update
        self.covariance_matrix = (
            (1.0 - self.c_rank_one - self.c_rank_mu) * self.covariance_matrix
            + self.c_rank_one
            * (
                np.outer(self.evolution_path_c, self.evolution_path_c)
                + (1.0 - h_sigma)
                * self.c_cov
                * (2.0 - self.c_cov)
                * self.covariance_matrix
            )
            + self.c_rank_mu * rank_mu_update
        )

        # --- Step-size control (CSA) ---
        self.global_step_size = self.global_step_size * np.exp(
            (self.c_sigma / self.damping_sigma)
            * (ps_norm / self.expected_norm_gaussian - 1.0)
        )

        # --- Periodic eigendecomposition to refresh B, D, C^{-1/2} ---
        # Classic rule of thumb: every ~ 1 / ((c1 + cmu) * n) iterations; we implement a simple counter-based trigger.
        if (self.num_func_evaluations - self.last_eigensystem_eval_at) > (
            self.population_size
            / (self.c_rank_one + self.c_rank_mu)
            / self.dimension
            / 10.0
        ):
            self._update_eigensystem()
            self.last_eigensystem_eval_at = self.num_func_evaluations


@njit(parallel=True)
def optimize_static(
    optimizer: HedgingOptimizer,
    dynamics: DynamicsJIT,
    num_generations: int,
    candidate_seeds: np.ndarray,
    demand_seeds: np.ndarray,
    num_trajectories: int = 1_000,
    periods_per_trajectory: int = 60,
    warmup_periods: int = 0,
    num_trajectories_increment: int = 100,
):
    """
    Tune the safety-stock factors of the static base-stock policy with CMA-ES.

    Every generation draws new demand paths and evaluates all candidates on them.

    Parameters
    ----------
    optimizer : HedgingOptimizer
        CMA-ES state, initialized with the starting factors.
    dynamics : DynamicsJIT
        Dynamics the candidates are simulated with.
    num_generations : int
        Number of CMA-ES generations.
    candidate_seeds, demand_seeds : np.ndarray
        One seed per generation for sampling the candidates and the demand paths.
    num_trajectories : int
        Trajectories per candidate evaluation in the first generation.
    periods_per_trajectory : int
        Periods per trajectory whose cost counts.
    warmup_periods : int
        Periods simulated before the cost starts to count.
    num_trajectories_increment : int
        Trajectories added to the evaluation after every generation.

    Returns
    -------
    mean_history : np.ndarray, shape (num_generations, dimension)
        Mean of the search distribution after each generation; the last row is the result.
    sigma_history : np.ndarray, shape (num_generations,)
        Global step size after each generation.
    """
    mean_history = np.zeros((num_generations, optimizer.dimension))
    sigma_history = np.zeros(num_generations)
    for g in range(num_generations):
        demand_realizations = precompute_demand_realizations(
            dynamics.supply_chain_config,
            dynamics.supply_chain_config.aggregate_demand_model_history,
            num_trajectories,
            periods_per_trajectory + warmup_periods,
            int(demand_seeds[g]),
        )
        np.random.seed(int(candidate_seeds[g]))
        candidates, standard_normals = optimizer.ask()
        fvals = np.zeros(optimizer.population_size)
        for k in prange(optimizer.population_size):
            policy = StaticBaseStockPolicyShortfallJIT(
                    dynamics.supply_chain_config, candidates[k, :], True
                )
            fvals[k], _ = evaluate_policy_jit(
                dynamics,
                policy,
                num_trajectories,
                periods_per_trajectory,
                demand_realizations,
                warmup_periods,
            )
        optimizer.tell(candidates, standard_normals, fvals)
        num_trajectories += num_trajectories_increment
        mean_history[g, :] = optimizer.mean_parameters.copy()
        sigma_history[g] = optimizer.global_step_size
    return (mean_history, sigma_history)

@njit(parallel=True)
def optimize_dynamic(
    optimizer: HedgingOptimizer,
    dynamics: DynamicsJIT,
    num_generations: int,
    candidate_seeds: np.ndarray,
    demand_seeds: np.ndarray,
    num_trajectories: int = 1_000,
    periods_per_trajectory: int = 60,
    warmup_periods: int = 0,
    num_trajectories_increment: int = 100,
):
    """
    Tune the safety-stock factors of the dynamic base-stock policy with CMA-ES.

    Same parameters and return values as optimize_static.
    """
    mean_history = np.zeros((num_generations, optimizer.dimension))
    sigma_history = np.zeros(num_generations)
    for g in range(num_generations):
        demand_realizations = precompute_demand_realizations(
            dynamics.supply_chain_config,
            dynamics.supply_chain_config.aggregate_demand_model_history,
            num_trajectories,
            periods_per_trajectory + warmup_periods,
            int(demand_seeds[g]),
        )
        np.random.seed(int(candidate_seeds[g]))
        candidates, standard_normals = optimizer.ask()
        fvals = np.zeros(optimizer.population_size)
        for k in prange(optimizer.population_size):
            policy = DynamicBaseStockPolicyShortfallJIT(
                    dynamics.supply_chain_config, candidates[k, :], True
                )
            fvals[k], _ = evaluate_policy_jit(
                dynamics,
                policy,
                num_trajectories,
                periods_per_trajectory,
                demand_realizations,
                warmup_periods,
            )
        optimizer.tell(candidates, standard_normals, fvals)
        num_trajectories += num_trajectories_increment
        mean_history[g, :] = optimizer.mean_parameters.copy()
        sigma_history[g] = optimizer.global_step_size
    return (mean_history, sigma_history)
