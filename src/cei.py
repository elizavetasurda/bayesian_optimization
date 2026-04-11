"""CEI (Constrained Expected Improvement) method."""

from typing import Optional

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from src.bayesian_optimization.base import BaseBayesianOptimization


class CEIBayesianOptimization(BaseBayesianOptimization):
    """Bayesian optimization with CEI."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        constraints: list,
        n_init: int = 10,
        kernel: str = "matern",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize CEI optimizer."""
        super().__init__(bounds, n_init, kernel, random_state)

        self.constraints = constraints
        self.n_constraints = len(constraints)

        self.constraint_gps: list[GaussianProcessRegressor] = []
        self._init_constraint_gps(kernel, random_state)

        self.constraint_values: Optional[np.ndarray] = None

    def _init_constraint_gps(self, kernel: str, random_state: Optional[int]) -> None:
        """Initialize GP models for constraints."""
        for _ in range(self.n_constraints):
            if kernel == "rbf":
                base_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            else:
                base_kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2))

            kernel_full = base_kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-2))

            gp = GaussianProcessRegressor(
                kernel=kernel_full,
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=random_state,
            )
            self.constraint_gps.append(gp)

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray) -> None:
        """Initialize with constraint values."""
        super().initialize(X_init, y_init)

        self.constraint_values = np.zeros((len(X_init), self.n_constraints))

        for i, x in enumerate(X_init):
            for j, constraint in enumerate(self.constraints):
                self.constraint_values[i, j] = float(constraint(x))

        for j in range(self.n_constraints):
            self.constraint_gps[j].fit(self.X, self.constraint_values[:, j])

    def _constraint_feasibility_probability(self, X: np.ndarray) -> np.ndarray:
        """Compute probability of satisfying all constraints."""
        X = X.reshape(-1, self.dim)
        prob = np.ones(len(X))

        for j, gp in enumerate(self.constraint_gps):
            mu, sigma = gp.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-6)
            prob_j = norm.cdf(-mu / sigma)
            prob *= prob_j

        return prob

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Compute standard Expected Improvement."""
        X = X.reshape(-1, self.dim)

        if self.y is None:
            return np.zeros(len(X))

        f_min = np.min(self.y)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        gamma = (f_min - mu) / sigma
        ei = (f_min - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        return np.maximum(ei, 0)

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """CEI acquisition function."""
        ei = self._expected_improvement(X)
        prob_feasible = self._constraint_feasibility_probability(X)

        result = ei * prob_feasible
        if result.ndim == 0:
            result = np.array([result])
        return result

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Update model with new data."""
        super().update(X_new, y_new)

        if self.constraint_values is not None:
            new_constraint_values = np.zeros((1, self.n_constraints))

            for j, constraint in enumerate(self.constraints):
                new_constraint_values[0, j] = float(constraint(X_new))

            self.constraint_values = np.vstack([self.constraint_values, new_constraint_values])

            for j in range(self.n_constraints):
                self.constraint_gps[j].fit(self.X, self.constraint_values[:, j])