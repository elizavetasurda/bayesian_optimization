"""Base class for Bayesian optimization."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel


class BaseBayesianOptimization(ABC):
    """Abstract base class for Bayesian optimization methods."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        n_init: int = 10,
        kernel: str = "matern",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize base optimizer.

        Args:
            bounds: Variable bounds [(min, max), ...]
            n_init: Initial sample size
            kernel: Kernel type ('rbf' or 'matern')
            random_state: Random seed
        """
        self.bounds = bounds
        self.n_init = n_init
        self.dim = len(bounds)
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Select and configure kernel for GP
        if kernel == "rbf":
            base_kernel = RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2)
            )
        elif kernel == "matern_12":
            base_kernel = Matern(
                length_scale=1.0,
                nu=0.5,
                length_scale_bounds=(1e-2, 1e2)
            )
        elif kernel == "matern_32":
            base_kernel = Matern(
                length_scale=1.0,
                nu=1.5,
                length_scale_bounds=(1e-2, 1e2)
            )
        else:  # matern_25 default
            base_kernel = Matern(
                length_scale=1.0,
                nu=2.5,
                length_scale_bounds=(1e-2, 1e2)
            )

        kernel_full = base_kernel + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-6, 1e-2)
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel_full,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=random_state,
        )

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray) -> None:
        """Initialize model with initial data."""
        self.X = X_init.copy()
        self.y = y_init.copy()
        self._update_gp()

    def _update_gp(self) -> None:
        """Update GP model with current data."""
        if self.X is not None and len(self.X) > 0 and self.y is not None:
            self.gp.fit(self.X, self.y)

    @abstractmethod
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Acquisition function (must be overridden)."""
        pass

    def _optimize_acquisition(self, n_restarts: int = 25) -> np.ndarray:
        """
        Optimize acquisition function with multiple restarts.

        Args:
            n_restarts: Number of random restarts (increased from 10 to 25)

        Returns:
            Point maximizing acquisition function
        """
        from scipy.optimize import minimize

        best_x = None
        best_acq = -np.inf

        bounds_array = np.array(self.bounds)

        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])

            def neg_acq(x: np.ndarray) -> float:
                """Negative acquisition for minimization."""
                acq_val = self._acquisition_function(x.reshape(1, -1))
                if isinstance(acq_val, np.ndarray):
                    acq_val = acq_val[0] if len(acq_val) > 0 else -np.inf
                return -float(acq_val)

            result = minimize(
                neg_acq,
                x0,
                bounds=self.bounds,
                method="L-BFGS-B",
                options={"ftol": 1e-8, "gtol": 1e-7, "maxiter": 500},  # stricter tolerance
            )

            if result.fun < -best_acq:
                best_acq = -result.fun
                best_x = result.x

        if best_x is None:
            best_x = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])

        return best_x

    def suggest_next_point(self) -> np.ndarray:
        """
        Suggest next point for evaluation.

        If initial sample not filled, returns random point.
        Otherwise optimizes acquisition function with 20% exploration chance.

        Returns:
            Next point to evaluate
        """
        if self.X is None or len(self.X) < self.n_init:
            bounds_array = np.array(self.bounds)
            return np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])

        # 20% chance of random exploration to avoid local minima
        if np.random.random() < 0.2:
            bounds_array = np.array(self.bounds)
            return np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])

        best_x = self._optimize_acquisition()
        return np.array(best_x).flatten()

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Update model with new data."""
        X_new_reshaped = X_new.reshape(1, -1)

        if self.X is None:
            self.X = X_new_reshaped
            self.y = np.array([y_new])
        else:
            self.X = np.vstack([self.X, X_new_reshaped])
            self.y = np.append(self.y, y_new)

        self._update_gp()

    def get_best_point(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get best found point."""
        if self.y is None or len(self.y) == 0:
            return None, None

        best_idx = int(np.argmin(self.y))
        if self.X is not None:
            return self.X[best_idx].copy(), float(self.y[best_idx])

        return None, None