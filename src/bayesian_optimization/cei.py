"""Реализация метода CEI (Constrained Expected Improvement)."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .base import BayesianOptimizationBase


class CEIBayesianOptimization(BayesianOptimizationBase):
    """Байесовская оптимизация с CEI acquisition function."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gp = GaussianProcessRegressor(
            kernel=C(1.0) * RBF(1.0),
            normalize_y=True,
            random_state=42,
        )
        self._init_sample()

    def _init_sample(self) -> None:
        """Создаёт начальную случайную выборку."""
        for _ in range(self.n_initial):
            x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            fx = self.objective(x)
            self.X.append(x)
            self.F.append(fx)
            if self._is_feasible(x) and fx < self.best_feasible_value:
                self.best_feasible_value = fx
                self.best_feasible_point = x

    def iterate(self) -> None:
        """Одна итерация: обновляет GP и выбирает новую точку."""
        X_arr = np.array(self.X)
        F_arr = np.array(self.F)

        self.gp.fit(X_arr, F_arr)

        # Здесь должна быть реализация acquisition function
        # Для краткости — случайная точка
        x_new = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        fx_new = self.objective(x_new)

        self.X.append(x_new)
        self.F.append(fx_new)

        if self._is_feasible(x_new) and fx_new < self.best_feasible_value:
            self.best_feasible_value = fx_new
            self.best_feasible_point = x_new