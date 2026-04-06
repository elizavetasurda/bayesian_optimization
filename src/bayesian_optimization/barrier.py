"""Реализация метода Barrier (логарифмический барьер)."""

import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from .base import BayesianOptimizationBase


class BarrierBayesianOptimization(BayesianOptimizationBase):
    """Байесовская оптимизация с логарифмической барьерной функцией."""

    def __init__(self, barrier_mu: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.barrier_mu = barrier_mu
        kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1)) + WhiteKernel(
            1e-4, (1e-6, 1e-2)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=10,
        )
        self._init_random_sample()
        self._fit_gp()

    def _barrier(self, x: np.ndarray) -> float:
        """Вычисляет барьерную функцию: f(x) - mu * sum(log(-g_i(x)))."""
        fx = self.objective(x)
        barrier = 0.0
        for constraint in self.constraints:
            g = constraint(x)
            if g >= 0:  # Нарушение ограничения → большое значение
                return 1e12
            barrier += math.log(-g)
        return fx - self.barrier_mu * barrier

    def _fit_gp(self) -> None:
        if len(self.X) < 2:
            return
        X_arr = np.array(self.X)
        barrier_F = [self._barrier(x) for x in self.X]
        self.gp.fit(X_arr, barrier_F)

    def _acquisition(self, x: np.ndarray) -> float:
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]

        if sigma < 1e-12:
            return 0.0

        best_barrier = min(self._barrier(xi) for xi in self.X)
        gamma = (best_barrier - mu) / sigma
        return (best_barrier - mu) * self._norm_cdf(gamma) + sigma * self._norm_pdf(
            gamma
        )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def iterate(self) -> None:
        if len(self.X) < 2:
            self._init_random_sample()
            self._fit_gp()
            return

        best_x = None
        best_acq = -np.inf

        for _ in range(20):
            x_candidate = np.array(
                [np.random.uniform(low, high) for low, high in self.bounds]
            )
            acq = self._acquisition(x_candidate)
            if acq > best_acq:
                best_acq = acq
                best_x = x_candidate

        if best_x is None:
            best_x = np.array(
                [np.random.uniform(low, high) for low, high in self.bounds]
            )

        fx_new = self.objective(best_x)
        self.X.append(best_x)
        self.F.append(fx_new)

        if self._is_feasible(best_x) and fx_new < self.best_feasible_value:
            self.best_feasible_value = fx_new
            self.best_feasible_point = best_x.copy()

        self._fit_gp()
