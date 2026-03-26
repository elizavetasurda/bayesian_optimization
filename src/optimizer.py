"""Модуль байесовской оптимизации с ограничениями."""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Результаты оптимизации."""
    best_x: Optional[np.ndarray]
    best_value: float
    history: List[float]
    first_feasible: Optional[int]
    n_evaluations: int
    is_feasible: bool
    method: str


class BayesianOptimizer:
    """Байесовская оптимизация с ограничениями."""
    
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        constraints: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        method: str = 'CEI',
        n_start: int = 10,
        penalty_coef: float = 1e4,
        lagrange_coef: Tuple[float, float] = (1000.0, 500.0),
        barrier_coef: float = 100.0
    ):
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
        self.method = method
        self.n_start = n_start
        self.penalty_coef = penalty_coef
        self.lagrange_coef = lagrange_coef
        self.barrier_coef = barrier_coef
        
        self._optimization_func = self._get_optimization_func()
        self._is_feasible = lambda x: np.all(constraints(x) <= 1e-6)
        
        self._X: List[np.ndarray] = []
        self._F: List[float] = []
        self._G: List[np.ndarray] = []
        self._best_value: float = 1e9
        self._best_x: Optional[np.ndarray] = None
        self._first_feasible: Optional[int] = None
        self._history: List[float] = []
        self._gp_f: Optional[GaussianProcessRegressor] = None
        self._gp_g: List[GaussianProcessRegressor] = []
    
    def _get_optimization_func(self) -> Callable[[np.ndarray], float]:
        if self.method == 'Penalty':
            return lambda x: self.objective(x) + self.penalty_coef * self._how_bad(x)**2
        elif self.method == 'Lagrange':
            return lambda x: self._lagrange_func(x)
        elif self.method == 'Barrier':
            return lambda x: self._barrier_func(x)
        return self.objective
    
    def _how_bad(self, x: np.ndarray) -> float:
        return np.sum(np.maximum(0, self.constraints(x)))
    
    def _lagrange_func(self, x: np.ndarray) -> float:
        g = self.constraints(x)
        lagrange_val = self.objective(x)
        c1, c2 = self.lagrange_coef
        for gi in g:
            if gi > 0:
                lagrange_val += c1 * gi + c2 * gi**2
        return lagrange_val
    
    def _barrier_func(self, x: np.ndarray) -> float:
        if not self._is_feasible(x):
            return 1e10
        g = self.constraints(x)
        barrier_val = self.objective(x)
        for gi in g:
            barrier_val -= self.barrier_coef * np.log(-gi + 1e-8)
        return barrier_val
    
    def _lhs_sample(self, n: int) -> np.ndarray:
        sampler = LatinHypercube(d=self.bounds.shape[0])
        s = sampler.random(n=n)
        X = np.zeros((n, self.bounds.shape[0]))
        for i in range(self.bounds.shape[0]):
            X[:, i] = s[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        return X
    
    def _init_data(self) -> None:
        X0 = self._lhs_sample(self.n_start)
        for x in X0:
            self._X.append(x)
            self._F.append(self._optimization_func(x))
            self._G.append(self.constraints(x))
            if self._is_feasible(x) and self.objective(x) < self._best_value:
                self._best_value = self.objective(x)
                self._best_x = x.copy()
                if self._first_feasible is None:
                    self._first_feasible = len(self._X) - 1
        self._X = np.array(self._X)
        self._F = np.array(self._F)
        self._G = np.array(self._G)
        self._history.append(self._best_value if self._best_value < 1e9 else np.nan)
    
    def _train_gps(self) -> None:
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.1, nu=2.5) + WhiteKernel(1e-6)
        self._gp_f = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp_f.fit(self._X.reshape(-1, self.bounds.shape[0]), self._F)
        if self.method == 'CEI':
            self._gp_g = []
            n_constraints = self._G.shape[1] if len(self._G.shape) > 1 else 1
            for i in range(n_constraints):
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
                g_vals = self._G[:, i] if n_constraints > 1 else self._G
                gp.fit(self._X.reshape(-1, self.bounds.shape[0]), g_vals)
                self._gp_g.append(gp)
    
    def _ei(self, x: np.ndarray, best_val: float) -> float:
        x = x.reshape(1, -1)
        mu, sigma = self._gp_f.predict(x, return_std=True)
        sigma = sigma[0]
        if sigma < 1e-10:
            return 0.0
        gamma = (best_val - mu) / sigma
        return sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    
    def _p_feasible(self, x: np.ndarray) -> float:
        x = x.reshape(1, -1)
        prob = 1.0
        for gp in self._gp_g:
            mu, sigma = gp.predict(x, return_std=True)
            prob *= norm.cdf(-mu / (sigma[0] + 1e-8))
        return prob
    
    def _cei(self, x: np.ndarray, best_val: float) -> float:
        return self._ei(x, best_val) * self._p_feasible(x)
    
    def _next_point(self, n_tries: int = 30) -> np.ndarray:
        best_x = None
        best_val = -1e9
        current_best = self._best_value if self._best_value < 1e9 else 1e9
        for _ in range(n_tries):
            x0 = np.array([np.random.uniform(*self.bounds[i]) for i in range(self.bounds.shape[0])])
            try:
                if self.method == 'CEI':
                    res = minimize(lambda x: -self._cei(x, current_best), x0, method='L-BFGS-B', bounds=self.bounds, options={'maxiter': 100})
                else:
                    res = minimize(self._optimization_func, x0, method='L-BFGS-B', bounds=self.bounds, options={'maxiter': 100})
                if res.success and -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x
            except (ValueError, RuntimeError):
                continue
        return best_x if best_x is not None else self._lhs_sample(1)[0]
    
    def optimize(self, n_iter: int = 50) -> OptimizationResult:
        self._init_data()
        for _ in range(n_iter):
            self._train_gps()
            x_new = self._next_point()
            self._X = np.vstack([self._X, x_new])
            self._F = np.append(self._F, self._optimization_func(x_new))
            self._G = np.vstack([self._G, self.constraints(x_new)])
            if self._is_feasible(x_new) and self.objective(x_new) < self._best_value:
                self._best_value = self.objective(x_new)
                self._best_x = x_new.copy()
                if self._first_feasible is None:
                    self._first_feasible = len(self._X) - 1
            self._history.append(self._best_value if self._best_value < 1e9 else np.nan)
        return OptimizationResult(
            best_x=self._best_x,
            best_value=self._best_value,
            history=self._history,
            first_feasible=self._first_feasible,
            n_evaluations=len(self._X),
            is_feasible=self._is_feasible(self._best_x) if self._best_x is not None else False,
            method=self.method
        )
