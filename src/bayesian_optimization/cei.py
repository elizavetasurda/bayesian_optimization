"""
Метод Constrained Expected Improvement (CEI)
"""

import numpy as np
from scipy.stats import norm
from .base import BaseBayesianOptimization


class CEIBayesianOptimization(BaseBayesianOptimization):
    """
    Байесовская оптимизация с использованием Constrained Expected Improvement.
    
    CEI = EI(x) * P(feasible)
    где EI(x) - ожидаемое улучшение,
    P(feasible) - вероятность выполнимости ограничения.
    """
    
    def __init__(
        self,
        objective,
        constraint,
        bounds,
        n_init=10,
        n_iter=50,
        random_state=None
    ):
        super().__init__(
            objective, constraint, bounds, n_init, n_iter, random_state
        )
        
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление Constrained Expected Improvement.
        
        CEI(x) = EI(x) * P(g(x) <= 0)
        """
        mu_f, sigma_f, mu_g, sigma_g = self._predict(X)
        
        # Находим лучшее выполнимое решение
        feasible_mask = self.c <= 0
        if np.any(feasible_mask):
            f_best = np.min(self.y[feasible_mask])
        else:
            # Если нет выполнимых решений, используем минимум по всем
            f_best = np.min(self.y)
        
        # Расчет Expected Improvement
        delta = f_best - mu_f
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = delta / sigma_f
            EI = delta * norm.cdf(Z) + sigma_f * norm.pdf(Z)
        
        # Обнуляем там, где sigma_f близко к нулю
        EI[sigma_f < 1e-9] = 0
        
        # Расчет вероятности выполнимости
        P_feasible = norm.cdf(0, loc=mu_g, scale=sigma_g)
        
        # CEI = EI * P(feasible)
        cei = EI * P_feasible
        
        return cei