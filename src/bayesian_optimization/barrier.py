"""
Метод Interior Point (Barrier)
"""

import numpy as np
from scipy.stats import norm
from .base import BaseBayesianOptimization


class BarrierBayesianOptimization(BaseBayesianOptimization):
    """
    Байесовская оптимизация с барьерной функцией.
    
    Используется логарифмический барьер:
    f_barrier(x) = f(x) - μ * log(-g(x))
    """
    
    def __init__(
        self,
        objective,
        constraint,
        bounds,
        n_init=10,
        n_iter=50,
        random_state=None,
        mu=1.0,        # Параметр барьера
        mu_reduction=0.5  # Коэффициент уменьшения μ
    ):
        super().__init__(
            objective, constraint, bounds, n_init, n_iter, random_state
        )
        self.mu = mu
        self.mu_reduction = mu_reduction
        
    def _barrier_objective(self, x: np.ndarray) -> float:
        """
        Барьерная целевая функция.
        """
        f_val = self.objective(x)
        g_val = self.constraint(x)
        
        if g_val < 0:
            # Логарифмический барьер
            barrier = -self.mu * np.log(-g_val)
            return f_val + barrier
        else:
            # Для невыполнимых точек - большое значение
            return f_val + 1e6 * g_val
    
    def _initialize(self) -> None:
        """Инициализация с барьерной функцией."""
        from src.experimental_design import lhs_sample
        
        # Генерация начальной выборки
        self.X = lhs_sample(self.bounds, self.n_init, self.random_state)
        
        # Вычисление барьерной функции
        self.y = np.array([self._barrier_objective(x) for x in self.X])
        self.c = np.array([self.constraint(x) for x in self.X])
        
        # Обучение моделей
        self._train_models()
        
        # Сохранение в историю
        self._update_history()
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """
        Expected Improvement для барьерной функции.
        """
        mu_f, sigma_f, _, _ = self._predict(X)
        
        f_best = np.min(self.y)
        
        delta = f_best - mu_f
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = delta / sigma_f
            EI = delta * norm.cdf(Z) + sigma_f * norm.pdf(Z)
        
        EI[sigma_f < 1e-9] = 0
        
        return EI
    
    def optimize(self):
        """
        Оптимизация с барьерной функцией.
        """
        result = super().optimize()
        
        # Пересчет для исходной функции
        final_X = result['history']['X'][-1]
        final_y_original = np.array([self.objective(x) for x in final_X])
        final_c = result['history']['c'][-1]
        
        feasible_mask = final_c <= 0
        if np.any(feasible_mask):
            best_idx = np.argmin(final_y_original[feasible_mask])
            result['best_solution'] = {
                'x': final_X[feasible_mask][best_idx],
                'f': final_y_original[feasible_mask][best_idx],
                'g': final_c[feasible_mask][best_idx]
            }
        
        return result