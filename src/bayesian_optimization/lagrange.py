"""
Метод Augmented Lagrangian
"""

import numpy as np
from scipy.stats import norm
from .base import BaseBayesianOptimization


class LagrangeBayesianOptimization(BaseBayesianOptimization):
    """
    Байесовская оптимизация с Augmented Lagrangian методом.
    
    Используется лагранжиан:
    L(x, λ, ρ) = f(x) + λ * g(x) + (ρ/2) * max(0, g(x))^2
    """
    
    def __init__(
        self,
        objective,
        constraint,
        bounds,
        n_init=10,
        n_iter=50,
        random_state=None,
        rho=1.0,      # Штрафной параметр
        lambda0=0.0    # Начальное значение множителя Лагранжа
    ):
        super().__init__(
            objective, constraint, bounds, n_init, n_iter, random_state
        )
        self.rho = rho
        self.lambda_ = lambda0
        
    def _lagrangian(self, x: np.ndarray) -> float:
        """
        Вычисление Augmented Lagrangian.
        """
        f_val = self.objective(x)
        g_val = self.constraint(x)
        
        # Augmented Lagrangian
        if g_val > 0:
            lag = f_val + self.lambda_ * g_val + 0.5 * self.rho * g_val ** 2
        else:
            lag = f_val + self.lambda_ * g_val
        
        return lag
    
    def _update_lagrange_multiplier(self, g_val: float) -> None:
        """
        Обновление множителя Лагранжа.
        """
        self.lambda_ = max(0, self.lambda_ + self.rho * g_val)
    
    def _initialize(self) -> None:
        """Инициализация с использованием лагранжиана."""
        from src.experimental_design import lhs_sample
        
        # Генерация начальной выборки
        self.X = lhs_sample(self.bounds, self.n_init, self.random_state)
        
        # Вычисление лагранжиана
        self.y = np.array([self._lagrangian(x) for x in self.X])
        self.c = np.array([self.constraint(x) for x in self.X])
        
        # Обновление множителя Лагранжа
        worst_g = np.max(self.c)
        if worst_g > 0:
            self._update_lagrange_multiplier(worst_g)
        
        # Обучение моделей
        self._train_models()
        
        # Сохранение в историю
        self._update_history()
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """
        Expected Improvement для лагранжиана.
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
        Оптимизация с Augmented Lagrangian.
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