"""
Метод Penalty - штрафная функция
"""

import numpy as np
from scipy.stats import norm
from .base import BaseBayesianOptimization


class PenaltyBayesianOptimization(BaseBayesianOptimization):
    """
    Байесовская оптимизация с использованием штрафной функции.
    
    Используется штрафованная целевая функция:
    f_penalty(x) = f(x) + ρ * max(0, g(x))^2
    """
    
    def __init__(
        self,
        objective,
        constraint,
        bounds,
        n_init=10,
        n_iter=50,
        random_state=None,
        penalty_coeff=1e3  # Коэффициент штрафа
    ):
        super().__init__(
            objective, constraint, bounds, n_init, n_iter, random_state
        )
        self.penalty_coeff = penalty_coeff
        
    def _penalized_objective(self, x: np.ndarray) -> float:
        """
        Штрафованная целевая функция.
        """
        f_val = self.objective(x)
        g_val = self.constraint(x)
        
        # Квадратичный штраф
        penalty = self.penalty_coeff * max(0, g_val) ** 2
        
        return f_val + penalty
    
    def _initialize(self) -> None:
        """Переопределяем инициализацию для использования штрафованной функции."""
        from src.experimental_design import lhs_sample
        
        # Генерация начальной выборки
        self.X = lhs_sample(self.bounds, self.n_init, self.random_state)
        
        # Вычисление штрафованных значений
        self.y = np.array([self._penalized_objective(x) for x in self.X])
        self.c = np.array([self.constraint(x) for x in self.X])
        
        # Обучение моделей
        self._train_models()
        
        # Сохранение в историю
        self._update_history()
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """
        Стандартный Expected Improvement для штрафованной функции.
        """
        mu_f, sigma_f, _, _ = self._predict(X)
        
        # Лучшее найденное значение штрафованной функции
        f_best = np.min(self.y)
        
        # Расчет EI
        delta = f_best - mu_f
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = delta / sigma_f
            EI = delta * norm.cdf(Z) + sigma_f * norm.pdf(Z)
        
        EI[sigma_f < 1e-9] = 0
        
        return EI
    
    def optimize(self):
        """
        Оптимизация с использованием штрафованной функции.
        """
        # Запускаем оптимизацию для штрафованной функции
        result = super().optimize()
        
        # Пересчитываем результат для исходной целевой функции
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