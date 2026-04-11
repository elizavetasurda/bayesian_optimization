"""Penalty метод байесовской оптимизации.

Преобразует ограниченную задачу в безусловную через штрафную функцию:
F(x) = f(x) + ρ * Σ max(0, g_i(x))^2
"""

from typing import Optional

import numpy as np

from src.bayesian_optimization.base import BaseBayesianOptimization


class PenaltyBayesianOptimization(BaseBayesianOptimization):
    """Байесовская оптимизация с использованием штрафного метода.

    Штрафная функция: F(x) = f(x) + ρ * Σ max(0, g_i(x))^2
    Коэффициент штрафа ρ адаптивно увеличивается при нарушении ограничений.

    Атрибуты:
        constraints: Список функций ограничений g_i(x) <= 0
        penalty_coef: Текущий коэффициент штрафа ρ
        penalty_coef_init: Начальный коэффициент штрафа
        penalty_coef_growth: Множитель роста штрафа
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        constraints: list,
        n_init: int = 10,
        kernel: str = "matern",
        random_state: Optional[int] = None,
        penalty_coef_init: float = 1.0,
        penalty_coef_growth: float = 2.0,
    ) -> None:
        """Инициализация Penalty оптимизатора.

        Аргументы:
            bounds: Границы переменных [(min, max), ...]
            constraints: Список функций ограничений (g_i(x) <= 0)
            n_init: Размер начальной выборки
            kernel: Тип ядра GP ('rbf' или 'matern')
            random_state: Seed для случайных чисел
            penalty_coef_init: Начальный коэффициент штрафа
            penalty_coef_growth: Множитель роста штрафа
        """
        super().__init__(bounds, n_init, kernel, random_state)

        self.constraints = constraints
        self.penalty_coef = penalty_coef_init
        self.penalty_coef_init = penalty_coef_init
        self.penalty_coef_growth = penalty_coef_growth

    def _compute_penalty(self, X: np.ndarray) -> np.ndarray:
        """Вычисление штрафа для точек.

        Штраф = Σ max(0, g_i(x))^2

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения штрафа (n_points,)
        """
        X = X.reshape(-1, self.dim)
        penalty = np.zeros(len(X))

        for i, x in enumerate(X):
            violation_sum = 0.0
            for constraint in self.constraints:
                g_val = float(constraint(x))
                violation_sum += max(0, g_val) ** 2
            penalty[i] = violation_sum

        return penalty

    def _penalized_objective(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Вычисление штрафной целевой функции.

        F(x) = f(x) + ρ * штраф(x)

        Аргументы:
            X: Точки для оценки (n_points, dim)
            y: Значения исходной функции (если None, используем self.y)

        Возвращает:
            Значения штрафной функции (n_points,)
        """
        penalty = self._compute_penalty(X)

        if y is not None:
            return y + self.penalty_coef * penalty

        if self.y is not None and np.array_equal(X, self.X):
            return self.y + self.penalty_coef * penalty

        return penalty

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """EI функция приобретения для штрафной задачи.

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения EI (n_points,)
        """
        from scipy.stats import norm

        X = X.reshape(-1, self.dim)

        if self.y is None:
            return np.zeros(len(X))

        # Получаем штрафные значения для всех точек
        penalized_y = self._penalized_objective(self.X, self.y)
        f_min_penalized = np.min(penalized_y)

        # Предсказания GP для штрафной функции
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        gamma = (f_min_penalized - mu) / sigma
        ei = (f_min_penalized - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        result = np.maximum(ei, 0)
        if result.ndim == 0:
            result = np.array([result])
        return result

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Обновление модели с адаптацией коэффициента штрафа.

        Аргументы:
            X_new: Новая точка (dim,)
            y_new: Значение функции в новой точке
        """
        # Проверяем выполнение ограничений
        is_feasible = True
        for constraint in self.constraints:
            if float(constraint(X_new)) > 0:
                is_feasible = False
                break

        # Адаптивно увеличиваем штраф при нарушении ограничений
        if not is_feasible:
            self.penalty_coef *= self.penalty_coef_growth

        # Вычисляем штрафное значение
        penalty = self._compute_penalty(X_new.reshape(1, -1))[0]
        penalized_y = y_new + self.penalty_coef * penalty

        # Обновляем GP с использованием штрафного значения
        super().update(X_new, penalized_y)