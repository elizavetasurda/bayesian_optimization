"""CEI (Constrained Expected Improvement) метод байесовской оптимизации.

CEI учитывает ограничения через вероятность их выполнения:
CEI(x) = EI(x) * P(выполнения ограничений)
"""

from typing import Optional

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from src.bayesian_optimization.base import BaseBayesianOptimization


class CEIBayesianOptimization(BaseBayesianOptimization):
    """Байесовская оптимизация с использованием CEI (Constrained Expected Improvement).

    CEI модифицирует стандартный Expected Improvement, умножая его на
    вероятность выполнения всех ограничений.

    Атрибуты:
        constraints: Список функций ограничений g_i(x) <= 0
        n_constraints: Количество ограничений
        constraint_gps: Список GP моделей для каждого ограничения
        constraint_values: Значения ограничений для всех точек
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        constraints: list,
        n_init: int = 10,
        kernel: str = "matern",
        random_state: Optional[int] = None,
    ) -> None:
        """Инициализация CEI оптимизатора.

        Аргументы:
            bounds: Границы переменных [(min, max), ...]
            constraints: Список функций ограничений (g_i(x) <= 0)
            n_init: Размер начальной выборки
            kernel: Тип ядра GP ('rbf' или 'matern')
            random_state: Seed для случайных чисел
        """
        super().__init__(bounds, n_init, kernel, random_state)

        self.constraints = constraints
        self.n_constraints = len(constraints)

        # Инициализируем GP для каждого ограничения
        self.constraint_gps: list[GaussianProcessRegressor] = []
        self._init_constraint_gps(kernel, random_state)

        self.constraint_values: Optional[np.ndarray] = None

    def _init_constraint_gps(self, kernel: str, random_state: Optional[int]) -> None:
        """Инициализация GP моделей для ограничений.

        Аргументы:
            kernel: Тип ядра
            random_state: Seed для случайных чисел
        """
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
        """Инициализация с вычислением значений ограничений.

        Аргументы:
            X_init: Начальные точки (n_init, dim)
            y_init: Значения функции (n_init,)
        """
        super().initialize(X_init, y_init)

        # Вычисляем значения ограничений для начальных точек
        self.constraint_values = np.zeros((len(X_init), self.n_constraints))

        for i, x in enumerate(X_init):
            for j, constraint in enumerate(self.constraints):
                self.constraint_values[i, j] = float(constraint(x))

        # Обучаем GP для каждого ограничения
        for j in range(self.n_constraints):
            self.constraint_gps[j].fit(self.X, self.constraint_values[:, j])

    def _constraint_feasibility_probability(self, X: np.ndarray) -> np.ndarray:
        """Вычисление вероятности выполнения всех ограничений.

        Для каждого ограничения g_j(x) ~ N(mu, sigma^2) вычисляем:
        P(g_j(x) <= 0) = Φ(-mu / sigma)

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Вероятность выполнения всех ограничений (n_points,)
        """
        X = X.reshape(-1, self.dim)
        prob = np.ones(len(X))

        for j, gp in enumerate(self.constraint_gps):
            mu, sigma = gp.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-6)
            # P(g_j(x) <= 0)
            prob_j = norm.cdf(-mu / sigma)
            prob *= prob_j

        return prob

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Вычисление стандартного Expected Improvement.

        EI(x) = (f_min - μ) * Φ((f_min - μ)/σ) + σ * φ((f_min - μ)/σ)

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения EI (n_points,)
        """
        X = X.reshape(-1, self.dim)

        # Получаем текущий минимум
        if self.y is None:
            return np.zeros(len(X))

        f_min = np.min(self.y)

        # Предсказания GP
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        # Вычисляем EI
        gamma = (f_min - mu) / sigma
        ei = (f_min - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        return np.maximum(ei, 0)

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """CEI функция приобретения.

        CEI(x) = EI(x) * P(выполнения ограничений)

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения CEI (n_points,)
        """
        ei = self._expected_improvement(X)
        prob_feasible = self._constraint_feasibility_probability(X)

        result = ei * prob_feasible
        # Убеждаемся, что возвращается 1D массив
        if result.ndim == 0:
            result = np.array([result])
        return result

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Обновление модели новыми данными (включая ограничения).

        Аргументы:
            X_new: Новая точка (dim,)
            y_new: Значение функции в новой точке
        """
        super().update(X_new, y_new)

        # Вычисляем значения ограничений для новой точки
        if self.constraint_values is not None:
            new_constraint_values = np.zeros((1, self.n_constraints))

            for j, constraint in enumerate(self.constraints):
                new_constraint_values[0, j] = float(constraint(X_new))

            self.constraint_values = np.vstack([self.constraint_values, new_constraint_values])

            # Переобучаем GP для ограничений
            for j in range(self.n_constraints):
                self.constraint_gps[j].fit(self.X, self.constraint_values[:, j])