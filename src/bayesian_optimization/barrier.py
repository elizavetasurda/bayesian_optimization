"""Barrier (Interior Point) метод байесовской оптимизации.

Использует логарифмическую барьерную функцию:
F(x) = f(x) - μ * Σ log(-g_i(x))
"""

from typing import Optional

import numpy as np

from src.bayesian_optimization.base import BaseBayesianOptimization


class BarrierBayesianOptimization(BaseBayesianOptimization):
    """Байесовская оптимизация с использованием барьерного метода.

    Метод подходит только для задач со строго выполнимыми ограничениями
    (существует внутренняя точка, где все g_i(x) < 0).

    Атрибуты:
        constraints: Список функций ограничений g_i(x) <= 0
        barrier_mu: Параметр барьера μ
        mu_reduction_factor: Множитель уменьшения μ
        barrier_tolerance: Допуск по барьеру
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        constraints: list,
        n_init: int = 10,
        kernel: str = "matern",
        random_state: Optional[int] = None,
        barrier_mu_init: float = 1.0,
        mu_reduction_factor: float = 0.5,
        barrier_tolerance: float = 1e-8,
    ) -> None:
        """Инициализация Barrier оптимизатора.

        Аргументы:
            bounds: Границы переменных [(min, max), ...]
            constraints: Список функций ограничений (g_i(x) <= 0)
            n_init: Размер начальной выборки
            kernel: Тип ядра GP ('rbf' или 'matern')
            random_state: Seed для случайных чисел
            barrier_mu_init: Начальный параметр барьера μ
            mu_reduction_factor: Множитель уменьшения μ
            barrier_tolerance: Допуск по барьеру
        """
        super().__init__(bounds, n_init, kernel, random_state)

        self.constraints = constraints
        self.n_constraints = len(constraints)
        self.barrier_mu = barrier_mu_init
        self.barrier_mu_init = barrier_mu_init
        self.mu_reduction_factor = mu_reduction_factor
        self.barrier_tolerance = barrier_tolerance

    def _compute_barrier(self, X: np.ndarray) -> np.ndarray:
        """Вычисление барьерного члена.

        B(x) = -μ * Σ log(-g_i(x))

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения барьера (n_points,)
        """
        X = X.reshape(-1, self.dim)
        barrier = np.zeros(len(X))

        for i, x in enumerate(X):
            log_sum = 0.0
            feasible = True

            for constraint in self.constraints:
                g_val = float(constraint(x))

                if g_val >= 0:
                    feasible = False
                    break

                log_sum += np.log(-g_val)

            if feasible:
                barrier[i] = -self.barrier_mu * log_sum
            else:
                barrier[i] = np.inf

        return barrier

    def _barrier_objective(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Вычисление барьерной целевой функции.

        F(x) = f(x) - μ * Σ log(-g_i(x))

        Аргументы:
            X: Точки для оценки (n_points, dim)
            y: Значения исходной функции (если None, используем self.y)

        Возвращает:
            Значения барьерной функции (n_points,)
        """
        barrier = self._compute_barrier(X)

        if y is not None:
            return y + barrier

        if self.y is not None and np.array_equal(X, self.X):
            return self.y + barrier

        return barrier

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """EI функция приобретения для барьерной задачи.

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения EI (n_points,)
        """
        from scipy.stats import norm

        X = X.reshape(-1, self.dim)

        if self.y is None or self.X is None:
            return np.zeros(len(X))

        # Вычисляем барьерные значения для имеющихся точек
        barrier_values = self._barrier_objective(self.X)
        barrier_values = np.where(np.isfinite(barrier_values), barrier_values, np.inf)

        if np.all(np.isinf(barrier_values)):
            return np.zeros(len(X))

        f_min_barrier = np.min(barrier_values[barrier_values < np.inf])

        # Предсказания GP
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        gamma = (f_min_barrier - mu) / sigma
        ei = (f_min_barrier - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        result = np.maximum(ei, 0)
        if result.ndim == 0:
            result = np.array([result])
        return result

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Обновление модели с возможным уменьшением μ.

        Аргументы:
            X_new: Новая точка (dim,)
            y_new: Значение функции в новой точке
        """
        # Вычисляем барьерное значение
        barrier_value = self._compute_barrier(X_new.reshape(1, -1))[0]

        if np.isinf(barrier_value):
            # Точка недопустима - не добавляем её в модель
            return

        barrier_y = y_new + barrier_value

        # Обновляем GP
        super().update(X_new, barrier_y)

        # Уменьшаем μ каждые 5 итераций
        if self.X is not None and len(self.X) % 5 == 0:
            self.barrier_mu *= self.mu_reduction_factor
            self._update_gp()