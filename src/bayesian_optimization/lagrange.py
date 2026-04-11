"""Lagrange (Augmented Lagrangian) метод байесовской оптимизации.

Использует модифицированную функцию Лагранжа с квадратичным штрафом:
L(x, λ, ρ) = f(x) + Σ λ_i * max(0, g_i(x)) + (ρ/2) * Σ max(0, g_i(x))^2
"""

from typing import Optional

import numpy as np

from src.bayesian_optimization.base import BaseBayesianOptimization


class LagrangeBayesianOptimization(BaseBayesianOptimization):
    """Байесовская оптимизация с использованием Augmented Lagrangian.

    Метод адаптивно обновляет множители Лагранжа λ_i и штрафной коэффициент ρ.

    Атрибуты:
        constraints: Список функций ограничений g_i(x) <= 0
        multipliers: Множители Лагранжа λ_i
        penalty_coef: Штрафной коэффициент ρ
        penalty_coef_growth: Множитель роста штрафа
        constraint_tolerance: Допуск по ограничениям
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
        constraint_tolerance: float = 1e-4,
    ) -> None:
        """Инициализация Lagrange оптимизатора.

        Аргументы:
            bounds: Границы переменных [(min, max), ...]
            constraints: Список функций ограничений (g_i(x) <= 0)
            n_init: Размер начальной выборки
            kernel: Тип ядра GP ('rbf' или 'matern')
            random_state: Seed для случайных чисел
            penalty_coef_init: Начальный штрафной коэффициент
            penalty_coef_growth: Множитель роста штрафа
            constraint_tolerance: Допуск по ограничениям
        """
        super().__init__(bounds, n_init, kernel, random_state)

        self.constraints = constraints
        self.n_constraints = len(constraints)
        self.multipliers = np.zeros(self.n_constraints)
        self.penalty_coef = penalty_coef_init
        self.penalty_coef_init = penalty_coef_init
        self.penalty_coef_growth = penalty_coef_growth
        self.constraint_tolerance = constraint_tolerance

    def _compute_constraint_violations(self, X: np.ndarray) -> np.ndarray:
        """Вычисление нарушений ограничений.

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Массив нарушений (n_points, n_constraints)
        """
        X = X.reshape(-1, self.dim)
        violations = np.zeros((len(X), self.n_constraints))

        for i, x in enumerate(X):
            for j, constraint in enumerate(self.constraints):
                g_val = float(constraint(x))
                violations[i, j] = max(0, g_val)

        return violations

    def _augmented_lagrangian(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Вычисление Augmented Lagrangian функции.

        L(x) = f(x) + Σ λ_i * max(0, g_i(x)) + (ρ/2) * Σ max(0, g_i(x))^2

        Аргументы:
            X: Точки для оценки (n_points, dim)
            y: Значения исходной функции (если None, используем self.y)

        Возвращает:
            Значения Augmented Lagrangian (n_points,)
        """
        violations = self._compute_constraint_violations(X)

        if y is not None:
            lagrangian = y.copy()
        elif self.y is not None and np.array_equal(X, self.X):
            lagrangian = self.y.copy()
        else:
            lagrangian = np.zeros(len(X))

        # Добавляем вклад ограничений
        for j in range(self.n_constraints):
            v_j = violations[:, j]
            lagrangian += self.multipliers[j] * v_j + 0.5 * self.penalty_coef * (v_j ** 2)

        return lagrangian

    def _update_multipliers(self, violations: np.ndarray) -> None:
        """Обновление множителей Лагранжа.

        λ_i_new = max(0, λ_i + ρ * g_i(x))

        Аргументы:
            violations: Нарушения ограничений для текущей точки
        """
        for j in range(self.n_constraints):
            self.multipliers[j] = max(0.0, self.multipliers[j] + self.penalty_coef * violations[j])

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """EI функция приобретения для Augmented Lagrangian.

        Аргументы:
            X: Точки для оценки (n_points, dim)

        Возвращает:
            Значения EI (n_points,)
        """
        from scipy.stats import norm

        X = X.reshape(-1, self.dim)

        if self.y is None or self.X is None:
            return np.zeros(len(X))

        # Вычисляем Lagrangian для имеющихся точек
        lagrangian_values = self._augmented_lagrangian(self.X)
        f_min_lagrangian = np.min(lagrangian_values)

        # Для новых точек предсказываем Lagrangian
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        gamma = (f_min_lagrangian - mu) / sigma
        ei = (f_min_lagrangian - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)

        result = np.maximum(ei, 0)
        if result.ndim == 0:
            result = np.array([result])
        return result

    def update(self, X_new: np.ndarray, y_new: float) -> None:
        """Обновление модели с обновлением множителей Лагранжа.

        Аргументы:
            X_new: Новая точка (dim,)
            y_new: Значение функции в новой точке
        """
        # Вычисляем нарушения для новой точки
        violations = self._compute_constraint_violations(X_new.reshape(1, -1))[0]

        # Обновляем множители Лагранжа
        self._update_multipliers(violations)

        # Проверяем необходимость увеличения штрафа
        max_violation = np.max(violations)
        if max_violation > self.constraint_tolerance:
            self.penalty_coef *= self.penalty_coef_growth

        # Вычисляем значение Augmented Lagrangian
        lagrangian_value = y_new
        for j, constraint in enumerate(self.constraints):
            g_val = float(constraint(X_new))
            v_j = max(0, g_val)
            lagrangian_value += self.multipliers[j] * v_j + 0.5 * self.penalty_coef * (v_j ** 2)

        # Обновляем GP с использованием Lagrangian
        super().update(X_new, lagrangian_value)