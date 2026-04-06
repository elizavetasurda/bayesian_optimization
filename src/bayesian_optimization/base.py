"""Базовый класс для всех методов байесовской оптимизации с ограничениями."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, List, Tuple


class BayesianOptimizationBase(ABC):
    """Абстрактный базовый класс оптимизатора."""

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        constraints: List[Callable[[np.ndarray], float]],
        bounds: List[Tuple[float, float]],
        n_initial: int = 10,
    ) -> None:
        """Инициализация оптимизатора.

        Args:
            objective: Целевая функция f(x)
            constraints: Список функций ограничений g_i(x) <= 0
            bounds: Список кортежей (min, max) для каждой переменной
            n_initial: Размер начальной выборки
        """
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
        self.n_initial = n_initial
        self.dim = len(bounds)

        self.X: List[np.ndarray] = []
        self.F: List[float] = []
        self.best_feasible_point: np.ndarray | None = None
        self.best_feasible_value: float = np.inf

    @abstractmethod
    def iterate(self) -> None:
        """Выполняет одну итерацию оптимизации."""
        pass

    def _is_feasible(self, x: np.ndarray) -> bool:
        """Проверяет, удовлетворяет ли точка всем ограничениям.

        Args:
            x: Точка для проверки

        Returns:
            True, если все ограничения выполнены, иначе False
        """
        for constraint in self.constraints:
            if constraint(x) > 1e-6:  # допуск
                return False
        return True
