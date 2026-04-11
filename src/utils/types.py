"""Определение типов данных для эксперимента."""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Any

import numpy as np


@dataclass
class OptimizationProblem:
    """Класс для хранения задачи оптимизации с ограничениями."""

    name: str
    dim: int
    bounds: List[Tuple[float, float]]
    objective: Callable[[np.ndarray], float]
    constraints: List[Callable[[np.ndarray], float]]
    optimal_value: float
    optimal_point: np.ndarray
    constraint_tolerance: float = 1e-6


@dataclass
class ExperimentResult:
    """Результаты эксперимента для одного метода и задачи."""

    method_name: str
    problem_name: str
    best_values: np.ndarray          # форма (n_trials, n_iter)
    times: np.ndarray                # форма (n_trials,)
    final_points: List[np.ndarray]
    convergence_rates: np.ndarray


@dataclass
class TrialResult:
    """Результаты одного прогона (одного повторения эксперимента)."""

    best_values: List[float]         # лучшие значения по итерациям
    time: float                      # время выполнения прогона
    final_point: np.ndarray          # финальная точка
    feasibility: bool = False        # выполнены ли ограничения


@dataclass
class MethodResults:
    """Результаты одного метода на одной задаче (многократные прогоны)."""

    method_name: str
    all_trials: List[TrialResult] = field(default_factory=list)

    @property
    def best_values_matrix(self) -> np.ndarray:
        """Матрица лучших значений: (n_trials, n_iterations)."""
        if not self.all_trials:
            return np.array([])
        n_iter = len(self.all_trials[0].best_values)
        matrix = np.array([t.best_values for t in self.all_trials])
        return matrix

    @property
    def mean_best_values(self) -> np.ndarray:
        """Среднее по прогонам лучших значений."""
        return np.mean(self.best_values_matrix, axis=0)

    @property
    def std_best_values(self) -> np.ndarray:
        """Стандартное отклонение лучших значений."""
        return np.std(self.best_values_matrix, axis=0)
