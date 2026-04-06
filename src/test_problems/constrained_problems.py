"""Набор тестовых задач оптимизации с ограничениями."""

import numpy as np
from typing import Callable, List, Tuple, Dict


def sphere_objective(x: np.ndarray) -> float:
    """Сферическая функция."""
    return np.sum(x**2)


def sphere_constraints(x: np.ndarray) -> List[float]:
    """Ограничения для сферической функции."""
    return [np.sum(x) - 1.0]  # sum(x) <= 1


def rosenbrock_objective(x: np.ndarray) -> float:
    """Функция Розенброка."""
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def rosenbrock_constraints(x: np.ndarray) -> List[float]:
    """Ограничения для Розенброка."""
    return [x[0]**2 + x[1]**2 - 1.5]  # x0^2 + x1^2 <= 1.5


def pressure_vessel_objective(x: np.ndarray) -> float:
    """Целевая функция для задачи 'сосуд под давлением'."""
    x1, x2, x3, x4 = x
    return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3


def pressure_vessel_constraints(x: np.ndarray) -> List[float]:
    """Ограничения задачи 'сосуд под давлением'."""
    x1, x2, x3, x4 = x
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
    g4 = x4 - 240
    return [g1, g2, g3, g4]


PROBLEMS: Dict[str, dict] = {
    "sphere": {
        "objective": sphere_objective,
        "constraints": sphere_constraints,
        "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
    },
    "rosenbrock": {
        "objective": rosenbrock_objective,
        "constraints": rosenbrock_constraints,
        "bounds": [(-2.0, 2.0), (-2.0, 2.0)],
    },
    "pressure_vessel": {
        "objective": pressure_vessel_objective,
        "constraints": pressure_vessel_constraints,
        "bounds": [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 200.0)],
    },
}


def get_problem(name: str) -> dict:
    """Возвращает задачу по имени."""
    return PROBLEMS[name]