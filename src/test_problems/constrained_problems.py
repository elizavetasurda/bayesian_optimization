"""Набор тестовых задач оптимизации с ограничениями."""

import numpy as np
from typing import Callable, List, Tuple, Dict


def sphere_objective(x: np.ndarray) -> float:
    """Сферическая функция."""
    return np.sum(x**2)


# Каждое ограничение — отдельная функция
def sphere_constraint_1(x: np.ndarray) -> float:
    """Ограничение: sum(x) <= 1"""
    return np.sum(x) - 1.0


def rosenbrock_objective(x: np.ndarray) -> float:
    """Функция Розенброка."""
    return sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)
    )


def rosenbrock_constraint_1(x: np.ndarray) -> float:
    """Ограничение: x0^2 + x1^2 <= 1.5"""
    return x[0] ** 2 + x[1] ** 2 - 1.5


def pressure_vessel_objective(x: np.ndarray) -> float:
    """Целевая функция для задачи 'сосуд под давлением'."""
    x1, x2, x3, x4 = x
    return (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3**2
        + 3.1661 * x1**2 * x4
        + 19.84 * x1**2 * x3
    )


# Отдельные функции для каждого ограничения
def pressure_vessel_constraint_1(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return -x1 + 0.0193 * x3


def pressure_vessel_constraint_2(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return -x2 + 0.00954 * x3


def pressure_vessel_constraint_3(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return -np.pi * x3**2 * x4 - (4 / 3) * np.pi * x3**3 + 1296000


def pressure_vessel_constraint_4(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return x4 - 240


PROBLEMS: Dict[str, dict] = {
    "sphere": {
        "objective": sphere_objective,
        "constraints": [sphere_constraint_1],  # СПИСОК ФУНКЦИЙ
        "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
    },
    "rosenbrock": {
        "objective": rosenbrock_objective,
        "constraints": [rosenbrock_constraint_1],  # СПИСОК ФУНКЦИЙ
        "bounds": [(-2.0, 2.0), (-2.0, 2.0)],
    },
    "pressure_vessel": {
        "objective": pressure_vessel_objective,
        "constraints": [
            pressure_vessel_constraint_1,
            pressure_vessel_constraint_2,
            pressure_vessel_constraint_3,
            pressure_vessel_constraint_4,
        ],  # СПИСОК ФУНКЦИЙ
        "bounds": [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 200.0)],
    },
}


def get_problem(name: str) -> dict:
    """Возвращает задачу по имени."""
    return PROBLEMS[name]
