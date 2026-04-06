"""
Модуль планирования эксперимента (Design of Experiments)
Содержит чистые функции для генерации начальных выборок
"""

import numpy as np
from typing import Optional, Tuple


def lhs_sample(
    bounds: np.ndarray, n_samples: int, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Генерация выборки методом латинского гиперкуба (LHS).

    Parameters
    ----------
    bounds : np.ndarray
        Массив размера (n_dims, 2) с границами [lower, upper] для каждого измерения
    n_samples : int
        Количество точек в выборке
    random_state : Optional[int]
        Seed для генератора случайных чисел

    Returns
    -------
    np.ndarray
        Массив размера (n_samples, n_dims) с точками выборки
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_dims = bounds.shape[0]
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_dims):
        # Разбиваем интервал на n_samples равных отрезков
        intervals = np.linspace(bounds[i, 0], bounds[i, 1], n_samples + 1)

        # Выбираем случайную точку в каждом интервале
        for j in range(n_samples):
            samples[j, i] = np.random.uniform(intervals[j], intervals[j + 1])

        # Перемешиваем порядок для каждого измерения
        np.random.shuffle(samples[:, i])

    return samples


def uniform_sample(
    bounds: np.ndarray, n_samples: int, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Генерация равномерной случайной выборки.

    Parameters
    ----------
    bounds : np.ndarray
        Массив размера (n_dims, 2) с границами [lower, upper]
    n_samples : int
        Количество точек
    random_state : Optional[int]
        Seed для генератора

    Returns
    -------
    np.ndarray
        Массив размера (n_samples, n_dims) с точками
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_dims = bounds.shape[0]
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_dims):
        samples[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_samples)

    return samples


def grid_sample(bounds: np.ndarray, points_per_dim: int) -> np.ndarray:
    """
    Генерация равномерной сеточной выборки.

    Parameters
    ----------
    bounds : np.ndarray
        Массив размера (n_dims, 2) с границами
    points_per_dim : int
        Количество точек по каждому измерению

    Returns
    -------
    np.ndarray
        Массив размера (points_per_dim^n_dims, n_dims) с точками сетки
    """
    n_dims = bounds.shape[0]
    grids = []

    for i in range(n_dims):
        grids.append(np.linspace(bounds[i, 0], bounds[i, 1], points_per_dim))

    mesh = np.meshgrid(*grids, indexing="ij")
    samples = np.column_stack([m.ravel() for m in mesh])

    return samples
