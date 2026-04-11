"""Модуль для латинского гиперкубического планирования эксперимента (LHS).

LHS обеспечивает равномерное покрытие пространства параметров
при ограниченном количестве начальных точек.
"""

import numpy as np


def latin_hypercube_sample(
    bounds: list[tuple[float, float]],
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Генерация выборки методом латинского гиперкуба.

    Алгоритм:
    1. Каждый диапазон переменной делится на n_samples интервалов
    2. В каждом интервале выбирается случайная точка
    3. Порядок интервалов случайно перемешивается для каждой переменной

    Аргументы:
        bounds: Список границ для каждой переменной [(min, max), ...]
        n_samples: Количество точек выборки
        random_state: Seed для генератора случайных чисел

    Возвращает:
        Массив точек размера (n_samples, n_dim)

    Пример:
        >>> bounds = [(0, 1), (0, 1)]
        >>> samples = latin_hypercube_sample(bounds, 10)
        >>> samples.shape
        (10, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_dim = len(bounds)
    samples = np.zeros((n_samples, n_dim))

    for i, (low, high) in enumerate(bounds):
        # Делим диапазон на n_samples равных интервалов
        edges = np.linspace(low, high, n_samples + 1)

        # Выбираем случайную точку в каждом интервале
        points = np.random.uniform(edges[:-1], edges[1:])

        # Перемешиваем для случайного порядка
        samples[:, i] = np.random.permutation(points)

    return samples


def random_sample(
    bounds: list[tuple[float, float]],
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Генерация случайной выборки (равномерное распределение).

    Аргументы:
        bounds: Список границ для каждой переменной [(min, max), ...]
        n_samples: Количество точек выборки
        random_state: Seed для генератора случайных чисел

    Возвращает:
        Массив точек размера (n_samples, n_dim)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_dim = len(bounds)
    samples = np.zeros((n_samples, n_dim))

    for i, (low, high) in enumerate(bounds):
        samples[:, i] = np.random.uniform(low, high, n_samples)

    return samples


def lhs_initialize(
    problem_bounds: list[tuple[float, float]],
    n_points: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Удобная обёртка для LHS-инициализации оптимизации.

    Аргументы:
        problem_bounds: Границы задачи [(min, max), ...]
        n_points: Количество начальных точек
        random_state: Seed для случайных чисел

    Возвращает:
        Массив начальных точек размера (n_points, n_dim)
    """
    return latin_hypercube_sample(problem_bounds, n_points, random_state)