"""Модуль для запуска экспериментов по байесовской оптимизации.

Содержит функции для проведения полных экспериментов:
- Многократные прогоны (trials) для усреднения результатов
- Сравнение различных методов (CEI, Penalty, Lagrange, Barrier)
- Сбор статистики по времени и сходимости
"""

import time
from typing import Any

import numpy as np

from src.bayesian_optimization import (
    CEIBayesianOptimization,
    PenaltyBayesianOptimization,
    LagrangeBayesianOptimization,
    BarrierBayesianOptimization,
)
from src.experimental_design.lhs import lhs_initialize
from src.utils.types import OptimizationProblem, TrialResult


def evaluate_point(
    x: np.ndarray,
    problem: OptimizationProblem,
) -> tuple[float, list[float]]:
    """
    Вычисление целевой функции и ограничений в точке.

    Аргументы:
        x: Точка для оценки
        problem: Задача оптимизации

    Возвращает:
        Кортеж (значение функции, список значений ограничений)
    """
    obj_value = problem.objective(x)
    constraint_values = [c(x) for c in problem.constraints]
    return obj_value, constraint_values


def is_feasible(
    constraint_values: list[float],
    tolerance: float = 1e-6,
) -> bool:
    """
    Проверка выполнения всех ограничений.

    Аргументы:
        constraint_values: Список значений ограничений g_i(x)
        tolerance: Допуск (g_i(x) <= tolerance)

    Возвращает:
        True если все ограничения выполнены
    """
    return all(g <= tolerance for g in constraint_values)


def run_single_trial(
    problem: OptimizationProblem,
    method_name: str,
    n_init: int,
    n_iter: int,
    random_state: int,
) -> TrialResult:
    """
    Запуск одного прогона (trial) оптимизации.

    Аргументы:
        problem: Задача оптимизации
        method_name: Название метода ('CEI', 'Penalty', 'Lagrange', 'Barrier')
        n_init: Размер начальной выборки
        n_iter: Количество итераций оптимизации
        random_state: Seed для воспроизводимости

    Возвращает:
        TrialResult с результатами прогона
    """
    np.random.seed(random_state)

    # Создаём начальную выборку методом LHS
    X_init = lhs_initialize(problem.bounds, n_init, random_state)

    # Вычисляем значения функции и ограничений для начальных точек
    y_init = []
    constraint_values_list = []

    for x in X_init:
        y_val, c_vals = evaluate_point(x, problem)
        y_init.append(y_val)
        constraint_values_list.append(c_vals)

    y_init = np.array(y_init)

    # Выбираем и инициализируем метод
    if method_name == "CEI":
        optimizer = CEIBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
        )
    elif method_name == "Penalty":
        optimizer = PenaltyBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
            penalty_coef_init=1.0,
            penalty_coef_growth=2.0,
        )
    elif method_name == "Lagrange":
        optimizer = LagrangeBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
            penalty_coef_init=1.0,
            penalty_coef_growth=2.0,
        )
    elif method_name == "Barrier":
        optimizer = BarrierBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
            barrier_mu_init=1.0,
            mu_reduction_factor=0.5,
        )
    else:
        raise ValueError(f"Неизвестный метод: {method_name}")

    # Инициализируем оптимизатор
    optimizer.initialize(X_init, y_init)

    # История лучших значений
    best_values_history = []

    # Находим лучшее начальное значение
    best_value = np.min(y_init)
    best_values_history.append(best_value)

    start_time = time.time()

    # Основной цикл оптимизации
    for _ in range(n_iter):
        # Предлагаем следующую точку
        x_next = optimizer.suggest_next_point()

        # Вычисляем значение функции
        y_next, _ = evaluate_point(x_next, problem)

        # Обновляем оптимизатор
        optimizer.update(x_next, y_next)

        # Обновляем лучшее значение
        current_best = optimizer.get_best_point()[1]
        if current_best is not None:
            best_value = min(best_value, current_best)
        best_values_history.append(best_value)

    elapsed_time = time.time() - start_time

    # Получаем финальную лучшую точку
    final_point, final_value = optimizer.get_best_point()
    if final_point is None:
        final_point = X_init[0]
        final_value = y_init[0]

    return TrialResult(
        best_values=best_values_history,
        time=elapsed_time,
        final_point=final_point,
        feasibility=is_feasible([c(final_point) for c in problem.constraints]),
    )


def run_experiments(
    problems: list[OptimizationProblem],
    n_trials: int,
    n_init: int,
    n_iter: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Запуск полного эксперимента на наборе задач.

    Аргументы:
        problems: Список задач для тестирования
        n_trials: Количество повторений для усреднения
        n_init: Размер начальной выборки
        n_iter: Количество итераций оптимизации

    Возвращает:
        Словарь результатов:
        {
            "problem_name": {
                "CEI": {"best_values": np.ndarray, "times": np.ndarray, ...},
                "Penalty": {...},
                ...
            }
        }
    """
    methods = ["CEI", "Penalty", "Lagrange", "Barrier"]
    all_results = {}

    for prob_idx, problem in enumerate(problems):
        print(f"\n Обработка задачи {prob_idx + 1}/{len(problems)}: {problem.name}")

        problem_results = {}

        for method in methods:
            print(f"   Метод: {method}...", end=" ", flush=True)

            # Запускаем многократные прогоны
            all_best_values = []
            all_times = []
            all_final_points = []

            for trial in range(n_trials):
                random_state = 42 + prob_idx * 100 + trial * 10

                trial_result = run_single_trial(
                    problem=problem,
                    method_name=method,
                    n_init=n_init,
                    n_iter=n_iter,
                    random_state=random_state,
                )

                all_best_values.append(trial_result.best_values)
                all_times.append(trial_result.time)
                all_final_points.append(trial_result.final_point)

            # Усредняем результаты по прогонам
            best_values_matrix = np.array(all_best_values)  # (n_trials, n_iter+1)

            problem_results[method] = {
                "best_values": best_values_matrix,
                "times": np.array(all_times),
                "final_points": all_final_points,
                "mean_best": np.mean(best_values_matrix, axis=0),
                "std_best": np.std(best_values_matrix, axis=0),
                "median_best": np.median(best_values_matrix, axis=0),
                "mean_time": np.mean(all_times),
                "std_time": np.std(all_times),
            }

            print(f"готово (среднее время: {np.mean(all_times):.2f}с)")

        all_results[problem.name] = problem_results

    return all_results