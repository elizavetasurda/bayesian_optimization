"""Основной скрипт для запуска байесовской оптимизации с ограничениями.

Запускает все методы (CEI, Penalty, Lagrange, Barrier) на тестовых задачах,
сохраняет результаты и строит графики сходимости.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.bayesian_optimization.cei import CEIBayesianOptimization
from src.bayesian_optimization.penalty import PenaltyBayesianOptimization
from src.bayesian_optimization.lagrange import LagrangeBayesianOptimization
from src.bayesian_optimization.barrier import BarrierBayesianOptimization
from src.test_problems.constrained_problems import get_problem
from src.utils.result_saver import save_results, Result


def run_optimization(
    problem_name: str,
    method_name: str,
    n_iterations: int = 50,
    n_initial: int = 10,
) -> Result:
    """Запускает оптимизацию заданным методом на заданной задаче.

    Args:
        problem_name: Имя тестовой задачи ('sphere', 'rosenbrock', 'pressure_vessel')
        method_name: Имя метода ('CEI', 'Penalty', 'Lagrange', 'Barrier')
        n_iterations: Количество итераций оптимизации
        n_initial: Размер начальной случайной выборки

    Returns:
        Result: Объект с результатами оптимизации
    """
    problem = get_problem(problem_name)

    if method_name == "CEI":
        optimizer = CEIBayesianOptimization(
            objective=problem.objective,
            constraints=problem.constraints,
            bounds=problem.bounds,
            n_initial=n_initial,
        )
    elif method_name == "Penalty":
        optimizer = PenaltyBayesianOptimization(
            objective=problem.objective,
            constraints=problem.constraints,
            bounds=problem.bounds,
            n_initial=n_initial,
        )
    elif method_name == "Lagrange":
        optimizer = LagrangeBayesianOptimization(
            objective=problem.objective,
            constraints=problem.constraints,
            bounds=problem.bounds,
            n_initial=n_initial,
        )
    elif method_name == "Barrier":
        optimizer = BarrierBayesianOptimization(
            objective=problem.objective,
            constraints=problem.constraints,
            bounds=problem.bounds,
            n_initial=n_initial,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    best_values = []
    for _ in range(n_iterations):
        optimizer.iterate()
        best_values.append(optimizer.best_feasible_value)

    return Result(
        method=method_name,
        problem=problem_name,
        best_value=optimizer.best_feasible_value,
        best_point=optimizer.best_feasible_point,
        convergence=np.array(best_values),
    )


def main() -> None:
    """Главная функция: запускает все методы на всех задачах."""
    problems = ["sphere", "rosenbrock", "pressure_vessel"]
    methods = ["CEI", "Penalty", "Lagrange", "Barrier"]
    results: list[Result] = []

    for problem in problems:
        for method in methods:
            print(f"Running {method} on {problem}...")
            res = run_optimization(problem, method)
            results.append(res)

    save_results(results, "results/optimization_results.txt")

    # Построение графиков
    for problem in problems:
        plt.figure(figsize=(10, 6))
        for method in methods:
            res = next(r for r in results if r.problem == problem and r.method == method)
            plt.plot(res.convergence, label=method)
        plt.axvline(x=10, color='gray', linestyle='--', label='Начальная выборка (10)')
        plt.xlabel("Итерация")
        plt.ylabel("Лучшее допустимое значение")
        plt.title(f"Сходимость на {problem}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/convergence_{problem}.png")
        plt.close()


if __name__ == "__main__":
    main()