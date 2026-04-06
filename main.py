"""Основной скрипт для запуска байесовской оптимизации с ограничениями."""

import warnings

warnings.filterwarnings("ignore")

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
    seed: int = 42,
    verbose: bool = True,
) -> Result:
    """Запускает оптимизацию заданным методом на заданной задаче.

    Args:
        problem_name: Имя задачи ('sphere', 'rosenbrock', 'pressure_vessel')
        method_name: Имя метода ('CEI', 'Penalty', 'Lagrange', 'Barrier')
        n_iterations: Количество итераций
        n_initial: Размер начальной выборки
        seed: Seed для воспроизводимости
        verbose: Выводить ли прогресс каждой итерации

    Returns:
        Result: Объект с результатами
    """
    np.random.seed(seed)
    problem = get_problem(problem_name)

    if method_name == "CEI":
        optimizer = CEIBayesianOptimization(
            objective=problem["objective"],
            constraints=problem["constraints"],
            bounds=problem["bounds"],
            n_initial=n_initial,
        )
    elif method_name == "Penalty":
        optimizer = PenaltyBayesianOptimization(
            objective=problem["objective"],
            constraints=problem["constraints"],
            bounds=problem["bounds"],
            n_initial=n_initial,
        )
    elif method_name == "Lagrange":
        optimizer = LagrangeBayesianOptimization(
            objective=problem["objective"],
            constraints=problem["constraints"],
            bounds=problem["bounds"],
            n_initial=n_initial,
        )
    elif method_name == "Barrier":
        optimizer = BarrierBayesianOptimization(
            objective=problem["objective"],
            constraints=problem["constraints"],
            bounds=problem["bounds"],
            n_initial=n_initial,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    best_values = []
    
    if verbose:
        print(f"    Начальная выборка: {n_initial} точек")
    
    for i in range(n_iterations):
        optimizer.iterate()
        current_best = optimizer.best_feasible_value
        best_values.append(current_best)
        
        if verbose:
            # Выводим каждые 5 итераций или если улучшение
            if (i + 1) % 5 == 0 or i == 0 or i == n_iterations - 1:
                print(f"    Итерация {i+1:3d}/{n_iterations}: best = {current_best:.8f}")

    return Result(
        method=method_name,
        problem=problem_name,
        best_value=optimizer.best_feasible_value,
        best_point=optimizer.best_feasible_point,
        convergence=np.array(best_values),
        seed=seed,
    )


def run_with_seeds(
    problem_name: str,
    method_name: str,
    n_iterations: int = 50,
    n_initial: int = 10,
    n_seeds: int = 5,
    verbose: bool = True,
) -> dict:
    """Запускает оптимизацию несколько раз с разными seed и усредняет результаты.

    Args:
        problem_name: Имя задачи
        method_name: Имя метода
        n_iterations: Количество итераций
        n_initial: Размер начальной выборки
        n_seeds: Количество случайных запусков
        verbose: Выводить ли прогресс

    Returns:
        dict: Усреднённые результаты
    """
    all_results = []
    for seed in range(n_seeds):
        if verbose:
            print(f"    Seed {seed+1}/{n_seeds}...")
        res = run_optimization(
            problem_name=problem_name,
            method_name=method_name,
            n_iterations=n_iterations,
            n_initial=n_initial,
            seed=seed,
            verbose=verbose,
        )
        all_results.append(res)

    convergences = np.array([r.convergence for r in all_results])
    mean_convergence = np.mean(convergences, axis=0)
    std_convergence = np.std(convergences, axis=0)
    mean_best = np.mean([r.best_value for r in all_results])
    std_best = np.std([r.best_value for r in all_results])

    return {
        "method": method_name,
        "problem": problem_name,
        "mean_best": mean_best,
        "std_best": std_best,
        "mean_convergence": mean_convergence,
        "std_convergence": std_convergence,
    }


def main() -> None:
    """Главная функция: запускает все методы на всех задачах с разными размерностями."""
    problems = ["sphere", "rosenbrock"]
    methods = ["CEI", "Penalty", "Lagrange", "Barrier"]
    dimensions = [2, 4, 8]
    n_iterations = 50
    n_initial = 10
    n_seeds = 3  # для быстрого теста, потом увеличить до 10-20

    all_summaries = []

    for problem in problems:
        for dim in dimensions:
            # Получаем задачу с нужной размерностью
            base_problem = get_problem(problem)
            # Изменяем bounds для нужной размерности
            if problem == "sphere":
                bounds = [(-5.0, 5.0)] * dim
            elif problem == "rosenbrock":
                bounds = [(-2.0, 2.0)] * dim
            else:
                bounds = base_problem["bounds"]

            # Временно подменяем bounds
            base_problem["bounds"] = bounds

            for method in methods:
                print(f"\n{'='*70}")
                print(f"▶ Запуск: {method} на {problem} (размерность={dim})")
                print(f"{'='*70}")

                summary = run_with_seeds(
                    problem_name=problem,
                    method_name=method,
                    n_iterations=n_iterations,
                    n_initial=n_initial,
                    n_seeds=n_seeds,
                    verbose=True,
                )
                summary["dimension"] = dim
                all_summaries.append(summary)

                print(f"   Результат: {summary['mean_best']:.8f} ± {summary['std_best']:.8f}")
                print()

    # Сохраняем сводку
    import os
    os.makedirs("results", exist_ok=True)
    
    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write("Результаты экспериментов\n")
        f.write("=" * 70 + "\n\n")
        for s in all_summaries:
            f.write(
                f"{s['method']} on {s['problem']} (dim={s['dimension']}):\n"
                f"  Среднее: {s['mean_best']:.8f}\n"
                f"  Стандартное отклонение: {s['std_best']:.8f}\n\n"
            )

    # Построение графиков для каждой размерности
    print("\n" + "=" * 70)
    print(" Построение графиков...")
    print("=" * 70)
    
    for problem in problems:
        for dim in dimensions:
            plt.figure(figsize=(10, 6))
            for method in methods:
                data = next(
                    (s for s in all_summaries 
                     if s["problem"] == problem and s["method"] == method and s["dimension"] == dim),
                    None,
                )
                if data is not None:
                    iterations = range(len(data["mean_convergence"]))
                    plt.plot(iterations, data["mean_convergence"], label=method, linewidth=2)
                    plt.fill_between(
                        iterations,
                        data["mean_convergence"] - data["std_convergence"],
                        data["mean_convergence"] + data["std_convergence"],
                        alpha=0.2,
                    )
            
            plt.axvline(x=n_initial, color="gray", linestyle="--", linewidth=1.5, 
                       label=f"Начальная выборка ({n_initial})")
            plt.xlabel("Итерация", fontsize=12)
            plt.ylabel("Лучшее допустимое значение (среднее)", fontsize=12)
            plt.title(f"Сходимость методов на {problem} (размерность={dim})", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"results/convergence_{problem}_dim{dim}.png", dpi=150)
            plt.close()
            print(f"   Сохранён график: results/convergence_{problem}_dim{dim}.png")

    # Сохраняем все результаты в JSON
    all_results = []
    for s in all_summaries:
        all_results.append(Result(
            method=s["method"],
            problem=s["problem"],
            best_value=s["mean_best"],
            best_point=None,
            convergence=s["mean_convergence"],
            seed=0,
        ))
    save_results(all_results, "results/all_results.json")

    print("\n" + "=" * 70)
    print(" ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print(" Результаты сохранены в папке results/")
    print("   - summary.txt - сводка результатов")
    print("   - all_results.json - полные результаты")
    print("   - convergence_*.png - графики сходимости")
    print("=" * 70)


if __name__ == "__main__":
    main()