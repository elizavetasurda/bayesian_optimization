"""Основной скрипт для запуска байесовской оптимизации с ограничениями."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import os

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
    """Запускает оптимизацию заданным методом на заданной задаче."""
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
    """Запускает оптимизацию несколько раз с разными seed и усредняет результаты."""
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


def plot_convergence_comparison(all_summaries: list, problems: list, dimensions: list, n_initial: int) -> None:
    """Строит графики сходимости для всех задач и размерностей."""
    
    # Словарь для переименования методов на русский (если нужно)
    method_names_ru = {
        "CEI": "CEI",
        "Penalty": "Штраф",
        "Lagrange": "Лагранж",
        "Barrier": "Барьер"
    }
    
    for problem in problems:
        for dim in dimensions:
            # Фильтруем данные для текущей задачи и размерности
            problem_data = [s for s in all_summaries 
                           if s["problem"] == problem and s["dimension"] == dim]
            
            if not problem_data:
                print(f"Нет данных для {problem} dim={dim}")
                continue
                
            plt.figure(figsize=(10, 6))
            
            for data in problem_data:
                method = data["method"]
                iterations = range(len(data["mean_convergence"]))
                
                # Основная линия
                plt.plot(iterations, data["mean_convergence"], 
                        label=method_names_ru.get(method, method), linewidth=2)
                
                # Доверительный интервал
                plt.fill_between(
                    iterations,
                    data["mean_convergence"] - data["std_convergence"],
                    data["mean_convergence"] + data["std_convergence"],
                    alpha=0.2,
                )
            
            # Вертикальная линия для начальной выборки
            plt.axvline(x=n_initial, color="gray", linestyle="--", linewidth=1.5, 
                       label=f"Начальная выборка ({n_initial})")
            
            plt.xlabel("Итерация", fontsize=12)
            plt.ylabel("Лучшее допустимое значение (среднее)", fontsize=12)
            plt.title(f"Сходимость методов на {problem.capitalize()} (размерность={dim})", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Логарифмическая шкала для большинства задач
            if problem in ["sphere", "rosenbrock"]:
                plt.yscale("log")
            
            plt.tight_layout()
            
            # ИСПРАВЛЕНО: Сохраняем с правильными именами файлов
            filename = f"graph_convergence_{problem}_c1_dim{dim}.png"
            filepath = os.path.join("results", filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   Сохранён график: {filepath}")


def plot_all_methods_comparison(all_summaries: list, problems: list, dimensions: list) -> None:
    """Строит сравнительные графики для всех методов на одной фигуре."""
    
    fig, axes = plt.subplots(len(problems), len(dimensions), figsize=(15, 10))
    
    # Если только одна подзадача, делаем axes двумерным
    if len(problems) == 1:
        axes = axes.reshape(1, -1)
    if len(dimensions) == 1:
        axes = axes.reshape(-1, 1)
    
    method_names_ru = {
        "CEI": "CEI",
        "Penalty": "Штраф",
        "Lagrange": "Лагранж",
        "Barrier": "Барьер"
    }
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, problem in enumerate(problems):
        for j, dim in enumerate(dimensions):
            ax = axes[i, j]
            
            problem_data = [s for s in all_summaries 
                           if s["problem"] == problem and s["dimension"] == dim]
            
            for idx, data in enumerate(problem_data):
                method = data["method"]
                iterations = range(len(data["mean_convergence"]))
                
                ax.plot(iterations, data["mean_convergence"], 
                       label=method_names_ru.get(method, method),
                       linewidth=2, color=colors[idx % len(colors)])
                ax.fill_between(
                    iterations,
                    data["mean_convergence"] - data["std_convergence"],
                    data["mean_convergence"] + data["std_convergence"],
                    alpha=0.2,
                    color=colors[idx % len(colors)]
                )
            
            ax.set_xlabel("Итерация")
            ax.set_ylabel("Лучшее значение")
            ax.set_title(f"{problem.capitalize()}, dim={dim}")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("results/graph_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Сохранён график: results/graph_comparison.png")


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

    # Создаём папку для результатов
    os.makedirs("results", exist_ok=True)
    
    # Сохраняем сводку
    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write("Результаты экспериментов\n")
        f.write("=" * 70 + "\n\n")
        for s in all_summaries:
            f.write(
                f"{s['method']} on {s['problem']} (dim={s['dimension']}):\n"
                f"  Среднее: {s['mean_best']:.8f}\n"
                f"  Стандартное отклонение: {s['std_best']:.8f}\n\n"
            )

    # Построение графиков
    print("\n" + "=" * 70)
    print(" Построение графиков...")
    print("=" * 70)
    
    # Графики для каждой задачи и размерности
    plot_convergence_comparison(all_summaries, problems, dimensions, n_initial)
    
    # Сравнительный график всех методов
    plot_all_methods_comparison(all_summaries, problems, dimensions)

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
    print("   - graph_*.png - графики сходимости")
    print("=" * 70)


if __name__ == "__main__":
    main()