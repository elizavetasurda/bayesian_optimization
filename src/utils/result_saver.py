"""Модуль для сохранения результатов экспериментов и построения графиков."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.utils.types import OptimizationProblem


def save_results(
    results: dict[str, dict[str, dict[str, Any]]],
    filepath: Path,
    problems: list[OptimizationProblem],
) -> None:
    """
    Сохранение результатов эксперимента в текстовый файл.

    Аргументы:
        results: Словарь с результатами
        filepath: Путь для сохранения файла
        problems: Список задач для получения информации об оптимумах
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ ПО БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ\n")
        f.write("=" * 100 + "\n\n")

        # Создаём словарь для быстрого доступа к задачам
        problem_dict = {p.name: p for p in problems}

        for prob_name, prob_results in results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"ЗАДАЧА: {prob_name}\n")
            f.write(f"{'=' * 80}\n")

            # Информация о задаче
            if prob_name in problem_dict:
                problem = problem_dict[prob_name]
                f.write(f"Размерность: {problem.dim}\n")
                f.write(f"Оптимальное значение: {problem.optimal_value:.10f}\n")
                f.write(f"Границы: {problem.bounds}\n\n")

            f.write(f"{'Метод':<12} {'Среднее финальное':<20} {'Стд. отклонение':<18} "
                    f"{'Медиана':<15} {'Ср. время (с)':<15}\n")
            f.write("-" * 80 + "\n")

            for method_name, method_results in prob_results.items():
                final_values = method_results["best_values"][:, -1]
                mean_final = np.mean(final_values)
                std_final = np.std(final_values)
                median_final = np.median(final_values)
                
                # Проверяем наличие ключа mean_time
                if "mean_time" in method_results:
                    mean_time = method_results["mean_time"]
                else:
                    mean_time = np.mean(method_results["times"])

                f.write(f"{method_name:<12} {mean_final:<20.6e} {std_final:<18.6e} "
                        f"{median_final:<15.6e} {mean_time:<15.2f}\n")

            f.write("\n")

    print(f"Результаты сохранены в: {filepath}")


def save_convergence_plots(
    results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    n_init: int,
    problems: list[OptimizationProblem],
) -> None:
    """
    Построение и сохранение графиков сходимости для всех задач.

    Аргументы:
        results: Словарь с результатами
        output_dir: Директория для сохранения графиков
        n_init: Размер начальной выборки (для вертикальной линии)
        problems: Список задач
    """
    # Настройка стиля графиков
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = {"CEI": "#2ecc71", "Penalty": "#e74c3c", "Lagrange": "#3498db", "Barrier": "#f39c12"}
    linestyles = {"CEI": "-", "Penalty": "--", "Lagrange": "-.", "Barrier": ":"}

    # Создаём словарь задач для быстрого доступа
    problem_dict = {p.name: p for p in problems}

    for prob_name, prob_results in results.items():
        fig, ax = plt.subplots(figsize=(12, 8))

        # Получаем оптимальное значение
        optimal_value = problem_dict[prob_name].optimal_value if prob_name in problem_dict else 0

        for method_name, method_results in prob_results.items():
            # Проверяем наличие ключа mean_best
            if "mean_best" in method_results:
                mean_values = method_results["mean_best"]
                std_values = method_results["std_best"]
            else:
                # Если нет, вычисляем из best_values
                best_values_matrix = method_results["best_values"]
                mean_values = np.mean(best_values_matrix, axis=0)
                std_values = np.std(best_values_matrix, axis=0)

            # Вычисляем расстояние до оптимума
            distance = np.abs(mean_values - optimal_value)
            distance = np.maximum(distance, 1e-16)

            # Верхняя и нижняя границы
            upper = distance + std_values
            lower = np.maximum(distance - std_values, 1e-16)

            iterations = range(len(mean_values))

            # Рисуем линию сходимости
            ax.plot(
                iterations,
                distance,
                label=method_name,
                color=colors.get(method_name, "gray"),
                linestyle=linestyles.get(method_name, "-"),
                linewidth=2,
                alpha=0.9,
            )

            # Закрашиваем область стандартного отклонения
            ax.fill_between(
                iterations,
                lower,
                upper,
                color=colors.get(method_name, "gray"),
                alpha=0.2,
            )

        # Вертикальная линия, обозначающая конец начальной выборки
        ax.axvline(x=n_init, color="black", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(
            n_init + 0.5,
            ax.get_ylim()[1] * 0.9,
            f"Initial sample (n={n_init})",
            rotation=90,
            fontsize=9,
            alpha=0.7,
        )

        # Настройка осей
        ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax.set_ylabel("Distance to optimum (log scale)", fontsize=12, fontweight="bold")
        ax.set_title(f"Convergence - {prob_name}", fontsize=14, fontweight="bold")

        # Логарифмическая шкала по Y
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")

        # Легенда
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

        # Сохраняем график
        safe_name = prob_name.replace(" ", "_").replace("/", "_")
        plt.tight_layout()
        plt.savefig(output_dir / f"convergence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Создаём общий сравнительный график
    _create_comparison_plot(results, output_dir, n_init, problem_dict)


def _create_comparison_plot(
    results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    n_init: int,
    problem_dict: dict[str, OptimizationProblem],
) -> None:
    """
    Создание общего сравнительного графика для первых 4 задач.

    Аргументы:
        results: Словарь с результатами
        output_dir: Директория для сохранения
        n_init: Размер начальной выборки
        problem_dict: Словарь задач
    """
    colors = {"CEI": "#2ecc71", "Penalty": "#e74c3c", "Lagrange": "#3498db", "Barrier": "#f39c12"}

    # Берём первые 4 задачи
    problem_names = list(results.keys())[:4]

    if len(problem_names) < 4:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, prob_name in enumerate(problem_names[:4]):
        ax = axes[idx]
        prob_results = results[prob_name]
        optimal_value = problem_dict[prob_name].optimal_value if prob_name in problem_dict else 0

        for method_name, method_results in prob_results.items():
            # Проверяем наличие ключа mean_best
            if "mean_best" in method_results:
                mean_values = method_results["mean_best"]
            else:
                best_values_matrix = method_results["best_values"]
                mean_values = np.mean(best_values_matrix, axis=0)

            distance = np.abs(mean_values - optimal_value)
            distance = np.maximum(distance, 1e-16)

            iterations = range(len(mean_values))

            ax.plot(
                iterations,
                distance,
                label=method_name,
                color=colors.get(method_name, "gray"),
                linewidth=2,
                alpha=0.9,
            )

        # Вертикальная линия начальной выборки
        ax.axvline(x=n_init, color="black", linestyle=":", alpha=0.7, linewidth=1.5)

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Distance to optimum", fontsize=10)
        ax.set_title(prob_name, fontsize=11, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Comparison of Bayesian Optimization Methods", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)