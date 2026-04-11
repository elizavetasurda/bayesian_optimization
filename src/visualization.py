"""Визуализация результатов оптимизации."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    results: Dict,
    filename: str = "convergence",
    show_initial_split: bool = True
) -> None:
    """Построение графика сходимости.
    
    Args:
        results: Словарь с результатами оптимизации для разных методов.
        filename: Имя файла для сохранения.
        show_initial_split: Показывать ли вертикальную линию разделения начальной выборки.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {'CEI': 'blue', 'Penalty': 'green', 'Lagrange': 'red', 'Barrier': 'orange'}
    
    for method_name, result in results.items():
        history = result.history
        n_initial = result.n_initial
        
        # График сходимости
        plt.plot(
            range(len(history)),
            history,
            label=method_name,
            color=colors.get(method_name, 'black'),
            linewidth=2
        )
        
        # Вертикальная линия разделения начальной выборки
        if show_initial_split and n_initial > 0:
            plt.axvline(x=n_initial - 1, linestyle='--', color='gray', alpha=0.5)
    
    plt.xlabel('Итерация')
    plt.ylabel('Лучшее значение f(x)')
    plt.title('Сходимость байесовской оптимизации')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show_initial_split:
        plt.text(
            n_initial - 1, plt.ylim()[1] * 0.9,
            f'Начальная выборка ({n_initial} точек)',
            rotation=90, fontsize=8, alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=150)
    plt.close()


def plot_comparison(results: Dict, filename: str = "comparison") -> None:
    """Построение сравнительной диаграммы.
    
    Args:
        results: Словарь с результатами оптимизации.
        filename: Имя файла для сохранения.
    """
    methods = list(results.keys())
    best_values = [results[m].best_f for m in methods]
    success_flags = [results[m].success for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График лучших значений
    bars = ax1.bar(methods, best_values, color=['blue', 'green', 'red', 'orange'])
    ax1.set_ylabel('Лучшее значение f(x)')
    ax1.set_title('Сравнение лучших значений')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Подписи значений на столбцах
    for bar, val in zip(bars, best_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # График успешности
    colors_success = ['green' if s else 'red' for s in success_flags]
    ax2.bar(methods, success_flags, color=colors_success)
    ax2.set_ylabel('Успех (1 - да, 0 - нет)')
    ax2.set_title('Достижение допустимого решения')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=150)
    plt.close()