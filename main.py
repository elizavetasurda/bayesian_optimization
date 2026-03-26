"""Главный модуль для запуска экспериментов."""

import sys
import time
from pathlib import Path

# Добавляем путь для импорта наших модулей
sys.path.insert(0, str(Path(__file__).parent))

from src.problems import get_problems, get_theoretical_optimum
from src.experiment import run_experiments, save_results


def create_plots(results, m_opt):
    """Создает графики для всех задач."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Список методов
        methods = ['CEI', 'Penalty', 'Lagrange', 'Barrier']
        
        # =========================================================
        # ГРАФИК 1: СХОДИМОСТЬ (для всех задач в одном окне)
        # =========================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Сходимость методов оптимизации', fontsize=14, fontweight='bold')
        
        # Задача 1: Сферическая функция
        ax = axes[0]
        for method in methods:
            if method in results['sphere']:
                exp = results['sphere'][method]
                # Собираем историю сходимости по всем запускам
                histories = [r.history for r in exp.runs if r.is_feasible and r.history]
                if histories:
                    # Выравниваем длину и усредняем
                    max_len = max(len(h) for h in histories)
                    padded = [h + [h[-1]]*(max_len - len(h)) for h in histories]
                    mean_hist = np.mean(padded, axis=0)
                    ax.plot(mean_hist[:51], label=method, linewidth=2)
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Значение функции')
        ax.set_title('Сферическая функция (оптимум = 0)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Задача 2: Функция Розенброка
        ax = axes[1]
        for method in methods:
            if method in results['rosenbrock']:
                exp = results['rosenbrock'][method]
                histories = [r.history for r in exp.runs if r.is_feasible and r.history]
                if histories:
                    max_len = max(len(h) for h in histories)
                    padded = [h + [h[-1]]*(max_len - len(h)) for h in histories]
                    mean_hist = np.mean(padded, axis=0)
                    ax.plot(mean_hist[:51], label=method, linewidth=2)
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Значение функции')
        ax.set_title('Функция Розенброка (оптимум = 0)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Задача 3: Сосуд под давлением
        ax = axes[2]
        for method in methods:
            if method in results['pressure_vessel']:
                exp = results['pressure_vessel'][method]
                histories = [r.history for r in exp.runs if r.is_feasible and r.history]
                if histories:
                    max_len = max(len(h) for h in histories)
                    padded = [h + [h[-1]]*(max_len - len(h)) for h in histories]
                    mean_hist = np.mean(padded, axis=0)
                    ax.plot(mean_hist[:51], label=method, linewidth=2)
        ax.axhline(m_opt, color='k', linestyle='--', label='Теория')
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Масса, кг')
        ax.set_title('Сосуд под давлением')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('graph_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  График 1 (сходимость) сохранен: graph_convergence.png")
        
        # =========================================================
        # ГРАФИК 2: СРАВНЕНИЕ МЕТОДОВ (только для сосуда под давлением)
        # =========================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Сравнение методов для задачи о сосуде под давлением', fontsize=14, fontweight='bold')
        
        # Данные для графика
        method_names = []
        mean_values = []
        std_values = []
        best_values = []
        
        for method in methods:
            if method in results['pressure_vessel']:
                exp = results['pressure_vessel'][method]
                mean_val = exp.get_mean_best_value()
                std_val = exp.get_std_best_value()
                best_val = exp.get_best_value()
                success_rate = exp.get_success_rate() * 100
                
                if not np.isnan(mean_val):
                    method_names.append(f"{method}\n({success_rate:.0f}%)")
                    mean_values.append(mean_val)
                    std_values.append(std_val)
                    best_values.append(best_val)
        
        # Столбцы для средних значений
        x = np.arange(len(method_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mean_values, width, yerr=std_values, 
                       capsize=5, label='Средняя масса', color='skyblue')
        bars2 = ax.bar(x + width/2, best_values, width, 
                       label='Лучшая масса', color='darkorange')
        
        # Линия теоретического оптимума
        ax.axhline(m_opt, color='r', linestyle='--', linewidth=2, label=f'Теория: {m_opt:.0f} кг')
        
        ax.set_xlabel('Метод (успешность)')
        ax.set_ylabel('Масса, кг')
        ax.set_title('Сравнение методов')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Подписи значений на столбцах
        for bar, val in zip(bars1, mean_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, best_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('graph_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  График 2 (сравнение методов) сохранен: graph_comparison.png")
        
    except Exception as e:
        print(f"\nНе получилось создать графики: {e}")


def main():
    """Запускает эксперименты и сохраняет результаты."""
    print("\n" + "=" * 80)
    print(" " * 25 + "БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ С ОГРАНИЧЕНИЯМИ")
    print("=" * 80)
    
    # Теоретический оптимум для задачи о сосуде
    R_opt, m_opt = get_theoretical_optimum()
    print("\nТеоретический оптимум для задачи о сосуде:")
    print(f"  Радиус: {R_opt[0]:.4f} м")
    print(f"  Толщина: {R_opt[1]:.5f} м")
    print(f"  Масса: {m_opt:.0f} кг")
    
    # Получаем задачи
    problems = get_problems()
    
    # Настройки эксперимента
    N_RUNS = 10      # Количество запусков
    N_ITER = 50      # Количество итераций
    
    print(f"\nПараметры эксперимента:")
    print(f"  Задачи: {', '.join(problems.keys())}")
    print(f"  Методы: CEI, Penalty, Lagrange, Barrier")
    print(f"  Количество запусков: {N_RUNS}")
    print(f"  Итераций на запуск: {N_ITER}")
    
    # Запускаем эксперименты
    start_time = time.time()
    results = run_experiments(problems, n_runs=N_RUNS, n_iter=N_ITER)
    elapsed_time = time.time() - start_time
    
    print(f"\nВремя выполнения: {elapsed_time:.1f} секунд")
    
    # Сохраняем текстовые результаты
    save_results(results, 'results.txt')
    
    # Рисуем графики
    print("\nСоздание графиков...")
    create_plots(results, m_opt)
    
    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)
    print("\nФайлы результатов:")
    print("  results.txt          - текстовые результаты")
    print("  graph_convergence.png - график сходимости")
    print("  graph_comparison.png  - сравнение методов")
    print("=" * 80)


if __name__ == '__main__':
    main()
