import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time
import warnings
warnings.filterwarnings('ignore')

from src.bayesian_optimization.cei import CEIBayesianOptimization
from src.bayesian_optimization.penalty import PenaltyBayesianOptimization
from src.bayesian_optimization.lagrange import LagrangeBayesianOptimization
from src.bayesian_optimization.barrier import BarrierBayesianOptimization
from src.test_problems.bbob_constrained import create_test_problems

def run_optimization(problem: Dict, method: str, n_init: int = 8, n_iter: int = 20) -> Dict:
    """Запуск оптимизации заданным методом."""
    objective = problem['objective']
    constraint = problem['constraint']
    bounds = problem['bounds']
    random_state = 42
    
    if method == 'CEI':
        optimizer = CEIBayesianOptimization(objective, constraint, bounds, n_init, n_iter, random_state)
    elif method == 'Penalty':
        optimizer = PenaltyBayesianOptimization(objective, constraint, bounds, n_init, n_iter, random_state)
    elif method == 'Lagrange':
        optimizer = LagrangeBayesianOptimization(objective, constraint, bounds, n_init, n_iter, random_state)
    elif method == 'Barrier':
        optimizer = BarrierBayesianOptimization(objective, constraint, bounds, n_init, n_iter, random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    start_time = time.time()
    result = optimizer.optimize()
    elapsed_time = time.time() - start_time
    
    return {
        'method': method,
        'problem': problem['name'],
        'best_f': result['best_solution']['f'],
        'best_g': result['best_solution']['g'],
        'best_x': result['best_solution']['x'],
        'time': elapsed_time,
        'n_evaluations': result['n_evaluations'],
        'history': result['history']
    }

def plot_convergence(results: List[Dict], problem_name: str) -> None:
    """Построение графиков сходимости для всех методов."""
    plt.figure(figsize=(10, 6))
    
    colors = {'CEI': 'blue', 'Penalty': 'green', 'Lagrange': 'orange', 'Barrier': 'red'}
    
    for result in results:
        if result['problem'] != problem_name:
            continue
        
        method = result['method']
        history_best = result['history']['best_y']
        
        # Отфильтровываем inf значения
        history_clean = []
        for h in history_best:
            if np.isfinite(h):
                history_clean.append(h)
            elif history_clean:
                history_clean.append(history_clean[-1])
            else:
                history_clean.append(1e6)
        
        plt.plot(
            range(len(history_clean)), 
            history_clean, 
            label=f"{method} (final: {result['best_f']:.4f})",
            color=colors.get(method, 'black'),
            linewidth=2,
            marker='o',
            markersize=3
        )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best feasible objective value', fontsize=12)
    plt.title(f'Convergence on {problem_name}', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    # Сохраняем график
    filename = f'graph_convergence_{problem_name}.png'
    plt.savefig(filename, dpi=150)
    print(f"  ✓ Saved convergence plot: {filename}")
    plt.close()

def plot_comparison(all_results: List[Dict]) -> None:
    """Построение сравнительной диаграммы результатов."""
    methods = ['CEI', 'Penalty', 'Lagrange', 'Barrier']
    problems = list(set([r['problem'] for r in all_results]))
    
    n_problems = len(problems)
    fig, axes = plt.subplots(n_problems, 1, figsize=(12, 5 * n_problems))
    
    if n_problems == 1:
        axes = [axes]
    
    for idx, problem in enumerate(problems):
        ax = axes[idx]
        
        method_names = []
        best_values = []
        
        for method in methods:
            results_method = [r for r in all_results if r['problem'] == problem and r['method'] == method]
            if results_method:
                method_names.append(method)
                best_values.append(results_method[0]['best_f'])
        
        # Создаем столбцы с разными цветами
        bar_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(method_names)]
        bars = ax.bar(method_names, best_values, color=bar_colors, alpha=0.7)
        
        ax.set_ylabel('Best f(x)', fontsize=11)
        ax.set_title(f'Problem: {problem}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars, best_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('graph_comparison.png', dpi=150)
    print("\n✓ Saved comparison plot: graph_comparison.png")
    plt.close()

def main():
    print("=" * 60)
    print("Bayesian Optimization with Constraints - Benchmark")
    print("=" * 60)
    
    # Создание тестовых задач
    problems = create_test_problems()
    print(f"\nCreated {len(problems)} test problems")
    
    # Выбираем задачи для тестирования
    test_problems = [p for p in problems if p['name'] in [
        'sphere_c1_d2', 
        'ellipsoid_c1_d2', 
        'rastrigin_c1_d2',
        'linear_c1_d2'
    ]]
    
    print(f"Testing on {len(test_problems)} problems: {[p['name'] for p in test_problems]}")
    
    methods = ['CEI', 'Penalty', 'Lagrange', 'Barrier']
    all_results = []
    
    for problem in test_problems:
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem['name']}")
        print(f"{'=' * 60}")
        
        for method in methods:
            print(f"\nRunning {method}...")
            try:
                result = run_optimization(problem, method, n_init=8, n_iter=15)
                all_results.append(result)
                print(f"  ✓ Best f(x) = {result['best_f']:.6f}")
                print(f"  ✓ Constraint g(x) = {result['best_g']:.6f}")
                print(f"  ✓ Time = {result['time']:.2f} sec")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_results.append({
                    'method': method,
                    'problem': problem['name'],
                    'best_f': np.inf,
                    'best_g': np.inf,
                    'time': 0,
                    'history': {'best_y': [np.inf]}
                })
    
    # Построение графиков
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print(f"{'=' * 60}")
    
    # Графики сходимости для каждой задачи
    unique_problems = list(set([r['problem'] for r in all_results]))
    for problem_name in unique_problems:
        plot_convergence(all_results, problem_name)
    
    # Сравнительная диаграмма
    plot_comparison(all_results)
    
    # Вывод результатов
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Problem':<20} {'Method':<12} {'Best f(x)':<15} {'Feasible':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for result in sorted(all_results, key=lambda x: (x['problem'], x['method'])):
        feasible = "✓" if result['best_g'] <= 1e-6 else "✗"
        f_val = f"{result['best_f']:.6f}" if np.isfinite(result['best_f']) else "FAIL"
        print(f"{result['problem']:<20} {result['method']:<12} {f_val:<15} {feasible:<10} {result['time']:.2f}")
    
    print("=" * 60)
    print("\n✅ Plots saved:")
    print("   - graph_comparison.png")
    for problem_name in unique_problems:
        print(f"   - graph_convergence_{problem_name}.png")

if __name__ == "__main__":
    main()
