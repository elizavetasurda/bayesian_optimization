"""Модуль для проведения экспериментов."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .optimizer import BayesianOptimizer, OptimizationResult


@dataclass
class ExperimentResult:
    """Результаты эксперимента."""
    method_name: str
    problem_name: str
    runs: List[OptimizationResult] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Доля успешных запусков."""
        successful = sum(1 for r in self.runs if r.is_feasible)
        return successful / len(self.runs) if self.runs else 0.0
    
    def get_mean_best_value(self) -> float:
        """Среднее лучшее значение."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        return np.mean(values) if values else np.nan
    
    def get_median_best_value(self) -> float:
        """Медиана лучших значений."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        return np.median(values) if values else np.nan
    
    def get_std_best_value(self) -> float:
        """Стандартное отклонение."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        return np.std(values) if values else np.nan
    
    def get_best_value(self) -> float:
        """Лучшее значение."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        return np.min(values) if values else np.nan
    
    def get_worst_value(self) -> float:
        """Худшее значение."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        return np.max(values) if values else np.nan
    
    def get_mean_first_feasible(self) -> float:
        """Средняя итерация первого допустимого."""
        firsts = [r.first_feasible for r in self.runs if r.first_feasible is not None]
        return np.mean(firsts) if firsts else np.nan
    
    def get_quartiles(self) -> Dict[str, float]:
        """Квартили распределения."""
        values = [r.best_value for r in self.runs if r.is_feasible]
        if values:
            return {
                'q1': np.percentile(values, 25),
                'q2': np.percentile(values, 50),
                'q3': np.percentile(values, 75)
            }
        return {'q1': np.nan, 'q2': np.nan, 'q3': np.nan}


def run_experiment(
    problem: Dict[str, Any],
    method: str,
    n_runs: int = 10,
    n_iter: int = 50
) -> ExperimentResult:
    """Запуск эксперимента."""
    result = ExperimentResult(
        method_name=method,
        problem_name=problem.get('name', 'unknown')
    )
    
    for seed in range(n_runs):
        np.random.seed(42 + seed)
        
        # Подбираем коэффициенты для разных методов
        if method == 'CEI':
            optimizer = BayesianOptimizer(
                objective=problem['objective'],
                constraints=problem['constraints'],
                bounds=problem['bounds'],
                method=method,
                n_start=10
            )
        elif method == 'Penalty':
            optimizer = BayesianOptimizer(
                objective=problem['objective'],
                constraints=problem['constraints'],
                bounds=problem['bounds'],
                method=method,
                n_start=10,
                penalty_coef=1e5  # Увеличили штраф
            )
        elif method == 'Lagrange':
            optimizer = BayesianOptimizer(
                objective=problem['objective'],
                constraints=problem['constraints'],
                bounds=problem['bounds'],
                method=method,
                n_start=10,
                lagrange_coef=(2000.0, 1000.0)  # Увеличили коэффициенты
            )
        else:  # Barrier
            optimizer = BayesianOptimizer(
                objective=problem['objective'],
                constraints=problem['constraints'],
                bounds=problem['bounds'],
                method=method,
                n_start=10,
                barrier_coef=200.0  # Увеличили барьер
            )
        
        opt_result = optimizer.optimize(n_iter=n_iter)
        result.runs.append(opt_result)
    
    return result


def run_experiments(
    problems: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None,
    n_runs: int = 10,
    n_iter: int = 50
) -> Dict[str, Dict[str, ExperimentResult]]:
    """Запуск всех экспериментов."""
    if methods is None:
        methods = ['CEI', 'Penalty', 'Lagrange', 'Barrier']
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    
    for prob_name, problem in problems.items():
        problem['name'] = problem.get('name', prob_name)
        print(f"\n{'='*60}")
        print(f"Задача: {problem['name']}")
        print(f"{'='*60}")
        
        all_results[prob_name] = {}
        
        for method in methods:
            print(f"\n  Метод: {method}")
            result = run_experiment(problem, method, n_runs, n_iter)
            all_results[prob_name][method] = result
            
            success_rate = result.get_success_rate() * 100
            mean_val = result.get_mean_best_value()
            best_val = result.get_best_value()
            
            print(f"    Успешность: {success_rate:.0f}%")
            if not np.isnan(mean_val):
                print(f"    Среднее: {mean_val:.2f}")
                print(f"    Лучшее: {best_val:.2f}")
                if prob_name == 'pressure_vessel':
                    print(f"    Отклонение от теории: {(mean_val - 2414)/2414*100:+.1f}%")
    
    return all_results


def save_results(
    results: Dict[str, Dict[str, ExperimentResult]],
    filename: str = 'results.txt'
) -> None:
    """Сохранение результатов."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ\n")
        f.write("=" * 80 + "\n\n")
        
        for prob_name, methods_results in results.items():
            prob_display = list(methods_results.values())[0].problem_name
            f.write(f"Задача: {prob_display}\n")
            f.write("-" * 40 + "\n")
            
            # Таблица результатов
            f.write("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}\n".format(
                "Метод", "Успех%", "Среднее", "Медиана", "Лучшее", "Худшее"
            ))
            f.write("-" * 70 + "\n")
            
            for method, exp_result in methods_results.items():
                success = exp_result.get_success_rate() * 100
                mean_val = exp_result.get_mean_best_value()
                median_val = exp_result.get_median_best_value()
                best_val = exp_result.get_best_value()
                worst_val = exp_result.get_worst_value()
                
                if not np.isnan(mean_val):
                    f.write("{:<12} {:>9.0f}% {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}\n".format(
                        method, success, mean_val, median_val, best_val, worst_val
                    ))
                else:
                    f.write("{:<12} {:>9.0f}% {:>10} {:>10} {:>10} {:>10}\n".format(
                        method, success, "-", "-", "-", "-"
                    ))
            
            # Дополнительная статистика для сосуда
            if prob_name == 'pressure_vessel':
                f.write("\nОтклонение от теории (2414 кг):\n")
                for method, exp_result in methods_results.items():
                    mean_val = exp_result.get_mean_best_value()
                    if not np.isnan(mean_val):
                        dev = (mean_val - 2414) / 2414 * 100
                        f.write(f"  {method}: {dev:+.1f}%\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nРезультаты сохранены в файл: {filename}")
