"""Сохранение результатов экспериментов."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


@dataclass
class Result:
    """Результат одного запуска оптимизации.
    
    Attributes:
        method: Название метода оптимизации ('CEI', 'Penalty', 'Lagrange', 'Barrier')
        problem: Название тестовой задачи
        best_value: Лучшее найденное значение целевой функции
        best_point: Лучшая найденная точка (список координат)
        convergence: Массив значений сходимости по итерациям
        seed: Случайное зерно для воспроизводимости
    """
    method: str
    problem: str
    best_value: float
    best_point: Optional[np.ndarray]
    convergence: np.ndarray
    seed: int
    
    def to_dict(self) -> dict:
        """Преобразует результат в словарь для JSON сериализации."""
        return {
            'method': self.method,
            'problem': self.problem,
            'best_value': float(self.best_value),
            'best_point': self.best_point.tolist() if self.best_point is not None else None,
            'convergence': self.convergence.tolist(),
            'seed': self.seed,
        }


def save_results(results: List[Result], filepath: str) -> None:
    """Сохраняет список результатов в JSON файл.
    
    Args:
        results: Список объектов Result для сохранения
        filepath: Путь к файлу для сохранения
    """
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Преобразуем результаты в словари
    results_dict = [r.to_dict() for r in results]
    
    # Сохраняем в JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в {filepath}")


def load_results(filepath: str) -> List[Result]:
    """Загружает результаты из JSON файла.
    
    Args:
        filepath: Путь к файлу с результатами
        
    Returns:
        List[Result]: Список загруженных результатов
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)
    
    results = []
    for r in results_dict:
        result = Result(
            method=r['method'],
            problem=r['problem'],
            best_value=r['best_value'],
            best_point=np.array(r['best_point']) if r['best_point'] else None,
            convergence=np.array(r['convergence']),
            seed=r['seed'],
        )
        results.append(result)
    
    return results


def save_summary_text(summaries: List[dict], filepath: str) -> None:
    """Сохраняет сводку результатов в текстовый файл.
    
    Args:
        summaries: Список словарей с усредненными результатами
        filepath: Путь к текстовому файлу
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ\n")
        f.write("=" * 80 + "\n\n")
        
        for s in summaries:
            f.write(f"Метод: {s['method']}\n")
            f.write(f"Задача: {s['problem']}\n")
            f.write(f"Размерность: {s['dimension']}\n")
            f.write(f"Среднее лучшее значение: {s['mean_best']:.8f}\n")
            f.write(f"Стандартное отклонение: {s['std_best']:.8f}\n")
            f.write("-" * 80 + "\n")
        
        # Добавляем сводную таблицу
        f.write("\n\nСВОДНАЯ ТАБЛИЦА\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Метод':<12} {'Задача':<12} {'Dim':<6} {'Среднее':<15} {'Std':<15}\n")
        f.write("-" * 80 + "\n")
        
        for s in summaries:
            f.write(f"{s['method']:<12} {s['problem']:<12} {s['dimension']:<6} {s['mean_best']:<15.8f} {s['std_best']:<15.8f}\n")