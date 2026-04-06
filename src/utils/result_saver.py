"""Утилиты для сохранения результатов."""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Result:
    """Класс для хранения результатов оптимизации."""
    method: str
    problem: str
    best_value: float
    best_point: np.ndarray
    convergence: np.ndarray

    def to_dict(self) -> dict:
        """Преобразует в словарь для JSON."""
        d = asdict(self)
        d["best_point"] = d["best_point"].tolist()
        d["convergence"] = d["convergence"].tolist()
        return d


def save_results(results: List[Result], filepath: str) -> None:
    """Сохраняет результаты в текстовый файл.

    Args:
        results: Список объектов Result
        filepath: Путь к файлу для сохранения
    """
    data = [r.to_dict() for r in results]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
