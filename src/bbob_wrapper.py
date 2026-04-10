"""Обёртка для тестового набора bbob-constrained из OptunaHub."""
import numpy as np
from cocoex import Problem as CocoProblem

class BBOBConstrainedProblem:
    """Адаптер для задачи bbob-constrained."""
    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        # Используем функцию evaluate для получения значения и ограничений
        self.coco_problem = CocoProblem(f"bbob-constrained_{function_id}_i{instance_id}_d{dimension}")
        self.dim = dimension

    def objective(self, x: np.ndarray) -> float:
        """Целевая функция."""
        # coco ожидает list, возвращает значение функции
        return self.coco_problem(x.tolist())

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Вектор ограничений. Значения > 0 означают нарушение."""
        # В coco-experiment ограничения возвращаются через отдельный вызов
        cons_values = self.coco_problem.constraint(x.tolist())
        return np.array(cons_values)

    @property
    def bounds(self) -> list:
        """Границы поиска."""
        return [(self.coco_problem.lower_bounds[i], self.coco_problem.upper_bounds[i]) for i in range(self.dim)]