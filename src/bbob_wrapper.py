"""
Модуль с тестовыми задачами из набора bbob-constrained
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


class BBOBConstrainedProblem:
    """
    Класс для работы с задачами из набора bbob-constrained.
    
    Пример использования:
        problem = BBOBConstrainedProblem('sphere', n_constraints=1)
        objective = problem.get_objective()
        constraint = problem.get_constraint()
        bounds = problem.get_bounds()
    """
    
    def __init__(self, name: str, n_constraints: int, dimension: int = 2):
        """
        Parameters
        ----------
        name : str
            Название целевой функции: 'sphere', 'ellipsoid', 'rastrigin', 'linear'
        n_constraints : int
            Количество ограничений: 1, 3, 9
        dimension : int
            Размерность задачи (2, 3, 5, 10, 20, 40)
        """
        self.name = name
        self.n_constraints = n_constraints
        self.dimension = dimension
        
        # Случайное смещение оптимума
        self.x_opt = np.random.uniform(-4, 4, dimension)
        
        # Случайная матрица вращения для некоторых функций
        self.rotation = self._generate_rotation_matrix(dimension)
        
        # Генерация ограничений
        self.constraints_linear = self._generate_linear_constraints()
        
    def _generate_rotation_matrix(self, n: int) -> np.ndarray:
        """Генерация случайной ортогональной матрицы."""
        H = np.random.randn(n, n)
        Q, _ = np.linalg.qr(H)
        return Q
    
    def _generate_linear_constraints(self) -> List[Tuple[np.ndarray, float]]:
        """
        Генерация линейных ограничений вида a^T x <= b.
        """
        constraints = []
        
        for k in range(self.n_constraints):
            # Случайный вектор нормали
            a = np.random.randn(self.dimension)
            a = a / np.linalg.norm(a)
            
            # Случайное смещение
            if k == 0:
                # Первое ограничение активное в оптимуме
                b = a @ self.x_opt
            else:
                # Остальные ограничения могут быть неактивными
                b = a @ self.x_opt + np.random.uniform(0, 2)
            
            constraints.append((a, b))
        
        return constraints
    
    def objective_sphere(self, x: np.ndarray) -> float:
        """Сферическая функция."""
        return np.sum((x - self.x_opt) ** 2)
    
    def objective_ellipsoid(self, x: np.ndarray) -> float:
        """Эллипсоидальная функция."""
        z = x - self.x_opt
        n = self.dimension
        cond = 1e6
        result = 0
        for i in range(n):
            result += cond ** (i / (n - 1)) * z[i] ** 2
        return result
    
    def objective_rastrigin(self, x: np.ndarray) -> float:
        """Функция Растригина."""
        z = x - self.x_opt
        n = self.dimension
        return 10 * n + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z))
    
    def objective_linear(self, x: np.ndarray) -> float:
        """Линейная функция."""
        return np.sum(x - self.x_opt)
    
    def objective_rotated_ellipsoid(self, x: np.ndarray) -> float:
        """Вращенный эллипсоид."""
        z = self.rotation @ (x - self.x_opt)
        n = self.dimension
        cond = 1e6
        result = 0
        for i in range(n):
            result += cond ** (i / (n - 1)) * z[i] ** 2
        return result
    
    def constraint_linear(self, x: np.ndarray, k: int) -> float:
        """
        Линейное ограничение номер k.
        Возвращает g_k(x) = a^T x - b <= 0
        """
        a, b = self.constraints_linear[k]
        return a @ x - b
    
    def constraint_all(self, x: np.ndarray) -> float:
        """
        Агрегированное ограничение (максимум из всех).
        """
        max_g = -np.inf
        for k in range(self.n_constraints):
            g = self.constraint_linear(x, k)
            max_g = max(max_g, g)
        return max_g
    
    def get_objective(self) -> Callable:
        """Возвращает целевую функцию."""
        objectives = {
            'sphere': self.objective_sphere,
            'ellipsoid': self.objective_ellipsoid,
            'rastrigin': self.objective_rastrigin,
            'linear': self.objective_linear,
            'rotated_ellipsoid': self.objective_rotated_ellipsoid
        }
        return objectives.get(self.name, self.objective_sphere)
    
    def get_constraint(self) -> Callable:
        """Возвращает функцию ограничения."""
        return self.constraint_all
    
    def get_bounds(self) -> np.ndarray:
        """Возвращает границы поиска [-5, 5]."""
        bounds = np.array([[-5, 5]] * self.dimension)
        return bounds
    
    def get_feasible_starting_point(self) -> np.ndarray:
        """
        Возвращает допустимую начальную точку.
        """
        n_trials = 100
        for _ in range(n_trials):
            x = np.random.uniform(-4, 4, self.dimension)
            if self.constraint_all(x) <= 0:
                return x
        return np.zeros(self.dimension)


def create_test_problems() -> List[Dict]:
    """
    Создает набор тестовых задач для бенчмаркинга.
    
    Returns
    -------
    List[Dict]
        Список задач с полями: name, objective, constraint, bounds, dim
    """
    problems = []
    
    # Задачи с разными целевыми функциями и количеством ограничений
    test_configs = [
        ('sphere', 1),
        ('sphere', 3),
        ('sphere', 9),
        ('ellipsoid', 1),
        ('ellipsoid', 3),
        ('rastrigin', 1),
        ('rastrigin', 3),
        ('linear', 1),
        ('rotated_ellipsoid', 1),
    ]
    
    for name, n_cons in test_configs:
        problem = BBOBConstrainedProblem(name, n_cons, dimension=2)
        
        problems.append({
            'name': f"{name}_c{n_cons}",
            'objective': problem.get_objective(),
            'constraint': problem.get_constraint(),
            'bounds': problem.get_bounds(),
            'dim': 2,
            'x0': problem.get_feasible_starting_point()
        })
    
    return problems