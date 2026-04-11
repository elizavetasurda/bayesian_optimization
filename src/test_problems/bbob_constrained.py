"""Набор тестовых задач для байесовской оптимизации с ограничениями.

Содержит 10 задач различной сложности для тестирования методов оптимизации.
Все задачи имеют ограничения и известные глобальные оптимумы.
"""

import numpy as np
from src.utils.types import OptimizationProblem


class BBOBConstrainedProblems:
    """Класс-контейнер для тестовых задач."""

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Сферическая функция.

        Формула: f(x) = sum(x_i^2)
        Свойства: гладкая, выпуклая, унимодальная
        Глобальный минимум: 0 в точке 0

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение функции
        """
        return float(np.sum(x ** 2))

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Функция Розенброка.

        Формула: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        Свойства: овражная, сложная для оптимизации
        Глобальный минимум: 0 в точке (1,1,...,1)

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение функции
        """
        result = 0.0
        for i in range(len(x) - 1):
            result += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return float(result)

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Функция Растригина.

        Формула: f(x) = 10*n + sum(x_i^2 - 10*cos(2πx_i))
        Свойства: многоэкстремальная, множество локальных минимумов
        Глобальный минимум: 0 в точке 0

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение функции
        """
        n = len(x)
        return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    @staticmethod
    def quadratic(x: np.ndarray) -> float:
        """
        Квадратичная функция.

        Формула: f(x) = x0^2 + 2*x1^2 (для 2D) или +3*x2^2 (для 3D)
        Свойства: выпуклая, плохо обусловленная
        Глобальный минимум: 0 в точке 0

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение функции
        """
        if len(x) == 2:
            return float(x[0] ** 2 + 2 * x[1] ** 2)
        return float(x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2)

    @staticmethod
    def booth(x: np.ndarray) -> float:
        """
        Функция Бута.

        Формула: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
        Свойства: выпуклая, унимодальная
        Глобальный минимум: 0 в точке (1, 3)

        Параметры:
            x: Входной вектор (2D)

        Возвращает:
            Значение функции
        """
        return float((x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2)

    @staticmethod
    def linear_constraint_1(x: np.ndarray) -> float:
        """
        Линейное ограничение 1.

        Формула: x0 + x1 - 1 <= 0
        Создаёт верхнюю границу допустимой области.

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float(x[0] + x[1] - 1.0)

    @staticmethod
    def linear_constraint_2(x: np.ndarray) -> float:
        """
        Линейное ограничение 2.

        Формула: -x0 - x1 + 0.5 <= 0 (эквивалентно x0 + x1 >= 0.5)
        Создаёт нижнюю границу допустимой области.

        Параметры:
            x: Входной вектор

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float(-x[0] - x[1] + 0.5)

    @staticmethod
    def circle_constraint(x: np.ndarray) -> float:
        """
        Круговое ограничение.

        Формула: x0^2 + x1^2 - 1 <= 0
        Ограничивает точки кругом радиуса 1.

        Параметры:
            x: Входной вектор (2D)

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float(x[0] ** 2 + x[1] ** 2 - 1.0)

    @staticmethod
    def circle_constraint_3d(x: np.ndarray) -> float:
        """
        Шаровое ограничение (3D).

        Формула: x0^2 + x1^2 + x2^2 - 1 <= 0
        Ограничивает точки шаром радиуса 1.

        Параметры:
            x: Входной вектор (3D)

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0)

    @staticmethod
    def circle_constraint_offset(x: np.ndarray) -> float:
        """
        Смещённое круговое ограничение.

        Формула: (x0-0.5)^2 + (x1-0.5)^2 - 0.25 <= 0
        Ограничивает точки кругом радиуса 0.5 с центром в (0.5, 0.5).

        Параметры:
            x: Входной вектор (2D)

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.25)

    @staticmethod
    def nonlinear_constraint(x: np.ndarray) -> float:
        """
        Нелинейное ограничение.

        Формула: x0 * x1 - 0.1 <= 0
        Создаёт гиперболическую границу.

        Параметры:
            x: Входной вектор (2D)

        Возвращает:
            Значение ограничения (<=0 для допустимых точек)
        """
        return float(x[0] * x[1] - 0.1)


def create_bbob_problem_set() -> list[OptimizationProblem]:
    """
    Создание набора из 10 тестовых задач.

    Задачи включают различные типы целевых функций:
    - Простые (Sphere, Quadratic)
    - Овражные (Rosenbrock)
    - Многоэкстремальные (Rastrigin)
    - Выпуклые (Booth)
    - Различные размерности (2D и 3D)
    - Различные типы ограничений (линейные, круговые, нелинейные)

    Возвращает:
        Список из 10 задач OptimizationProblem
    """
    problems = []

    # Задача 1: Sphere с линейными ограничениями
    problems.append(OptimizationProblem(
        name="Sphere_Linear",
        dim=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=BBOBConstrainedProblems.sphere,
        constraints=[
            BBOBConstrainedProblems.linear_constraint_1,
            BBOBConstrainedProblems.linear_constraint_2,
        ],
        optimal_value=0.125,
        optimal_point=np.array([0.25, 0.25]),
    ))

    # Задача 2: Sphere с круговым ограничением
    problems.append(OptimizationProblem(
        name="Sphere_Circle",
        dim=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.sphere,
        constraints=[BBOBConstrainedProblems.circle_constraint],
        optimal_value=0.0,
        optimal_point=np.array([0.0, 0.0]),
    ))

    # Задача 3: Rosenbrock с круговым ограничением
    problems.append(OptimizationProblem(
        name="Rosenbrock_Circle",
        dim=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.rosenbrock,
        constraints=[BBOBConstrainedProblems.circle_constraint],
        optimal_value=0.0,
        optimal_point=np.array([1.0, 1.0]),
    ))

    # Задача 4: Rastrigin с круговым ограничением
    problems.append(OptimizationProblem(
        name="Rastrigin_Circle",
        dim=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.rastrigin,
        constraints=[BBOBConstrainedProblems.circle_constraint],
        optimal_value=0.0,
        optimal_point=np.array([0.0, 0.0]),
    ))

    # Задача 5: Booth с круговым ограничением
    problems.append(OptimizationProblem(
        name="Booth_Circle",
        dim=2,
        bounds=[(-4.0, 4.0), (-4.0, 4.0)],
        objective=BBOBConstrainedProblems.booth,
        constraints=[BBOBConstrainedProblems.circle_constraint],
        optimal_value=0.0,
        optimal_point=np.array([1.0, 3.0]),
    ))

    # Задача 6: Sphere с двумя кругами (пересечение)
    problems.append(OptimizationProblem(
        name="Sphere_TwoCircles",
        dim=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.sphere,
        constraints=[
            BBOBConstrainedProblems.circle_constraint,
            BBOBConstrainedProblems.circle_constraint_offset,
        ],
        optimal_value=0.0,
        optimal_point=np.array([0.0, 0.0]),
    ))

    # Задача 7: Quadratic с линейными ограничениями
    problems.append(OptimizationProblem(
        name="Quadratic_Linear",
        dim=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=BBOBConstrainedProblems.quadratic,
        constraints=[
            BBOBConstrainedProblems.linear_constraint_1,
            BBOBConstrainedProblems.linear_constraint_2,
        ],
        optimal_value=0.125,
        optimal_point=np.array([0.25, 0.25]),
    ))

    # Задача 8: Sphere 3D с шаровым ограничением
    problems.append(OptimizationProblem(
        name="Sphere_3D",
        dim=3,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.sphere,
        constraints=[BBOBConstrainedProblems.circle_constraint_3d],
        optimal_value=0.0,
        optimal_point=np.array([0.0, 0.0, 0.0]),
    ))

    # Задача 9: Rosenbrock 3D с шаровым ограничением
    problems.append(OptimizationProblem(
        name="Rosenbrock_3D",
        dim=3,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.rosenbrock,
        constraints=[BBOBConstrainedProblems.circle_constraint_3d],
        optimal_value=0.0,
        optimal_point=np.array([1.0, 1.0, 1.0]),
    ))

    # Задача 10: Rastrigin 3D с шаровым ограничением
    problems.append(OptimizationProblem(
        name="Rastrigin_3D",
        dim=3,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        objective=BBOBConstrainedProblems.rastrigin,
        constraints=[BBOBConstrainedProblems.circle_constraint_3d],
        optimal_value=0.0,
        optimal_point=np.array([0.0, 0.0, 0.0]),
    ))

    return problems


def get_problem_by_index(index: int) -> OptimizationProblem:
    """
    Получение задачи по индексу.

    Параметры:
        index: Индекс задачи (0-9)

    Возвращает:
        Задача OptimizationProblem

    Исключения:
        ValueError: если индекс вне диапазона
    """
    problems = create_bbob_problem_set()
    if 0 <= index < len(problems):
        return problems[index]
    raise ValueError(f"Index {index} out of range. Available: 0-{len(problems)-1}")