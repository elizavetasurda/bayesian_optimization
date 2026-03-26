"""Набор тестовых задач оптимизации с ограничениями."""

import numpy as np
from typing import Dict, Any, Tuple


def sphere_objective(x: np.ndarray) -> float:
    return np.sum(x**2)


def sphere_constraints(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[0] - 1.5, -x[0] - 1.5,
        x[1] - 1.5, -x[1] - 1.5
    ])


def rosenbrock_objective(x: np.ndarray) -> float:
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_constraints(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[0] + x[1] - 2.0,
        -x[0] - x[1] + 0.5
    ])


P = 4e6
sigma_max = 9.5e6
rho = 7800
V_min = 0.4


def pressure_vessel_objective(x: np.ndarray) -> float:
    R, t = x
    return rho * (4/3 * np.pi * ((R + t)**3 - R**3))


def pressure_vessel_constraints(x: np.ndarray) -> np.ndarray:
    R, t = x
    volume = 4/3 * np.pi * R**3
    stress = P * R / (2 * t)
    return np.array([V_min - volume, stress - sigma_max])


def get_theoretical_optimum() -> Tuple[np.ndarray, float]:
    R_star = (V_min * 3/(4 * np.pi))**(1/3)
    t_star = P * R_star / (2 * sigma_max)
    m_star = pressure_vessel_objective(np.array([R_star, t_star]))
    return np.array([R_star, t_star]), m_star


def get_problems() -> Dict[str, Dict[str, Any]]:
    R_opt, m_opt = get_theoretical_optimum()
    return {
        'sphere': {
            'name': 'Сферическая функция',
            'objective': sphere_objective,
            'constraints': sphere_constraints,
            'bounds': np.array([[-2.0, 2.0], [-2.0, 2.0]]),
            'optimal': 0.0,
        },
        'rosenbrock': {
            'name': 'Функция Розенброка',
            'objective': rosenbrock_objective,
            'constraints': rosenbrock_constraints,
            'bounds': np.array([[-1.5, 1.5], [-0.5, 2.5]]),
            'optimal': 0.0,
        },
        'pressure_vessel': {
            'name': 'Сферический сосуд',
            'objective': pressure_vessel_objective,
            'constraints': pressure_vessel_constraints,
            'bounds': np.array([[0.4, 0.6], [0.05, 0.15]]),
            'optimal': m_opt,
        }
    }
def get_pressure_vessel_optimum():
    """Возвращает более точный оптимум."""
    from scipy.optimize import minimize
    
    def objective(x):
        return pressure_vessel_objective(x)
    
    def constraint1(x):
        return V_min - (4/3 * np.pi * x[0]**3)
    
    def constraint2(x):
        return P * x[0] / (2 * x[1]) - sigma_max
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: -constraint1(x)},
        {'type': 'ineq', 'fun': lambda x: -constraint2(x)}
    ]
    
    bounds = [(0.4, 0.6), (0.05, 0.15)]
    x0 = [0.46, 0.097]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return result.x, result.fun
