"""
Test problems for constrained Bayesian optimization.
Includes standard benchmarks and helper to create constrained versions.
"""

import numpy as np
from typing import Callable, Tuple, Optional


class Problem:
    """
    Base class for optimization problems with constraints.

    Attributes
    ----------
    dim : int
        Problem dimension.
    bounds : np.ndarray, shape (dim, 2)
        Lower and upper bounds for each variable.
    has_constraints : bool
        Whether the problem has constraints.
    """

    def __init__(self, dim: int, bounds: np.ndarray):
        self.dim = dim
        self.bounds = bounds
        self.has_constraints = True

    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimize."""
        raise NotImplementedError

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Constraint function.
        Returns array where each element <= 0 means constraint satisfied.
        """
        raise NotImplementedError

    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if a point satisfies all constraints."""
        g = self.constraints(x)
        return np.all(g <= tol)

    def evaluate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return (objective, constraints) tuple."""
        return self.objective(x), self.constraints(x)


class Sphere(Problem):
    """
    Sphere function with inequality constraints.

    f(x) = sum(x_i^2)
    Constraints:
        g1(x) = x_0 - 0.5 <= 0
        g2(x) = -x_0 - 0.5 <= 0
        g3(x) = sum(x_i) - 1 <= 0
    """

    def __init__(self, dim: int = 2):
        bounds = np.array([[-5.0, 5.0]] * dim)
        super().__init__(dim, bounds)

    def objective(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        g1 = x[0] - 0.5
        g2 = -x[0] - 0.5
        g3 = np.sum(x) - 1.0
        return np.array([g1, g2, g3])


class Rosenbrock(Problem):
    """
    Rosenbrock function with constraints.

    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    """

    def __init__(self, dim: int = 2):
        bounds = np.array([[-2.0, 2.0]] * dim)
        super().__init__(dim, bounds)

    def objective(self, x: np.ndarray) -> float:
        n = len(x)
        f = 0.0
        for i in range(n - 1):
            f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        # Linear constraint: sum(x_i) <= 1
        g = np.sum(x) - 1.0
        return np.array([g])


class Ackley(Problem):
    """
    Ackley function with constraints.

    f(x) = -20*exp(-0.2*sqrt(1/n sum(x_i^2))) - exp(1/n sum(cos(2π x_i))) + 20 + e
    """

    def __init__(self, dim: int = 2):
        bounds = np.array([[-5.0, 5.0]] * dim)
        super().__init__(dim, bounds)

    def objective(self, x: np.ndarray) -> float:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))

        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        return term1 + term2 + 20.0 + np.e

    def constraints(self, x: np.ndarray) -> np.ndarray:
        # Quadratic constraint: x_0^2 + x_1^2 <= 2
        g = x[0] ** 2 + x[1] ** 2 - 2.0
        return np.array([g])


def make_constrained_problem(
    objective: Callable,
    constraint_func: Callable,
    bounds: np.ndarray,
    name: str = "Custom",
) -> Problem:
    """
    Factory function to create a Problem from given functions.

    Parameters
    ----------
    objective : callable
        Objective function f(x) -> float.
    constraint_func : callable
        Constraint function g(x) -> array or float.
    bounds : np.ndarray, shape (dim, 2)
        Variable bounds.
    name : str
        Problem name.

    Returns
    -------
    Problem
        A Problem instance.
    """

    class CustomProblem(Problem):
        def __init__(self):
            super().__init__(bounds.shape[0], bounds)
            self._name = name

        def objective(self, x: np.ndarray) -> float:
            return objective(x)

        def constraints(self, x: np.ndarray) -> np.ndarray:
            g = constraint_func(x)
            if np.isscalar(g):
                return np.array([g])
            return np.array(g)

    return CustomProblem()
