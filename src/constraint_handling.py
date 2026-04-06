"""
Constraint handling methods for Bayesian optimization.
Implements CEI, Penalty, Lagrange, and Barrier methods.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional, Callable


class ConstraintHandler:
    """
    Base class for constraint handling.
    """

    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim
        self.bounds = problem.bounds

    def compute_acquisition(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        constraints_mean: np.ndarray,
        constraints_var: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """Compute acquisition function values."""
        raise NotImplementedError


class CEIHandler(ConstraintHandler):
    """
    Constrained Expected Improvement (CEI).

    CEI = EI(x) * P(feasible(x))
    """

    def compute_acquisition(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        constraints_mean: np.ndarray,
        constraints_var: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """
        Compute CEI values.

        Parameters
        ----------
        mean : array, shape (n_points,)
            GP mean predictions for objective.
        var : array, shape (n_points,)
            GP variance predictions for objective.
        constraints_mean : array, shape (n_points, n_constraints)
            GP mean predictions for each constraint.
        constraints_var : array, shape (n_points, n_constraints)
            GP variance predictions for each constraint.
        f_best : float
            Best feasible objective value so far.

        Returns
        -------
        cei : array, shape (n_points,)
            CEI values.
        """
        # Expected Improvement
        sigma = np.sqrt(var)
        z = (f_best - mean) / (sigma + 1e-12)
        ei = (f_best - mean) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0

        # Probability of feasibility
        p_feas = np.ones(len(mean))
        for j in range(constraints_mean.shape[1]):
            g_mean = constraints_mean[:, j]
            g_var = constraints_var[:, j]
            g_sigma = np.sqrt(g_var)
            # P(g <= 0)
            p_j = norm.cdf(-g_mean / (g_sigma + 1e-12))
            p_feas *= p_j

        return ei * p_feas


class PenaltyHandler(ConstraintHandler):
    """
    Penalty method: f_penalty = f + ρ * sum(max(0, g))^2
    """

    def __init__(self, problem, penalty_coef: float = 1e3):
        super().__init__(problem)
        self.penalty_coef = penalty_coef
        self._adapt_penalty = True
        self._violation_history = []

    def compute_acquisition(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        constraints_mean: np.ndarray,
        constraints_var: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """
        Compute EI on penalized objective.
        """
        # Compute expected penalized objective
        # E[penalty] = E[f] + ρ * E[sum(max(0,g)^2)]
        penalty_mean = np.zeros(len(mean))

        for j in range(constraints_mean.shape[1]):
            g_mean = constraints_mean[:, j]
            g_var = constraints_var[:, j]
            g_sigma = np.sqrt(g_var)

            # Expected violation: E[max(0, g)^2] ≈ (g_mean^2 + g_var) * P(g > 0)
            z = g_mean / (g_sigma + 1e-12)
            p_pos = 1 - norm.cdf(z)

            # Approximate expectation
            expected_violation_sq = (g_mean**2 + g_var) * p_pos
            penalty_mean += expected_violation_sq

        penalized_mean = mean + self.penalty_coef * penalty_mean

        # Adapt penalty coefficient
        if self._adapt_penalty and len(self._violation_history) > 5:
            avg_violation = np.mean(self._violation_history[-5:])
            if avg_violation > 0.1:
                self.penalty_coef *= 1.2
            elif avg_violation < 0.01:
                self.penalty_coef *= 0.9

        # EI on penalized objective
        sigma = np.sqrt(var)
        z = (f_best - penalized_mean) / (sigma + 1e-12)
        ei = (f_best - penalized_mean) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0

        return ei

    def update_penalty(self, violations: np.ndarray):
        """Update penalty coefficient based on observed violations."""
        avg_violation = np.mean(violations)
        self._violation_history.append(avg_violation)
        if len(self._violation_history) > 20:
            self._violation_history.pop(0)


class LagrangeHandler(ConstraintHandler):
    """
    Lagrange multiplier method: f + Σ λ_j * max(0, g_j)
    """

    def __init__(self, problem, lambda_init: float = 1.0):
        super().__init__(problem)
        self.lambdas = (
            np.ones(problem.constraints(np.zeros(problem.dim)).shape[0]) * lambda_init
        )

    def compute_acquisition(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        constraints_mean: np.ndarray,
        constraints_var: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """
        Compute EI on Lagrangian.
        """
        # Expected Lagrangian: E[f] + Σ λ_j * E[max(0, g_j)]
        lagrangian_mean = mean.copy()

        for j, lam in enumerate(self.lambdas):
            g_mean = constraints_mean[:, j]
            g_var = constraints_var[:, j]
            g_sigma = np.sqrt(g_var)

            # E[max(0, g)] ≈ g_mean * P(g > 0) + g_sigma * pdf(z)
            z = g_mean / (g_sigma + 1e-12)
            p_pos = 1 - norm.cdf(z)
            expected_violation = g_mean * p_pos + g_sigma * norm.pdf(z)
            lagrangian_mean += lam * expected_violation

        # EI on Lagrangian
        sigma = np.sqrt(var)
        z = (f_best - lagrangian_mean) / (sigma + 1e-12)
        ei = (f_best - lagrangian_mean) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0

        return ei

    def update_multipliers(self, constraints: np.ndarray, step: float = 0.1):
        """Update Lagrange multipliers based on constraint violations."""
        violations = np.maximum(0, constraints)
        self.lambdas += step * violations
        self.lambdas = np.maximum(0, self.lambdas)


class BarrierHandler(ConstraintHandler):
    """
    Barrier method: f - μ * Σ log(-g_j) for interior points.
    """

    def __init__(self, problem, mu: float = 1.0, barrier_tol: float = 1e-6):
        super().__init__(problem)
        self.mu = mu
        self.barrier_tol = barrier_tol

    def compute_acquisition(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        constraints_mean: np.ndarray,
        constraints_var: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        """
        Compute EI with barrier for infeasible points.
        """
        # Standard EI
        sigma = np.sqrt(var)
        z = (f_best - mean) / (sigma + 1e-12)
        ei = (f_best - mean) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0

        # Barrier penalty for predicted infeasibility
        barrier_penalty = np.zeros(len(mean))
        for j in range(constraints_mean.shape[1]):
            g_mean = constraints_mean[:, j]
            g_var = constraints_var[:, j]
            g_sigma = np.sqrt(g_var)

            # Probability of violation
            z = g_mean / (g_sigma + 1e-12)
            p_violation = 1 - norm.cdf(-z)  # P(g > 0)

            # Expected barrier term
            # For points likely infeasible, add penalty
            barrier_penalty += (
                -self.mu * np.log(-g_mean + self.barrier_tol) * (1 - p_violation)
            )

        # Points with barrier_penalty > 0 are interior, others get negative penalty
        barrier_penalty = np.maximum(barrier_penalty, 0)

        return ei / (1 + barrier_penalty / self.mu)


def get_handler(name: str, problem) -> ConstraintHandler:
    """
    Factory function for constraint handlers.

    Parameters
    ----------
    name : str
        Handler name: 'CEI', 'Penalty', 'Lagrange', 'Barrier'
    problem : Problem
        Optimization problem.

    Returns
    -------
    ConstraintHandler
        Handler instance.
    """
    handlers = {
        "CEI": CEIHandler,
        "Penalty": PenaltyHandler,
        "Lagrange": LagrangeHandler,
        "Barrier": BarrierHandler,
    }

    if name not in handlers:
        raise ValueError(
            f"Unknown handler: {name}. Choose from {list(handlers.keys())}"
        )

    return handlers[name](problem)
