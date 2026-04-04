"""
Bayesian Optimization main loop with constraints.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time

from .surrogate_model import SurrogateModel
from .constraint_handling import get_handler
from .experimental_design import lhs_sample, ensure_feasible


@dataclass
class OptimizationResult:
    """Results of a Bayesian optimization run."""
    best_objective: float
    best_point: np.ndarray
    best_constraints: np.ndarray
    is_feasible: bool
    history: List[Dict]
    n_evaluations: int
    wall_time: float


class BayesianOptimization:
    """
    Bayesian Optimization with constraints.
    
    Supports multiple constraint handling methods:
    - CEI: Constrained Expected Improvement
    - Penalty: Penalty function method
    - Lagrange: Augmented Lagrangian
    - Barrier: Interior point barrier
    """
    
    def __init__(
        self,
        problem,
        method: str = "CEI",
        n_initial: int = 10,
        n_iterations: int = 50,
        random_seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        problem : Problem
            Optimization problem with objective and constraints.
        method : str
            Constraint handling method.
        n_initial : int
            Number of initial design points.
        n_iterations : int
            Number of BO iterations.
        random_seed : int, optional
            Random seed for reproducibility.
        """
        self.problem = problem
        self.method = method
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize surrogate models
        self.objective_model = SurrogateModel()
        self.constraint_models = []
        
        # Constraint handler
        self.handler = get_handler(method, problem)
        
        # Data storage
        self.X = None
        self.f = None
        self.g = None
        self.best_feasible_objective = np.inf
        self.best_feasible_point = None
        self.best_feasible_constraints = None
        
        self.history = []
    
    def _initialize_design(self) -> None:
        """Generate initial experimental design."""
        # Latin Hypercube Sampling
        X_init = lhs_sample(self.n_initial, self.problem.dim, self.problem.bounds)
        
        # Evaluate
        f_init = []
        g_init = []
        
        for x in X_init:
            f, g = self.problem.evaluate(x)
            f_init.append(f)
            g_init.append(g)
            
            # Update best feasible
            if self.problem.is_feasible(x) and f < self.best_feasible_objective:
                self.best_feasible_objective = f
                self.best_feasible_point = x.copy()
                self.best_feasible_constraints = g.copy()
        
        self.X = X_init
        self.f = np.array(f_init)
        self.g = np.array(g_init)
        
        # Fit surrogate models
        self._fit_models()
    
    def _fit_models(self) -> None:
        """Fit Gaussian Process models for objective and constraints."""
        # Objective model
        self.objective_model.fit(self.X, self.f)
        
        # Constraint models (one per constraint)
        n_constraints = self.g.shape[1]
        self.constraint_models = []
        for j in range(n_constraints):
            model = SurrogateModel()
            model.fit(self.X, self.g[:, j])
            self.constraint_models.append(model)
    
    def _acquisition_function(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Compute acquisition function values.
        
        Parameters
        ----------
        X_candidates : np.ndarray, shape (n_candidates, dim)
            Candidate points.
            
        Returns
        -------
        acq : np.ndarray, shape (n_candidates,)
            Acquisition values.
        """
        # Predict objective
        mean_f, var_f = self.objective_model.predict(X_candidates)
        
        # Predict constraints
        mean_g = np.zeros((len(X_candidates), len(self.constraint_models)))
        var_g = np.zeros_like(mean_g)
        
        for j, model in enumerate(self.constraint_models):
            m, v = model.predict(X_candidates)
            mean_g[:, j] = m
            var_g[:, j] = v
        
        # Compute acquisition using handler
        f_best = self.best_feasible_objective if self.best_feasible_point is not None else np.inf
        acq = self.handler.compute_acquisition(
            mean_f, var_f, mean_g, var_g, f_best
        )
        
        return acq
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Find point that maximizes acquisition function.
        
        Returns
        -------
        np.ndarray
            Next candidate point.
        """
        from scipy.optimize import minimize
        
        best_x = None
        best_acq = -np.inf
        
        # Multi-start optimization
        n_starts = 20
        candidates = lhs_sample(n_starts, self.problem.dim, self.problem.bounds)
        
        for x0 in candidates:
            # Local optimization
            res = minimize(
                lambda x: -self._acquisition_function(x.reshape(1, -1))[0],
                x0,
                bounds=self.problem.bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if res.success:
                acq_val = -res.fun
                if acq_val > best_acq:
                    best_acq = acq_val
                    best_x = res.x
        
        if best_x is None:
            # Fallback to random
            best_x = candidates[np.argmax(self._acquisition_function(candidates))]
        
        return best_x
    
    def optimize(self, verbose: bool = True) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Parameters
        ----------
        verbose : bool
            Print progress.
            
        Returns
        -------
        OptimizationResult
            Results of the optimization.
        """
        start_time = time.time()
        
        # Initial design
        if verbose:
            print(f"\n[BO] Method: {self.method}")
            print(f"[BO] Initial design ({self.n_initial} points)...")
        self._initialize_design()
        
        # Main loop
        for iteration in range(self.n_iterations):
            if verbose:
                print(f"[BO] Iteration {iteration+1}/{self.n_iterations}...", end=" ")
            
            # Find next point
            x_next = self._optimize_acquisition()
            
            # Evaluate
            f_next, g_next = self.problem.evaluate(x_next)
            
            # Update data
            self.X = np.vstack([self.X, x_next])
            self.f = np.hstack([self.f, f_next])
            self.g = np.vstack([self.g, g_next])
            
            # Update best feasible
            if self.problem.is_feasible(x_next) and f_next < self.best_feasible_objective:
                self.best_feasible_objective = f_next
                self.best_feasible_point = x_next.copy()
                self.best_feasible_constraints = g_next.copy()
                if verbose:
                    print(f"✓ New best: {f_next:.6f}", end=" ")
            
            # Update surrogate models
            self._fit_models()
            
            # Update handler parameters (if adaptive)
            if hasattr(self.handler, 'update_multipliers'):
                self.handler.update_multipliers(g_next)
            if hasattr(self.handler, 'update_penalty'):
                violations = np.maximum(0, g_next)
                self.handler.update_penalty(violations)
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'x': x_next.copy(),
                'f': f_next,
                'g': g_next.copy(),
                'best_f': self.best_feasible_objective,
                'is_feasible': self.problem.is_feasible(x_next),
            })
            
            if verbose:
                print(f"f={f_next:.4f}, feasible={self.problem.is_feasible(x_next)}")
        
        wall_time = time.time() - start_time
        
        if verbose:
            print(f"\n[BO] Completed in {wall_time:.2f}s")
            if self.best_feasible_point is not None:
                print(f"[BO] Best feasible objective: {self.best_feasible_objective:.6f}")
            else:
                print(f"[BO] No feasible point found!")
        
        return OptimizationResult(
            best_objective=self.best_feasible_objective,
            best_point=self.best_feasible_point if self.best_feasible_point is not None else self.X[0],
            best_constraints=self.best_feasible_constraints,
            is_feasible=self.best_feasible_point is not None,
            history=self.history,
            n_evaluations=len(self.f),
            wall_time=wall_time,
        )
