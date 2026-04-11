"""Module for running experiments."""

import time
from typing import Any

import numpy as np

from src.bayesian_optimization import (
    CEIBayesianOptimization,
    PenaltyBayesianOptimization,
    LagrangeBayesianOptimization,
    BarrierBayesianOptimization,
)
from src.utils.types import OptimizationProblem, TrialResult


def is_feasible(x: np.ndarray, constraints: list, tolerance: float = 1e-6) -> bool:
    """Check if point satisfies all constraints."""
    return all(c(x) <= tolerance for c in constraints)


def generate_feasible_point(problem: OptimizationProblem, max_attempts: int = 5000) -> np.ndarray | None:
    """Generate a feasible point using Monte Carlo method."""
    bounds_array = np.array(problem.bounds)
    
    for _ in range(max_attempts):
        x = np.random.uniform(bounds_array[:, 0], bounds_array[:, 1])
        if is_feasible(x, problem.constraints):
            return x
    return None


def run_single_trial(
    problem: OptimizationProblem,
    method_name: str,
    n_init: int,
    n_iter: int,
    random_state: int,
    verbose: bool = True,
) -> TrialResult:
    """Run a single optimization trial."""
    np.random.seed(random_state)
    
    # Generate feasible initial points
    X_init = []
    y_init = []
    
    if verbose:
        print(f"      Generating {n_init} feasible points...", end=" ")
    
    max_total_attempts = 50000
    
    while len(X_init) < n_init and len(X_init) * 1000 < max_total_attempts:
        x = generate_feasible_point(problem, max_attempts=1000)
        if x is not None:
            y_val = problem.objective(x)
            X_init.append(x)
            y_init.append(y_val)
    
    if len(X_init) == 0:
        print(f"NO FEASIBLE POINTS FOUND!")
        return TrialResult(
            best_values=[np.inf] * (n_iter + 1),
            time=0,
            final_point=np.zeros(problem.dim),
            feasibility=False,
        )
    
    X_init = np.array(X_init)
    y_init = np.array(y_init)
    
    if verbose:
        print(f"found {len(X_init)} points")
    
    # Initialize optimizer
    if method_name == "CEI":
        optimizer = CEIBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
        )
    elif method_name == "Penalty":
        optimizer = PenaltyBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
        )
    elif method_name == "Lagrange":
        optimizer = LagrangeBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
        )
    elif method_name == "Barrier":
        optimizer = BarrierBayesianOptimization(
            bounds=problem.bounds,
            constraints=problem.constraints,
            n_init=n_init,
            kernel="matern",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    optimizer.initialize(X_init, y_init)
    
    best_value = np.min(y_init)
    best_values_history = [best_value]
    
    start_time = time.time()
    
    for iteration in range(n_iter):
        x_next = optimizer.suggest_next_point()
        y_next = problem.objective(x_next)
        optimizer.update(x_next, y_next)
        
        if is_feasible(x_next, problem.constraints) and y_next < best_value:
            best_value = y_next
        
        best_values_history.append(best_value)
        
        if verbose and (iteration + 1) % 15 == 0:
            distance = abs(best_value - problem.optimal_value)
            print(f"      Iter {iteration + 1}/{n_iter}: best={best_value:.6f}, dist={distance:.6f}")
    
    elapsed_time = time.time() - start_time
    
    return TrialResult(
        best_values=best_values_history,
        time=elapsed_time,
        final_point=X_init[np.argmin(y_init)],
        feasibility=True,
    )


def run_experiments(
    problems: list[OptimizationProblem],
    n_trials: int,
    n_init: int,
    n_iter: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run full experiment on all problems."""
    methods = ["CEI", "Penalty", "Lagrange", "Barrier"]
    all_results = {}
    
    for prob_idx, problem in enumerate(problems):
        print(f"\n{'='*80}")
        print(f"PROBLEM {prob_idx + 1}/{len(problems)}: {problem.name}")
        print(f"{'='*80}")
        print(f"   Dim: {problem.dim}, Optimum: {problem.optimal_value:.6f}")
        print(f"   Constraints: {len(problem.constraints)}")
        
        problem_results = {}
        
        for method_idx, method in enumerate(methods):
            print(f"\n   {method} ({method_idx + 1}/{len(methods)})")
            
            all_best_values = []
            all_times = []
            
            for trial in range(n_trials):
                print(f"      Trial {trial + 1}/{n_trials}")
                random_state = 42 + prob_idx * 100 + trial * 10
                
                trial_result = run_single_trial(
                    problem=problem,
                    method_name=method,
                    n_init=n_init,
                    n_iter=n_iter,
                    random_state=random_state,
                    verbose=True,
                )
                
                all_best_values.append(trial_result.best_values)
                all_times.append(trial_result.time)
                
                final_val = trial_result.best_values[-1]
                print(f"      Done in {trial_result.time:.1f}s, final={final_val:.6f}")
            
            best_values_matrix = np.array(all_best_values)
            
            problem_results[method] = {
                "best_values": best_values_matrix,
                "times": np.array(all_times),
                "mean_best": np.mean(best_values_matrix, axis=0),
                "std_best": np.std(best_values_matrix, axis=0),
            }
            
            final_values = best_values_matrix[:, -1]
            mean_dist = np.mean(np.abs(final_values - problem.optimal_value))
            print(f"\n      {method}: mean distance = {mean_dist:.6f}\n")
        
        all_results[problem.name] = problem_results
    
    return all_results