"""
Experiment runner for comparing constraint handling methods.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

from .problems import Sphere, Rosenbrock, Ackley, make_constrained_problem
from .optimizer import BayesianOptimization


def run_experiments(
    dimension: int = 2,
    n_trials: int = 3,
    n_initial: int = 10,
    n_iterations: int = 50,
    methods: List[str] = None,
    use_coco: bool = True,
) -> Dict:
    """
    Run experiments comparing constraint handling methods.

    Parameters
    ----------
    dimension : int
        Problem dimension.
    n_trials : int
        Number of independent trials per problem/method.
    n_initial : int
        Initial design size.
    n_iterations : int
        Number of BO iterations.
    methods : list
        Constraint handling methods to compare.
    use_coco : bool
        Whether to use COCO bbob-constrained problems.

    Returns
    -------
    dict
        Results dictionary.
    """
    if methods is None:
        methods = ["CEI", "Penalty", "Lagrange", "Barrier"]

    # Define test problems
    problems = []
    problem_names = []

    # Built-in problems
    problems.append(Sphere(dim=dimension))
    problem_names.append("Sphere")

    problems.append(Rosenbrock(dim=dimension))
    problem_names.append("Rosenbrock")

    problems.append(Ackley(dim=dimension))
    problem_names.append("Ackley")

    # Try to add COCO problems
    if use_coco:
        try:
            from .bbob_wrapper import BBOBSuite

            coco_suite = BBOBSuite(dimension=dimension, instances=[1])

            # Add first 3 COCO problems for time reasons
            for i, prob_desc in enumerate(coco_suite.problems[:3]):
                # Create wrapper
                class COCOProblemWrapper:
                    def __init__(self, suite, prob_desc):
                        self.suite = suite
                        self.prob_desc = prob_desc
                        self.dim = prob_desc.dimension
                        self.bounds = prob_desc.bounds
                        self.has_constraints = True

                    def objective(self, x):
                        f, _ = self.suite.evaluate(self.prob_desc, x)
                        return f

                    def constraints(self, x):
                        _, g = self.suite.evaluate(self.prob_desc, x)
                        return g

                    def evaluate(self, x):
                        return self.suite.evaluate(self.prob_desc, x)

                    def is_feasible(self, x, tol=1e-8):
                        g = self.constraints(x)
                        return np.all(g <= tol)

                wrapped = COCOProblemWrapper(coco_suite, prob_desc)
                problems.append(wrapped)
                problem_names.append(prob_desc.function_name)
        except Exception as e:
            print(f"Warning: Could not load COCO problems: {e}")
            print("Using only built-in problems.")

    results = {
        "config": {
            "dimension": dimension,
            "n_trials": n_trials,
            "n_initial": n_initial,
            "n_iterations": n_iterations,
            "methods": methods,
        },
        "problems": {},
    }

    for prob, prob_name in zip(problems, problem_names):
        print(f"\n{'='*60}")
        print(f"Problem: {prob_name}")
        print(f"{'='*60}")

        prob_results = {}

        for method in methods:
            print(f"\n--- Method: {method} ---")

            trial_results = []
            for trial in range(n_trials):
                print(f"  Trial {trial+1}/{n_trials}...", end=" ")

                # Run optimization
                optimizer = BayesianOptimization(
                    problem=prob,
                    method=method,
                    n_initial=n_initial,
                    n_iterations=n_iterations,
                    random_seed=42 + trial,
                )

                result = optimizer.optimize(verbose=False)
                trial_results.append(result)

                print(
                    f"best={result.best_objective:.4f}, feasible={result.is_feasible}"
                )

            # Aggregate results
            best_values = [r.best_objective for r in trial_results if r.is_feasible]
            if best_values:
                mean_best = np.mean(best_values)
                std_best = np.std(best_values)
                success_rate = len(best_values) / n_trials
            else:
                mean_best = np.inf
                std_best = 0
                success_rate = 0

            prob_results[method] = {
                "mean_best": mean_best,
                "std_best": std_best,
                "success_rate": success_rate,
                "trials": [
                    {
                        "best": r.best_objective,
                        "feasible": r.is_feasible,
                        "n_evals": r.n_evaluations,
                        "time": r.wall_time,
                    }
                    for r in trial_results
                ],
                "history": [[h["best_f"] for h in r.history] for r in trial_results],
            }

            print(
                f"  Mean best: {mean_best:.4f} ± {std_best:.4f}, Success: {success_rate:.0%}"
            )

        results["problems"][prob_name] = prob_results

    # Save results to file
    output_file = Path("results.txt")
    with open(output_file, "w") as f:
        f.write("Bayesian Optimization with Constraints - Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dimension: {dimension}\n")
        f.write(f"Trials per problem: {n_trials}\n")
        f.write(f"Initial points: {n_initial}\n")
        f.write(f"BO iterations: {n_iterations}\n\n")

        for prob_name, prob_res in results["problems"].items():
            f.write(f"\n{prob_name}\n")
            f.write("-" * 40 + "\n")
            for method, res in prob_res.items():
                f.write(f"  {method}:\n")
                f.write(f"    Mean best: {res['mean_best']:.6f}\n")
                f.write(f"    Std: {res['std_best']:.6f}\n")
                f.write(f"    Success rate: {res['success_rate']:.0%}\n")
            f.write("\n")

    print(f"\n✓ Results saved to {output_file}")

    return results
