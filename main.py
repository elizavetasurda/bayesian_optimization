#!/usr/bin/env python3
"""Main module for running Bayesian optimization experiments."""

import warnings
from pathlib import Path

import numpy as np

from src.test_problems.bbob_constrained import create_bbob_problem_set
from src.utils.experiment import run_experiments
from src.utils.result_saver import save_results, save_convergence_plots
from src.utils.types import OptimizationProblem

warnings.filterwarnings("ignore")
np.random.seed(42)


def main() -> None:
    """Run the main experiment."""
    # Create test problems
    problems: list[OptimizationProblem] = create_bbob_problem_set()

    # OPTIMAL BALANCE: speed + quality
    n_trials: int = 5      # enough for statistics (was 10)
    n_init: int = 15       # balanced (was 20)
    n_iter: int = 60       # enough for convergence (was 100)

    print("=" * 80)
    print("RUNNING EXPERIMENTS (OPTIMIZED FOR SPEED & QUALITY)")
    print("=" * 80)
    print(f"Problems: {len(problems)}")
    print(f"Trials per problem: {n_trials}")
    print(f"Initial design (LHS): {n_init} points")
    print(f"Optimization iterations: {n_iter}")
    print(f"Expected time: ~2-5 minutes per method")
    print("=" * 80)

    # Run experiments
    results = run_experiments(
        problems=problems,
        n_trials=n_trials,
        n_init=n_init,
        n_iter=n_iter,
    )

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save text results
    save_results(
        results=results,
        filepath=results_dir / "experiment_results.txt",
        problems=problems,
    )

    # Save convergence plots
    save_convergence_plots(
        results=results,
        output_dir=results_dir,
        n_init=n_init,
        problems=problems,
    )

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (distance to optimum)")
    print("=" * 80)

    for problem in problems:
        if problem.name not in results:
            continue

        print(f"\n{problem.name}:")
        print(f"   Optimum: {problem.optimal_value:.6f}")

        prob_results = results[problem.name]

        for method_name, method_results in prob_results.items():
            final_values = method_results["best_values"][:, -1]
            distances = np.abs(final_values - problem.optimal_value)

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            best_dist = np.min(distances)

            print(f"   {method_name:10s}: "
                  f"mean = {mean_dist:.6e} +/- {std_dist:.6e}, "
                  f"best = {best_dist:.6e}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"Results saved in: {results_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()