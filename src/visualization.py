"""
Visualization utilities for Bayesian optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_convergence(
    histories: Dict[str, List[np.ndarray]], title: str = "Convergence"
):
    """
    Plot convergence curves for different methods.

    Parameters
    ----------
    histories : dict
        Dictionary mapping method name to list of history arrays.
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 6))

    colors = {"CEI": "blue", "Penalty": "green", "Lagrange": "orange", "Barrier": "red"}

    for method, histories_list in histories.items():
        # Compute mean and std across trials
        min_len = min(len(h) for h in histories_list)
        truncated = [h[:min_len] for h in histories_list]

        mean = np.mean(truncated, axis=0)
        std = np.std(truncated, axis=0)

        iterations = range(1, len(mean) + 1)

        color = colors.get(method, "gray")
        plt.plot(iterations, mean, label=method, color=color, linewidth=2)
        plt.fill_between(iterations, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Feasible Objective", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph_convergence.png", dpi=150)
    plt.close()
    print("✓ Convergence plot saved to graph_convergence.png")


def plot_comparison(results: Dict, title: str = "Method Comparison"):
    """
    Plot bar chart comparing methods.

    Parameters
    ----------
    results : dict
        Results from experiment.
    title : str
        Plot title.
    """
    problems = list(results["problems"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean best objective
    ax1 = axes[0]
    methods = list(results["problems"][problems[0]].keys())

    x = np.arange(len(problems))
    width = 0.2

    for i, method in enumerate(methods):
        means = []
        stds = []
        for prob in problems:
            res = results["problems"][prob][method]
            means.append(res["mean_best"])
            stds.append(res["std_best"])

        offset = (i - len(methods) / 2) * width + width / 2
        ax1.bar(x + offset, means, width, label=method, yerr=stds, capsize=3)

    ax1.set_xlabel("Problem", fontsize=12)
    ax1.set_ylabel("Best Feasible Objective", fontsize=12)
    ax1.set_title("Mean Best Objective (lower is better)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(problems, rotation=45, ha="right")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Right: Success rate
    ax2 = axes[1]

    for i, method in enumerate(methods):
        success_rates = []
        for prob in problems:
            res = results["problems"][prob][method]
            success_rates.append(res["success_rate"] * 100)

        offset = (i - len(methods) / 2) * width + width / 2
        ax2.bar(x + offset, success_rates, width, label=method)

    ax2.set_xlabel("Problem", fontsize=12)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_title("Feasible Solution Found", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(problems, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("graph_comparison.png", dpi=150)
    plt.close()
    print("✓ Comparison plot saved to graph_comparison.png")


def plot_all_results(results: Dict):
    """Generate all plots from results."""
    # Prepare convergence data
    convergence_data = {}
    first_problem = list(results["problems"].keys())[0]

    for method, method_res in results["problems"][first_problem].items():
        if "history" in method_res:
            histories = method_res["history"]
            convergence_data[method] = histories

    if convergence_data:
        plot_convergence(convergence_data, f"Convergence on {first_problem}")

    plot_comparison(
        results, "Bayesian Optimization with Constraints - Method Comparison"
    )
