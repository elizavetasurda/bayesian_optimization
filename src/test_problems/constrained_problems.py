"""Legacy module for constrained problems. Use bbob_constrained instead."""

from src.test_problems.bbob_constrained import (
    BBOBConstrainedProblems,
    create_bbob_problem_set,
    get_problem_by_index,
)

__all__ = [
    "BBOBConstrainedProblems",
    "create_bbob_problem_set",
    "get_problem_by_index",
]