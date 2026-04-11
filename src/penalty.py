"""Штрафной метод обработки ограничений."""

from typing import Optional

import numpy as np

from .base import ConstraintHandler


class PenaltyHandler(ConstraintHandler):
    def __init__(self, penalty_coef: float = 1e4, random_state: Optional[int] = None) -> None:
        super().__init__(random_state)
        self.penalty_coef = penalty_coef

    def acquisition(self, X: np.ndarray, f_min: float) -> np.ndarray:
        mu_f, _ = self.predict_objective(X)
        mu_g, _ = self.predict_constraints(X)

        violations = np.maximum(0, mu_g)
        total_penalty = self.penalty_coef * np.sum(violations, axis=1)

        return -(mu_f + total_penalty)

    def get_method_name(self) -> str:
        return "Penalty"
