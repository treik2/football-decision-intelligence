"""Goals model — Bivariate-independent Poisson baseline.
Expected goals (mu_home, mu_away) drive Poisson sampling.
In production load a trained Bayesian Poisson / Dixon-Coles model.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple

try:
    import joblib
    _GOALS_PATH = os.getenv("GOALS_MODEL_PATH", "ml/artifacts/goals_model.pkl")
    _goals_model = joblib.load(_GOALS_PATH) if os.path.exists(_GOALS_PATH) else None
except Exception:
    _goals_model = None


class GoalsModel:
    def expected_goals(self, features: Dict) -> Tuple[float, float]:
        if _goals_model is not None:
            return _goals_model.predict(features)
        # fallback: use raw xg features with slight home advantage
        mu_home = float(features.get("home_xg", 1.35)) * 1.05
        mu_away = float(features.get("away_xg", 1.05))
        return mu_home, mu_away

    def score_matrix(self, features: Dict, max_goals: int = 8) -> Dict[Tuple[int, int], float]:
        mu_h, mu_a = self.expected_goals(features)
        matrix: Dict[Tuple[int, int], float] = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                matrix[(h, a)] = float(poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a))
        total = sum(matrix.values())
        return {k: v / total for k, v in matrix.items()}

    def top_correct_scores(self, features: Dict, n: int = 10) -> list:
        matrix = self.score_matrix(features)
        return sorted([(f"{h}-{a}", p) for (h, a), p in matrix.items()], key=lambda x: -x[1])[:n]
