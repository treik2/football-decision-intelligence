import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple


class GoalsModel:
    def __init__(self, params: Dict = None):
        self.params = params or {}

    def expected_goals(self, features: Dict) -> Tuple[float, float]:
        home_xg = features.get("home_xg", 1.35)
        away_xg = features.get("away_xg", 1.05)
        if self.params:
            mean = self.params.get("mean_goals", 1.2)
            ht = features.get("home_team", "")
            at = features.get("away_team", "")
            ha = self.params.get("home_advantage", 1.1)
            h_att = self.params.get(ht, {}).get("attack", 1.0)
            h_def = self.params.get(ht, {}).get("defence", 1.0)
            a_att = self.params.get(at, {}).get("attack", 1.0)
            a_def = self.params.get(at, {}).get("defence", 1.0)
            home_xg = h_att * a_def * mean * ha
            away_xg = a_att * h_def * mean
        return float(np.clip(home_xg, 0.1, 6.0)), float(np.clip(away_xg, 0.1, 6.0))

    def score_matrix(self, features: Dict, max_goals: int = 7) -> np.ndarray:
        mu_h, mu_a = self.expected_goals(features)
        m = np.outer(
            [poisson.pmf(i, mu_h) for i in range(max_goals + 1)],
            [poisson.pmf(j, mu_a) for j in range(max_goals + 1)],
        )
        rho = self.params.get("rho", -0.08)
        corrections = {(0,0): 1 - mu_h*mu_a*rho, (0,1): 1 + mu_h*rho, (1,0): 1 + mu_a*rho, (1,1): 1 - rho}
        for (h, a), tau in corrections.items():
            m[h, a] = max(m[h, a] * tau, 1e-10)
        m /= m.sum()
        return m
