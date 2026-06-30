import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple
from backend.ml.loader import load_model
import logging

logger = logging.getLogger(__name__)


class GoalsModel:
    """
    Bivariate Poisson goals model.
    Uses trained lambda estimators or falls back to xG-based Poisson.
    """

    def __init__(self):
        self._model = load_model("goals")
        if self._model is None:
            logger.warning("Goals model not found — using xG Poisson fallback")

    def expected_goals(self, features: Dict) -> Tuple[float, float]:
        if self._model is not None:
            X = np.array([[
                features.get("home_xg", 1.3),
                features.get("away_xg", 1.0),
                features.get("elo_diff", 0.0),
                features.get("home_form", 0.5),
                features.get("away_form", 0.5),
                features.get("home_rest_days", 7),
                features.get("away_rest_days", 7),
            ]])
            preds = self._model.predict(X)[0]
            return float(max(preds[0], 0.1)), float(max(preds[1], 0.1))
        # fallback: direct xG + slight home boost
        home_xg = float(features.get("home_xg", 1.3)) * 1.05
        away_xg = float(features.get("away_xg", 1.0))
        return home_xg, away_xg

    def score_matrix(self, features: Dict, max_goals: int = 8) -> Dict[Tuple[int, int], float]:
        mu_h, mu_a = self.expected_goals(features)
        matrix = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                matrix[(h, a)] = float(poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a))
        total = sum(matrix.values())
        return {k: v / total for k, v in matrix.items()}
