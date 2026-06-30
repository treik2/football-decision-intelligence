import numpy as np
from typing import Dict
from backend.ml.loader import load_model
import logging

logger = logging.getLogger(__name__)


class WinProbModel:
    """
    Wrapper around a trained LightGBM/XGBoost classifier.
    Falls back to an Elo+xG heuristic when no trained artifact exists.
    """

    version = "1.0.0"

    def __init__(self):
        self._model = load_model("win_prob")
        if self._model is None:
            logger.warning("Win prob model not found — using heuristic fallback")

    def _feature_vector(self, features: Dict) -> np.ndarray:
        return np.array([
            features.get("elo_diff", 0.0),
            features.get("xg_diff", 0.0),
            features.get("home_xg", 1.3),
            features.get("away_xg", 1.0),
            features.get("home_form", 0.5),
            features.get("away_form", 0.5),
            features.get("home_rest_days", 7),
            features.get("away_rest_days", 7),
            features.get("home_injury_score", 0.0),
            features.get("away_injury_score", 0.0),
            features.get("temperature", 15.0),
            features.get("rain_mm", 0.0),
            features.get("home_motivation", 0.5),
            features.get("away_motivation", 0.5),
        ]).reshape(1, -1)

    def _heuristic(self, features: Dict) -> Dict[str, float]:
        elo_diff = features.get("elo_diff", 0.0)
        xg_diff = features.get("xg_diff", 0.0)
        home_p = 0.45 + 0.002 * elo_diff + 0.03 * np.tanh(xg_diff)
        home_p = float(np.clip(home_p, 0.05, 0.90))
        draw_p = float(np.clip(0.25 * (1.0 - abs(np.tanh(xg_diff))), 0.08, 0.35))
        away_p = float(np.clip(1.0 - home_p - draw_p, 0.05, 0.90))
        total = home_p + draw_p + away_p
        return {
            "home": round(home_p / total, 4),
            "draw": round(draw_p / total, 4),
            "away": round(away_p / total, 4),
        }

    def predict(self, features: Dict) -> Dict[str, float]:
        if self._model is None:
            return self._heuristic(features)
        X = self._feature_vector(features)
        proba = self._model.predict_proba(X)[0]
        # classes assumed: [away, draw, home]
        return {
            "away": round(float(proba[0]), 4),
            "draw": round(float(proba[1]), 4),
            "home": round(float(proba[2]), 4),
        }
