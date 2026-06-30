import numpy as np
from typing import Dict, Optional


class WinProbModel:
    FEATURE_COLS = [
        "elo_diff", "xg_diff", "home_xg", "away_xg",
        "home_form", "away_form", "home_rest_days", "away_rest_days",
        "home_injury_score", "away_injury_score", "home_advantage",
        "motivation_diff", "weather_temp", "weather_wind",
    ]

    def __init__(self, clf=None):
        self.clf = clf

    def _to_vector(self, features: Dict) -> np.ndarray:
        return np.array([features.get(c, 0.0) for c in self.FEATURE_COLS], dtype=np.float32).reshape(1, -1)

    def predict_proba(self, features: Dict) -> Dict[str, float]:
        if self.clf is not None:
            vec = self._to_vector(features)
            p = self.clf.predict_proba(vec)[0]
            return {"home": float(p[2]), "draw": float(p[1]), "away": float(p[0])}
        return self._fallback(features)

    def _fallback(self, features: Dict) -> Dict[str, float]:
        elo = features.get("elo_diff", 0.0)
        xgd = features.get("xg_diff", 0.0)
        ha = features.get("home_advantage", 0.1)
        h = float(np.clip(0.45 + 0.0015 * elo + 0.04 * np.tanh(xgd) + ha * 0.05, 0.10, 0.85))
        d = float(np.clip(0.26 * (1.0 - 0.5 * abs(np.tanh(xgd))), 0.08, 0.38))
        a = float(np.clip(1.0 - h - d, 0.05, 0.85))
        t = h + d + a
        return {"home": h/t, "draw": d/t, "away": a/t}
