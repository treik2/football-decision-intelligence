"""Win probability model wrapper.
In production: load a trained LightGBM / CatBoost model via joblib.
Fallback: Elo + xG logistic estimate so the API is always runnable.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict

try:
    import joblib
    _MODEL_PATH = os.getenv("WIN_MODEL_PATH", "ml/artifacts/win_prob_lgbm.pkl")
    _model = joblib.load(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else None
except Exception:
    _model = None


FEATURE_ORDER = [
    "elo_diff", "xg_diff", "home_xg", "away_xg",
    "home_form", "away_form", "home_rest_days", "away_rest_days",
    "home_injuries", "away_injuries", "is_neutral",
    "temp_c", "wind_mps", "rain_mm",
]


def _fallback_predict(features: Dict) -> Dict[str, float]:
    """Simple logistic estimate using Elo + xG when model file is absent."""
    elo_diff = features.get("elo_diff", 0.0)
    xg_diff = features.get("xg_diff", 0.0)
    raw = 0.45 + 0.0018 * elo_diff + 0.04 * np.tanh(xg_diff)
    home_p = float(np.clip(raw, 0.05, 0.90))
    draw_p = float(np.clip(0.26 - 0.10 * abs(np.tanh(xg_diff)), 0.08, 0.35))
    away_p = max(0.05, 1.0 - home_p - draw_p)
    total = home_p + draw_p + away_p
    return {"home": home_p / total, "draw": draw_p / total, "away": away_p / total}


class WinProbModel:
    def predict(self, features: Dict) -> Dict[str, float]:
        if _model is not None:
            row = np.array([[features.get(f, 0.0) for f in FEATURE_ORDER]])
            proba = _model.predict_proba(row)[0]  # shape (3,) [away, draw, home]
            return {"home": float(proba[2]), "draw": float(proba[1]), "away": float(proba[0])}
        return _fallback_predict(features)
