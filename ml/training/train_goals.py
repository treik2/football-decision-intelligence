"""Train Poisson goals model (home xG and away xG regressors)."""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURES = [
    "home_xg", "away_xg", "elo_diff",
    "home_form", "away_form", "home_rest_days", "away_rest_days",
]
TARGETS = ["home_score", "away_score"]
MODEL_PATH = os.getenv("MODEL_DIR", "backend/ml/artifacts")


def load_data(path: str = "data/matches.parquet") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
    df = pd.read_parquet(path)
    return df.dropna(subset=FEATURES + TARGETS)


def train(data_path: str = "data/matches.parquet") -> dict:
    logger.info("Loading training data for goals model...")
    df = load_data(data_path)

    X = df[FEATURES].values
    y = df[TARGETS].values

    base = PoissonRegressor(alpha=0.1, max_iter=500)
    model = MultiOutputRegressor(base, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        model.fit(X[tr_idx], y[tr_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])
        mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        logger.info(f"Fold {fold+1} MAE: {mae:.4f}")

    oof_mae = mean_absolute_error(y, oof_preds)
    logger.info(f"OOF MAE: {oof_mae:.4f}")

    # Retrain on full dataset
    model.fit(X, y)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_PATH, "goals.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Goals model saved to {model_file}")

    return {"oof_mae": oof_mae, "model_path": model_file}


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "data/matches.parquet"
    result = train(data)
    print(result)
