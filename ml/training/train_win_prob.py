"""Train LightGBM win probability classifier."""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURES = [
    "elo_diff", "xg_diff", "home_xg", "away_xg",
    "home_form", "away_form", "home_rest_days", "away_rest_days",
    "home_injury_score", "away_injury_score",
    "temperature", "rain_mm", "home_motivation", "away_motivation",
]
TARGET = "result"  # 0=away win, 1=draw, 2=home win
MODEL_PATH = os.getenv("MODEL_DIR", "backend/ml/artifacts")


def load_data(path: str = "data/matches.parquet") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
    df = pd.read_parquet(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df[TARGET] = np.select(conditions, [2, 1, 0], default=np.nan)
    return df.dropna(subset=[TARGET])


def train(data_path: str = "data/matches.parquet") -> dict:
    logger.info("Loading training data...")
    df = load_data(data_path)
    df = encode_target(df)

    X = df[FEATURES].values
    y = df[TARGET].astype(int).values

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y), 3))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_probs[val_idx] = model.predict_proba(X[val_idx])
        models.append(model)
        logger.info(f"Fold {fold+1} logloss: {log_loss(y[val_idx], oof_probs[val_idx]):.4f}")

    oof_logloss = log_loss(y, oof_probs)
    logger.info(f"OOF log loss: {oof_logloss:.4f}")

    # Calibrate best fold model
    best_model = models[0]
    calibrated = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
    calibrated.fit(X, y)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_PATH, "win_prob.pkl")
    joblib.dump(calibrated, model_file)
    logger.info(f"Model saved to {model_file}")

    return {"oof_logloss": oof_logloss, "model_path": model_file}


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "data/matches.parquet"
    result = train(data)
    print(result)
