import os
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "backend/ml/artifacts"))


def load_model(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        logger.warning(f"Model artifact {path} not found, using fallback estimator")
        return None
    return joblib.load(path)


def save_model(model, name: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
