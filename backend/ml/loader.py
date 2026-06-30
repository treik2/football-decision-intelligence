"""Model artifact loader — loads from disk or MLFlow model registry."""
from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)


def load_artifact(path: str):
    import joblib
    if not os.path.exists(path):
        logger.warning("Model artifact not found at %s — using fallback.", path)
        return None
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def load_from_mlflow(model_name: str, stage: str = "Production"):
    """Load model from MLFlow registry. Requires MLFLOW_TRACKING_URI env."""
    try:
        import mlflow.pyfunc
        uri = f"models:/{model_name}/{stage}"
        logger.info("Loading MLFlow model: %s", uri)
        return mlflow.pyfunc.load_model(uri)
    except Exception as e:
        logger.error("MLFlow load failed: %s", e)
        return None
