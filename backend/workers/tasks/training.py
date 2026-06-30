import logging
from backend.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="backend.workers.tasks.training.trigger_training", bind=True)
def trigger_training(self, model_name: str = "win_prob"):
    """
    Trigger retraining for a named model.
    In production: call your MLFlow/training pipeline.
    This task acts as the entry point from the scheduler.
    """
    logger.info(f"Training triggered for model: {model_name}")
    try:
        if model_name == "win_prob":
            from ml.training.train_win_prob import train
            result = train()
        elif model_name == "goals":
            from ml.training.train_goals import train
            result = train()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        logger.info(f"Training complete for {model_name}: {result}")
        return result
    except Exception as exc:
        logger.error(f"Training task failed: {exc}")
        raise
