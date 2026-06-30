from backend.workers.celery_app import celery_app
from backend.db.base import SessionLocal
from backend.db import models
from backend.features.engineering import compute_match_features
from backend.features.store import FeatureStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)
fs = FeatureStore()


@celery_app.task(name="backend.workers.tasks_features.materialize_all_features", bind=True, max_retries=2)
def materialize_all_features(self):
    """Recompute and materialize features for all upcoming scheduled matches."""
    db = SessionLocal()
    try:
        matches = db.query(models.Match).filter(models.Match.status == "scheduled").all()
        for match in matches:
            try:
                features = compute_match_features(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    context={},
                )
                fs.materialize(match.id, features)
                logger.info(f"Features materialized for match {match.id}")
            except Exception as e:
                logger.warning(f"Feature error for {match.id}: {e}")
    finally:
        db.close()
