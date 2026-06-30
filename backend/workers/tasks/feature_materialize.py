import logging
from backend.workers.celery_app import celery_app
from backend.features.engineering import FeatureEngineer
from backend.features.store import FeatureStore

logger = logging.getLogger(__name__)
engineer = FeatureEngineer()
store = FeatureStore()


@celery_app.task(name="backend.workers.tasks.feature_materialize.materialize_all", bind=True, max_retries=2)
def materialize_all(self):
    """
    For all scheduled matches in the store that lack computed features,
    trigger feature engineering and re-materialize.
    """
    try:
        all_keys = store.all_keys()
        refreshed = 0
        for match_id in all_keys:
            data = store.get_safe(match_id) or {}
            if not data.get("features_computed"):
                try:
                    features = engineer.compute_from_cache(match_id, data)
                    data.update(features)
                    data["features_computed"] = True
                    store.set(match_id, data)
                    refreshed += 1
                except Exception as e:
                    logger.warning(f"Feature computation failed for {match_id}: {e}")
        logger.info(f"Materialized features for {refreshed} matches")
        return {"refreshed": refreshed}
    except Exception as exc:
        logger.error(f"Materialize task failed: {exc}")
        raise self.retry(exc=exc, countdown=120)
