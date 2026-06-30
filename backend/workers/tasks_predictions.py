from backend.workers.celery_app import celery_app
from backend.db.base import SessionLocal
from backend.db import models
from backend.features.store import FeatureStore
from backend.ml.model_loader import ModelLoader
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.utils.logger import get_logger
from datetime import datetime
import uuid

logger = get_logger(__name__)
fs = FeatureStore()
loader = ModelLoader()
mc = MonteCarloSimulator()


@celery_app.task(name="backend.workers.tasks_predictions.predict_upcoming_matches", bind=True, max_retries=2)
def predict_upcoming_matches(self):
    """Run predictions for all scheduled matches that don't yet have a prediction."""
    db = SessionLocal()
    try:
        matches = db.query(models.Match).filter(models.Match.status == "scheduled").all()
        predicted_ids = {r.match_id for r in db.query(models.Prediction.match_id).all()}

        for match in matches:
            if match.id in predicted_ids:
                continue
            try:
                features = fs.get(match.id)
                win_model = loader.get_win_model()
                goals_model = loader.get_goals_model()
                win_probs = win_model.predict_proba(features)
                home_xg, away_xg = goals_model.expected_goals(features)
                sim = mc.simulate(home_xg=home_xg, away_xg=away_xg, n=100_000)

                prediction = models.Prediction(
                    id=str(uuid.uuid4()),
                    match_id=match.id,
                    home_win_prob=win_probs["home"],
                    draw_prob=win_probs["draw"],
                    away_win_prob=win_probs["away"],
                    home_xg=home_xg,
                    away_xg=away_xg,
                    over25_prob=sim["over_2_5"],
                    btts_prob=sim["btts"],
                    over15_prob=sim["over_1_5"],
                    over35_prob=sim["over_3_5"],
                    sim_n=100_000,
                    created_at=datetime.utcnow(),
                )
                db.add(prediction)
                db.commit()
                logger.info(f"Prediction created for match {match.id}")
            except KeyError:
                logger.warning(f"No features for match {match.id}, skipping")
            except Exception as e:
                logger.error(f"Prediction failed for {match.id}: {e}")
    finally:
        db.close()
