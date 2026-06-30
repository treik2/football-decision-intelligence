import asyncio
from backend.workers.celery_app import celery_app
from backend.features.odds_client import OddsAPIClient
from backend.db.base import SessionLocal
from backend.db import models
from backend.utils.logger import get_logger

logger = get_logger(__name__)
client = OddsAPIClient()


@celery_app.task(name="backend.workers.tasks_odds.ingest_live_odds", bind=True, max_retries=3)
def ingest_live_odds(self):
    """Fetch latest odds from OddsAPI and store in DB."""
    try:
        odds_list = asyncio.get_event_loop().run_until_complete(
            client.fetch_odds(sport="soccer", region="eu", market="h2h")
        )
        db = SessionLocal()
        try:
            for item in odds_list:
                record = models.OddsSnapshot(
                    match_external_id=item.get("id", ""),
                    home_odds=item.get("home_odds"),
                    draw_odds=item.get("draw_odds"),
                    away_odds=item.get("away_odds"),
                    bookmaker=item.get("bookmaker", "odds_api"),
                    raw=item,
                )
                db.add(record)
            db.commit()
            logger.info(f"Ingested {len(odds_list)} odds snapshots")
        finally:
            db.close()
    except Exception as exc:
        logger.error(f"Odds ingestion failed: {exc}")
        raise self.retry(exc=exc, countdown=60)
