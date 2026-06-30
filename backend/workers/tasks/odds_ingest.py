import asyncio
import logging
from backend.workers.celery_app import celery_app
from backend.features.odds_client import OddsAPIClient
from backend.features.store import FeatureStore

logger = logging.getLogger(__name__)
store = FeatureStore()


@celery_app.task(name="backend.workers.tasks.odds_ingest.ingest_live_odds", bind=True, max_retries=3)
def ingest_live_odds(self, sport: str = "soccer_epl"):
    """Fetch latest odds from OddsAPI and update the feature store."""
    try:
        client = OddsAPIClient()
        loop = asyncio.get_event_loop()
        events = loop.run_until_complete(client.get_odds(sport=sport))
        ingested = 0
        for event in events:
            match_id = event.get("id")
            if not match_id:
                continue
            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                continue
            odds_data = {}
            for bm in bookmakers[:1]:  # use first bookmaker
                for market in bm.get("markets", []):
                    key = market["key"]
                    for outcome in market.get("outcomes", []):
                        odds_data[f"{key}_{outcome['name']}"] = outcome["price"]
            existing = store.get_safe(match_id) or {}
            existing.update({"live_odds": odds_data})
            store.set(match_id, existing)
            ingested += 1
        logger.info(f"Ingested odds for {ingested} events")
        return {"ingested": ingested}
    except Exception as exc:
        logger.error(f"Odds ingest failed: {exc}")
        raise self.retry(exc=exc, countdown=60)
