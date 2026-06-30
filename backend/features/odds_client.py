"""Odds API client for fetching live odds from The Odds API.

Docs: https://the-odds-api.com/liteapi-docs/
Requires ODDS_API_KEY in environment.
"""
import httpx
from typing import List, Dict, Optional
from config import settings
from utils.logger import log

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORTS_KEY = "soccer_epl"  # change to any supported league key
MARKETS = "h2h,totals,btts"
ODDS_FORMAT = "decimal"


class OddsAPIClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or settings.odds_api_key

    async def get_sports(self) -> List[Dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ODDS_API_BASE}/sports", params={"apiKey": self.api_key})
            r.raise_for_status()
            return r.json()

    async def get_odds(self, sport: str = SPORTS_KEY, markets: str = MARKETS) -> List[Dict]:
        """Fetch current odds for all upcoming events in a sport."""
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": ODDS_FORMAT,
            "dateFormat": "iso",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{ODDS_API_BASE}/sports/{sport}/odds", params=params)
            r.raise_for_status()
            data = r.json()
            log.info(f"Fetched {len(data)} events from OddsAPI [{sport}]")
            return data

    def parse_event(self, event: Dict) -> Dict:
        """Normalise a single OddsAPI event into a flat structure."""
        result = {
            "external_id": event["id"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "sport": event["sport_key"],
            "kickoff_ts": event["commence_time"],
            "bookmakers": [],
        }
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                result["bookmakers"].append({
                    "bookmaker": bm["key"],
                    "market": market["key"],
                    "outcomes": outcomes,
                })
        return result


odds_client = OddsAPIClient()
