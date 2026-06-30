from fastapi import APIRouter, HTTPException
from backend.features.odds_client import OddsAPIClient
from backend.schemas.odds import OddsResponse
from typing import List

router = APIRouter()
client = OddsAPIClient()


@router.get("/live", response_model=List[OddsResponse])
async def get_live_odds(sport: str = "soccer", region: str = "eu", market: str = "h2h"):
    try:
        odds = await client.fetch_odds(sport=sport, region=region, market=market)
        return odds
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Odds provider error: {str(e)}")


@router.get("/match/{match_id}")
async def get_match_odds(match_id: str):
    try:
        return await client.fetch_match_odds(match_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
