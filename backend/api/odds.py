from fastapi import APIRouter, HTTPException
from backend.features.odds_client import OddsClient
from backend.schemas.odds import OddsResponse
from typing import List

router = APIRouter()
client = OddsClient()


@router.get("/live", response_model=List[OddsResponse])
async def get_live_odds(sport: str = "soccer_epl", regions: str = "eu"):
    try:
        data = await client.fetch_live(sport=sport, regions=regions)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/{match_id}", response_model=OddsResponse)
async def get_match_odds(match_id: str):
    try:
        data = await client.fetch_by_match_id(match_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
