from fastapi import APIRouter, HTTPException
from backend.features.odds_client import OddsAPIClient
from backend.schemas.odds import OddsResponse

router = APIRouter()
client = OddsAPIClient()


@router.get("/live", response_model=list)
async def get_live_odds(sport: str = "soccer_epl"):
    try:
        data = await client.get_odds(sport=sport)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/match/{match_id}")
async def get_match_odds(match_id: str):
    try:
        data = await client.get_match_odds(match_id=match_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
