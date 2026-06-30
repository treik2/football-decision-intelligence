from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.base import get_db
from backend.db.models import Match, Prediction
from backend.schemas.bet import BetRequest, BetSuggestionResponse
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.ml.goals import GoalsModel
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.betting.engine import BetEngine
from sqlalchemy import select, desc

router = APIRouter()
feature_store = FeatureStore()
win_model = WinProbModel()
goals_model = GoalsModel()
simulator = MonteCarloSimulator(goals_model)
bet_engine = BetEngine()


@router.post("/{match_id}/suggest", response_model=BetSuggestionResponse)
async def suggest_bets(
    match_id: str,
    payload: BetRequest,
    db: AsyncSession = Depends(get_db),
):
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    try:
        features = feature_store.get(match_id)
    except KeyError:
        raise HTTPException(status_code=422, detail="Features not ingested")

    win_probs = win_model.predict(features)
    sim_result = simulator.run(features, n=200_000)

    suggestions = bet_engine.build(
        match_id=match_id,
        win_probs=win_probs,
        sim_result=sim_result,
        odds=payload.odds,
        bankroll=payload.bankroll,
        min_ev=payload.min_ev,
        kelly_fraction=payload.kelly_fraction,
    )
    return {"match_id": match_id, "suggestions": suggestions}
