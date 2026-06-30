from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.base import get_db
from backend.db import models
from backend.features.store import FeatureStore
from backend.ml.model_loader import ModelLoader
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.betting.engine import BettingEngine
from backend.schemas.bet import BetRequest, BetSuggestionResponse
from typing import List

router = APIRouter()
fs = FeatureStore()
loader = ModelLoader()
mc = MonteCarloSimulator()
engine = BettingEngine()


@router.post("/{match_id}/suggest", response_model=List[BetSuggestionResponse])
def suggest_bets(
    match_id: str,
    payload: BetRequest,
    db: Session = Depends(get_db),
):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=422, detail="Features not materialized")

    win_model = loader.get_win_model()
    goals_model = loader.get_goals_model()
    win_probs = win_model.predict_proba(features)
    home_xg, away_xg = goals_model.expected_goals(features)

    sim_result = mc.simulate(home_xg=home_xg, away_xg=away_xg, n=200_000)

    suggestions = engine.build_suggestions(
        match_id=match_id,
        win_probs=win_probs,
        sim_result=sim_result,
        odds=payload.odds,
        bankroll=payload.bankroll,
        min_ev=payload.min_ev,
        kelly_fraction=payload.kelly_fraction,
    )
    return suggestions
