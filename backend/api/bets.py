from fastapi import APIRouter, HTTPException, Body
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.ml.goals import GoalsModel
from backend.simulation.montecarlo import MonteCarloSimulator
from backend.betting.engine import BetEngine
from backend.schemas.bet import BetRequest, BetResponse
from typing import List

router = APIRouter()
fs = FeatureStore()
win_model = WinProbModel()
goals_model = GoalsModel()
simulator = MonteCarloSimulator(goals_model)
engine = BetEngine()


@router.post("/{match_id}/suggest", response_model=List[BetResponse])
def suggest_bets(
    match_id: str,
    payload: BetRequest = Body(...),
):
    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Features not found. Ingest match first.")

    win_probs = win_model.predict(features)
    sim = simulator.run(features, n=200_000)

    model_probs = {
        "home_win": win_probs["home"],
        "draw": win_probs["draw"],
        "away_win": win_probs["away"],
        "over_2_5": sim["over_2_5"],
        "over_1_5": sim["over_1_5"],
        "over_3_5": sim["over_3_5"],
        "btts": sim["btts"],
    }

    suggestions = engine.build_suggestions(
        model_probs=model_probs,
        odds=payload.odds,
        bankroll=payload.bankroll,
        kelly_fraction=payload.kelly_fraction,
        min_ev=payload.min_ev,
        max_legs=payload.max_legs,
    )
    return suggestions
