from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.base import get_db
from backend.db.models import Match, Prediction
from backend.schemas.prediction import PredictionResponse
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.ml.goals import GoalsModel
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.utils.odds import implied_prob_from_odds
import uuid

router = APIRouter()
feature_store = FeatureStore()
win_model = WinProbModel()
goals_model = GoalsModel()
simulator = MonteCarloSimulator(goals_model)


@router.get("/{match_id}", response_model=PredictionResponse)
async def get_prediction(match_id: str, n_sims: int = 100000, db: AsyncSession = Depends(get_db)):
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    try:
        features = feature_store.get(match_id)
    except KeyError:
        raise HTTPException(status_code=422, detail="Features not yet ingested. Call /ingest-features first.")

    win_probs = win_model.predict(features)
    sim_result = simulator.run(features, n=n_sims)

    prediction = Prediction(
        id=str(uuid.uuid4()),
        match_id=match_id,
        home_win_prob=win_probs["home"],
        draw_prob=win_probs["draw"],
        away_win_prob=win_probs["away"],
        over25_prob=sim_result["over_2_5"],
        btts_prob=sim_result["btts"],
        home_goals_exp=sim_result["home_goals_mean"],
        away_goals_exp=sim_result["away_goals_mean"],
        model_version=win_model.version,
        n_simulations=n_sims,
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    return prediction
