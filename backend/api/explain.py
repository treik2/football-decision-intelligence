from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.base import get_db
from backend.db.models import Match
from backend.schemas.explain import ExplainRequest, ExplainResponse
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.ml.goals import GoalsModel
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.explainer.prompt import build_prompt
from backend.explainer.llm_client import LLMClient

router = APIRouter()
feature_store = FeatureStore()
win_model = WinProbModel()
goals_model = GoalsModel()
simulator = MonteCarloSimulator(goals_model)
llm = LLMClient()


@router.post("/{match_id}", response_model=ExplainResponse)
async def explain_match(
    match_id: str,
    payload: ExplainRequest,
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
    sim_result = simulator.run(features, n=100_000)
    prompt = build_prompt(
        home_team=match.home_team_id,
        away_team=match.away_team_id,
        features=features,
        win_probs=win_probs,
        sim_result=sim_result,
        market_odds=payload.market_odds,
    )
    explanation = await llm.complete(prompt)
    return {"match_id": match_id, "prompt": prompt, "explanation": explanation}
