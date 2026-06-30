from fastapi import APIRouter, HTTPException
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.ml.goals import GoalsModel
from backend.simulation.montecarlo import MonteCarloSimulator
from backend.schemas.prediction import PredictionResponse

router = APIRouter()
fs = FeatureStore()
win_model = WinProbModel()
goals_model = GoalsModel()
simulator = MonteCarloSimulator(goals_model)


@router.get("/{match_id}", response_model=PredictionResponse)
def predict(match_id: str, n_sims: int = 100_000):
    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Features not found. Ingest match first.")

    win_probs = win_model.predict(features)
    sim_result = simulator.run(features, n=n_sims)

    return PredictionResponse(
        match_id=match_id,
        home_win=win_probs["home"],
        draw=win_probs["draw"],
        away_win=win_probs["away"],
        over_2_5=sim_result["over_2_5"],
        btts=sim_result["btts"],
        over_1_5=sim_result["over_1_5"],
        over_3_5=sim_result["over_3_5"],
        home_xg=features.get("home_xg", 1.3),
        away_xg=features.get("away_xg", 1.0),
        n_simulations=n_sims,
    )
