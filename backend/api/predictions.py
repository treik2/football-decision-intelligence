from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from backend.db.base import get_db
from backend.db import models
from backend.features.store import FeatureStore
from backend.ml.model_loader import ModelLoader
from backend.simulation.monte_carlo import MonteCarloSimulator
from backend.schemas.prediction import PredictionResponse
import uuid
from datetime import datetime

router = APIRouter()
fs = FeatureStore()
loader = ModelLoader()
mc = MonteCarloSimulator()


@router.post("/{match_id}", response_model=PredictionResponse)
def predict(match_id: str, n_sims: int = 100000, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=422, detail="Features not yet materialized for this match")

    win_model = loader.get_win_model()
    goals_model = loader.get_goals_model()

    win_probs = win_model.predict_proba(features)
    home_xg, away_xg = goals_model.expected_goals(features)

    sim_result = mc.simulate(
        home_xg=home_xg,
        away_xg=away_xg,
        n=n_sims,
    )

    prediction = models.Prediction(
        id=str(uuid.uuid4()),
        match_id=match_id,
        home_win_prob=win_probs["home"],
        draw_prob=win_probs["draw"],
        away_win_prob=win_probs["away"],
        home_xg=home_xg,
        away_xg=away_xg,
        over25_prob=sim_result["over_2_5"],
        btts_prob=sim_result["btts"],
        over15_prob=sim_result["over_1_5"],
        over35_prob=sim_result["over_3_5"],
        sim_n=n_sims,
        created_at=datetime.utcnow(),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.get("/{match_id}", response_model=PredictionResponse)
def get_prediction(match_id: str, db: Session = Depends(get_db)):
    pred = (
        db.query(models.Prediction)
        .filter(models.Prediction.match_id == match_id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )
    if not pred:
        raise HTTPException(status_code=404, detail="No prediction found for match")
    return pred
