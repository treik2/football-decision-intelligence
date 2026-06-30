from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.base import get_db
from backend.db import models
from backend.features.store import FeatureStore
from backend.explainer.llm import ExplainerClient
from backend.schemas.explain import ExplainRequest, ExplainResponse

router = APIRouter()
fs = FeatureStore()
explainer = ExplainerClient()


@router.post("/{match_id}", response_model=ExplainResponse)
def explain_prediction(match_id: str, payload: ExplainRequest, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    pred = (
        db.query(models.Prediction)
        .filter(models.Prediction.match_id == match_id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )
    if not pred:
        raise HTTPException(status_code=404, detail="No prediction found; run /predictions/{match_id} first")

    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=422, detail="Features not materialized")

    explanation = explainer.explain(
        home=match.home_team,
        away=match.away_team,
        features=features,
        prediction=pred,
        market_move=payload.market_move,
    )
    return ExplainResponse(match_id=match_id, explanation=explanation)
