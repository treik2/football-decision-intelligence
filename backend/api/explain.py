from fastapi import APIRouter, HTTPException
from backend.features.store import FeatureStore
from backend.ml.win_prob import WinProbModel
from backend.explainer.llm import build_explanation
from backend.schemas.explain import ExplainResponse
from backend.utils.odds import implied_probs

router = APIRouter()
fs = FeatureStore()
win_model = WinProbModel()


@router.get("/{match_id}", response_model=ExplainResponse)
def explain_match(match_id: str):
    try:
        features = fs.get(match_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Features not found.")

    win_probs = win_model.predict(features)
    market_probs = implied_probs(features.get("odds", {}))
    explanation = build_explanation(
        features=features,
        model_probs=win_probs,
        market_probs=market_probs,
    )
    return ExplainResponse(
        match_id=match_id,
        explanation=explanation["text"],
        top_reasons=explanation["reasons"],
        calibration_note=explanation["calibration"],
    )
