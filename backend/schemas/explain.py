from pydantic import BaseModel
from typing import Optional, Dict


class ExplainRequest(BaseModel):
    match_id: str
    market: str = "home_win"
    include_llm_response: bool = False  # set True only if OPENAI_API_KEY configured


class ExplainResponse(BaseModel):
    match_id: str
    market: str
    model_probability: float
    implied_probability: float
    ev: float
    prompt: str
    llm_explanation: Optional[str] = None
    key_factors: list
    calibration_note: str
