from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class SimulationResult(BaseModel):
    n_simulations: int
    home_win: float
    draw: float
    away_win: float
    over_15: float
    over_25: float
    over_35: float
    btts: float
    home_clean_sheet: float
    away_clean_sheet: float
    top_scores: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    match_id: str
    model_version: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_25_prob: float
    btts_prob: float
    home_xg: float
    away_xg: float
    simulation: SimulationResult
    created_at: datetime

    class Config:
        from_attributes = True
