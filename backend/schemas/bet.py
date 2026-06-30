from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class Leg(BaseModel):
    name: str = Field(..., description="Market key e.g. home_win, over_2_5, btts")
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    variance: float = Field(..., ge=0)
    odds: float = Field(..., gt=1)
    ev: float
    implied_prob: float
    is_value: bool


class BetSuggestRequest(BaseModel):
    match_id: str
    bankroll: float = Field(1000.0, gt=0)
    odds: Dict[str, float] = Field(..., description="Current market odds per market")


class BetSuggestionResponse(BaseModel):
    bet_id: str
    match_id: str
    legs: List[Leg]
    combined_prob: float
    combined_odds: float
    combined_ev: float
    kelly_fraction: float
    stake_advice: float
    is_rejected: bool
    rejection_reason: Optional[str]
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ValueBetItem(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    kickoff_ts: datetime
    market: str
    model_prob: float
    implied_prob: float
    odds: float
    ev: float
    kelly_fraction: float
    confidence: float
