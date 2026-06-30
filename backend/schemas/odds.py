from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class BookmakerOdds(BaseModel):
    bookmaker: str
    market: str
    outcomes: Dict[str, float]  # {outcome_key: decimal_odds}
    captured_at: datetime


class LiveOddsResponse(BaseModel):
    external_id: str
    home_team: str
    away_team: str
    league: Optional[str]
    kickoff_ts: datetime
    bookmakers: List[BookmakerOdds]
    best_odds: Dict[str, float]  # {outcome: best_decimal_odds}
    market_implied_probs: Dict[str, float]
    overround: float
