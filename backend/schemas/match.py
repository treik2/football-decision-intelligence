from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class MatchFeatures(BaseModel):
    elo_diff: float = Field(0.0, description="Elo rating difference (home - away)")
    xg_diff: float = Field(0.0, description="Rolling xG difference (home - away)")
    home_xg: float = Field(1.3, ge=0)
    away_xg: float = Field(1.0, ge=0)
    home_form: float = Field(0.5, ge=0, le=1, description="Points per game (0-1 scaled)")
    away_form: float = Field(0.5, ge=0, le=1)
    home_rest_days: int = Field(7, ge=0)
    away_rest_days: int = Field(7, ge=0)
    home_injuries: int = Field(0, ge=0, description="Number of key injuries")
    away_injuries: int = Field(0, ge=0)
    home_squad_fitness_pct: float = Field(1.0, ge=0, le=1)
    away_squad_fitness_pct: float = Field(1.0, ge=0, le=1)
    home_avg_corners: float = Field(5.5, ge=0)
    away_avg_corners: float = Field(4.5, ge=0)
    weather_temp: float = Field(18.0)
    weather_rain_mm: float = Field(0.0, ge=0)
    referee: Optional[str] = None
    home_travel_km: float = Field(0.0, ge=0)
    away_travel_km: float = Field(0.0, ge=0)
    home_motivation: float = Field(0.5, ge=0, le=1)
    away_motivation: float = Field(0.5, ge=0, le=1)


class MatchIngest(BaseModel):
    external_id: str
    home_team: str
    away_team: str
    league: Optional[str] = None
    kickoff_ts: datetime
    odds: Dict[str, float] = Field(..., example={"home": 1.9, "draw": 3.5, "away": 4.2})
    features: MatchFeatures = Field(default_factory=MatchFeatures)

    @field_validator("odds")
    @classmethod
    def validate_odds(cls, v):
        for key, val in v.items():
            if val <= 1.0:
                raise ValueError(f"Odds for {key} must be > 1.0")
        return v


class MatchResponse(BaseModel):
    id: str
    external_id: str
    home_team: str
    away_team: str
    league: Optional[str]
    kickoff_ts: datetime
    status: str

    class Config:
        from_attributes = True
