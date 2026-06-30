from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.base import get_db
from backend.db import models
from backend.schemas.match import MatchCreate, MatchResponse
from backend.features.engineering import build_match_features
from backend.features.store import FeatureStore
from backend.features.weather_client import fetch_weather
from typing import List
import uuid

router = APIRouter()
fs = FeatureStore()


@router.post("/", response_model=MatchResponse)
def create_match(payload: MatchCreate, db: Session = Depends(get_db)):
    match = models.Match(
        id=str(uuid.uuid4()),
        home_team=payload.home_team,
        away_team=payload.away_team,
        league=payload.league,
        kickoff_ts=payload.kickoff_ts,
        venue=payload.venue,
        status="pending",
    )
    db.add(match)
    db.commit()
    db.refresh(match)

    # build and materialize features
    weather = fetch_weather(payload.venue_lat, payload.venue_lon, payload.kickoff_ts)
    features = build_match_features(
        home_team=payload.home_team,
        away_team=payload.away_team,
        context=payload.context or {},
        weather=weather,
    )
    fs.materialize(match.id, features)
    return match


@router.get("/", response_model=List[MatchResponse])
def list_matches(db: Session = Depends(get_db)):
    return db.query(models.Match).all()


@router.get("/{match_id}", response_model=MatchResponse)
def get_match(match_id: str, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match
