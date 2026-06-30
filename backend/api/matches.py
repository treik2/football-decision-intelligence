from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.base import get_db
from backend.db import models
from backend.schemas.match import MatchCreate, MatchResponse, MatchListResponse
from backend.features.engineering import compute_match_features
from backend.features.store import FeatureStore
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
        season=payload.season,
        status="scheduled",
    )
    db.add(match)
    db.commit()
    db.refresh(match)

    features = compute_match_features(
        home_team=payload.home_team,
        away_team=payload.away_team,
        context=payload.context or {},
    )
    fs.materialize(match.id, features)
    return match


@router.get("/", response_model=List[MatchListResponse])
def list_matches(
    league: str = None,
    status: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(models.Match)
    if league:
        q = q.filter(models.Match.league == league)
    if status:
        q = q.filter(models.Match.status == status)
    return q.order_by(models.Match.kickoff_ts.desc()).limit(limit).all()


@router.get("/{match_id}", response_model=MatchResponse)
def get_match(match_id: str, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.patch("/{match_id}/result")
def update_result(match_id: str, home_goals: int, away_goals: int, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    match.home_goals = home_goals
    match.away_goals = away_goals
    match.status = "finished"
    db.commit()
    return {"status": "updated"}
