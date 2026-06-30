from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from backend.db.base import get_db
from backend.db.models import Match, Team
from backend.schemas.match import MatchCreate, MatchResponse, MatchListResponse
from backend.features.engineering import FeatureEngineer
from backend.features.store import FeatureStore
from typing import List, Optional
import uuid

router = APIRouter()
feature_engineer = FeatureEngineer()
feature_store = FeatureStore()


@router.post("/", response_model=MatchResponse, status_code=201)
async def create_match(payload: MatchCreate, db: AsyncSession = Depends(get_db)):
    home = await db.get(Team, payload.home_team_id)
    away = await db.get(Team, payload.away_team_id)
    if not home or not away:
        raise HTTPException(status_code=404, detail="Team not found")
    match = Match(
        id=str(uuid.uuid4()),
        home_team_id=payload.home_team_id,
        away_team_id=payload.away_team_id,
        kickoff_ts=payload.kickoff_ts,
        league=payload.league,
        season=payload.season,
        venue=payload.venue,
        status="scheduled",
    )
    db.add(match)
    await db.commit()
    await db.refresh(match)
    return match


@router.get("/", response_model=MatchListResponse)
async def list_matches(
    league: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Match).order_by(desc(Match.kickoff_ts)).offset(offset).limit(limit)
    if league:
        stmt = stmt.where(Match.league == league)
    if status:
        stmt = stmt.where(Match.status == status)
    result = await db.execute(stmt)
    matches = result.scalars().all()
    return {"matches": matches, "total": len(matches)}


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(match_id: str, db: AsyncSession = Depends(get_db)):
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.post("/{match_id}/ingest-features")
async def ingest_features(match_id: str, db: AsyncSession = Depends(get_db)):
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    features = await feature_engineer.compute(match)
    feature_store.set(match_id, features)
    return {"status": "ok", "features": features}
