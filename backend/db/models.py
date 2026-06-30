from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.base import Base
import uuid


def gen_uuid():
    return str(uuid.uuid4())


class Match(Base):
    __tablename__ = "matches"

    id = Column(String, primary_key=True, default=gen_uuid)
    external_id = Column(String, unique=True, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String)
    kickoff_ts = Column(DateTime(timezone=True))
    status = Column(String, default="scheduled")  # scheduled | live | finished
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    features = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    predictions = relationship("Prediction", back_populates="match", cascade="all, delete")
    bets = relationship("BetSuggestion", back_populates="match", cascade="all, delete")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=gen_uuid)
    match_id = Column(String, ForeignKey("matches.id"), nullable=False)
    model_version = Column(String, default="v1")
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    over_25_prob = Column(Float)
    btts_prob = Column(Float)
    home_xg = Column(Float)
    away_xg = Column(Float)
    simulation_results = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="predictions")


class BetSuggestion(Base):
    __tablename__ = "bet_suggestions"

    id = Column(String, primary_key=True, default=gen_uuid)
    match_id = Column(String, ForeignKey("matches.id"), nullable=False)
    legs = Column(JSON, default=list)
    combined_ev = Column(Float)
    kelly_fraction = Column(Float)
    stake_advice = Column(Float)
    bankroll = Column(Float, default=1000.0)
    is_value = Column(Boolean, default=False)
    explanation_prompt = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    match = relationship("Match", back_populates="bets")


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(String, primary_key=True, default=gen_uuid)
    match_external_id = Column(String, index=True)
    bookmaker = Column(String)
    market = Column(String)  # h2h, totals, btts, etc.
    outcomes = Column(JSON)  # {outcome: odds}
    captured_at = Column(DateTime(timezone=True), server_default=func.now())


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(String, primary_key=True, default=gen_uuid)
    model_version = Column(String)
    run_date = Column(DateTime(timezone=True), server_default=func.now())
    total_bets = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    roi = Column(Float)
    avg_ev = Column(Float)
    calibration_score = Column(Float)
    details = Column(JSON, default=dict)
