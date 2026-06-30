"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teams",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("short_name", sa.String(32)),
        sa.Column("league", sa.String(64)),
        sa.Column("country", sa.String(64)),
        sa.Column("elo", sa.Float, default=1500.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("home_team_id", sa.String(36), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.String(36), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("kickoff_ts", sa.BigInteger, nullable=False),
        sa.Column("league", sa.String(64)),
        sa.Column("season", sa.String(16)),
        sa.Column("venue", sa.String(128)),
        sa.Column("status", sa.String(32), default="scheduled"),
        sa.Column("home_score", sa.Integer),
        sa.Column("away_score", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("match_id", sa.String(36), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("home_win_prob", sa.Float),
        sa.Column("draw_prob", sa.Float),
        sa.Column("away_win_prob", sa.Float),
        sa.Column("over25_prob", sa.Float),
        sa.Column("btts_prob", sa.Float),
        sa.Column("home_goals_exp", sa.Float),
        sa.Column("away_goals_exp", sa.Float),
        sa.Column("model_version", sa.String(32)),
        sa.Column("n_simulations", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "bet_suggestions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("match_id", sa.String(36), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.String(64), nullable=False),
        sa.Column("model_prob", sa.Float),
        sa.Column("implied_prob", sa.Float),
        sa.Column("ev", sa.Float),
        sa.Column("odds", sa.Float),
        sa.Column("kelly_fraction", sa.Float),
        sa.Column("stake_advice", sa.Float),
        sa.Column("bet_type", sa.String(16)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_matches_kickoff_ts", "matches", ["kickoff_ts"])
    op.create_index("ix_matches_league", "matches", ["league"])
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])
    op.create_index("ix_bet_suggestions_match_id", "bet_suggestions", ["match_id"])


def downgrade() -> None:
    op.drop_table("bet_suggestions")
    op.drop_table("predictions")
    op.drop_table("matches")
    op.drop_table("teams")
