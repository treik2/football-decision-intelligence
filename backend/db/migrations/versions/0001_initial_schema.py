"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "matches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("home_team", sa.String(100), nullable=False),
        sa.Column("away_team", sa.String(100), nullable=False),
        sa.Column("league", sa.String(100)),
        sa.Column("season", sa.String(20)),
        sa.Column("kickoff_ts", sa.BigInteger),
        sa.Column("status", sa.String(20), default="scheduled"),
        sa.Column("home_goals", sa.Integer),
        sa.Column("away_goals", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("match_id", sa.String(36), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("home_win_prob", sa.Float),
        sa.Column("draw_prob", sa.Float),
        sa.Column("away_win_prob", sa.Float),
        sa.Column("home_xg", sa.Float),
        sa.Column("away_xg", sa.Float),
        sa.Column("over25_prob", sa.Float),
        sa.Column("over15_prob", sa.Float),
        sa.Column("over35_prob", sa.Float),
        sa.Column("btts_prob", sa.Float),
        sa.Column("sim_n", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("match_external_id", sa.String(100)),
        sa.Column("home_odds", sa.Float),
        sa.Column("draw_odds", sa.Float),
        sa.Column("away_odds", sa.Float),
        sa.Column("bookmaker", sa.String(50)),
        sa.Column("raw", JSONB),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "bet_suggestions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("match_id", sa.String(36), sa.ForeignKey("matches.id")),
        sa.Column("type", sa.String(20)),
        sa.Column("legs", JSONB),
        sa.Column("combined_prob", sa.Float),
        sa.Column("combined_ev", sa.Float),
        sa.Column("combined_odds", sa.Float),
        sa.Column("kelly_fraction", sa.Float),
        sa.Column("stake_advice", sa.Float),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index("ix_matches_status", "matches", ["status"])
    op.create_index("ix_matches_league", "matches", ["league"])
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])
    op.create_index("ix_odds_external_id", "odds_snapshots", ["match_external_id"])


def downgrade() -> None:
    op.drop_table("bet_suggestions")
    op.drop_table("odds_snapshots")
    op.drop_table("predictions")
    op.drop_table("matches")
