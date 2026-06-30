"""Feature engineering pipeline for training data preparation."""
import pandas as pd
import numpy as np
from typing import List


def compute_elo(
    df: pd.DataFrame,
    k: float = 32.0,
    initial_elo: float = 1500.0,
) -> pd.DataFrame:
    """
    Compute Elo ratings via iterative match processing.
    Expects columns: home_team_id, away_team_id, home_score, away_score, kickoff_ts.
    Returns df with elo_home and elo_away columns added.
    """
    df = df.sort_values("kickoff_ts").copy()
    elo = {}

    elo_home_list, elo_away_list = [], []

    for _, row in df.iterrows():
        h, a = row["home_team_id"], row["away_team_id"]
        elo.setdefault(h, initial_elo)
        elo.setdefault(a, initial_elo)

        r_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        r_a = 1 - r_h

        if row["home_score"] > row["away_score"]:
            s_h, s_a = 1.0, 0.0
        elif row["home_score"] == row["away_score"]:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        elo_home_list.append(elo[h])
        elo_away_list.append(elo[a])

        elo[h] += k * (s_h - r_h)
        elo[a] += k * (s_a - r_a)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling xG, form, goals for/against per team.
    Produces home_xg, away_xg, home_form, away_form columns.
    """
    df = df.sort_values("kickoff_ts").copy()

    home_stats = df[["kickoff_ts", "home_team_id", "xg_home", "home_score"]].rename(
        columns={"home_team_id": "team_id", "xg_home": "xg_for", "home_score": "goals_for"}
    )
    away_stats = df[["kickoff_ts", "away_team_id", "xg_away", "away_score"]].rename(
        columns={"away_team_id": "team_id", "xg_away": "xg_for", "away_score": "goals_for"}
    )
    all_stats = pd.concat([home_stats, away_stats]).sort_values("kickoff_ts")
    all_stats["rolling_xg"] = all_stats.groupby("team_id")["xg_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    all_stats["form"] = all_stats.groupby("team_id")["goals_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean() / 3.0
    ).clip(0, 1)

    home_roll = all_stats[all_stats["team_id"].isin(df["home_team_id"])].copy()
    # Merge back
    df = df.merge(
        all_stats[["kickoff_ts", "team_id", "rolling_xg", "form"]].rename(
            columns={"team_id": "home_team_id", "rolling_xg": "home_xg", "form": "home_form"}
        ),
        on=["kickoff_ts", "home_team_id"], how="left",
    )
    df = df.merge(
        all_stats[["kickoff_ts", "team_id", "rolling_xg", "form"]].rename(
            columns={"team_id": "away_team_id", "rolling_xg": "away_xg", "form": "away_form"}
        ),
        on=["kickoff_ts", "away_team_id"], how="left",
    )
    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("kickoff_ts").copy()
    for side in ["home", "away"]:
        team_col = f"{side}_team_id"
        last_match = {}
        rest_days = []
        for _, row in df.iterrows():
            team = row[team_col]
            ts = row["kickoff_ts"]
            if team in last_match:
                days = (ts - last_match[team]) / 86400
            else:
                days = 7.0
            rest_days.append(days)
            last_match[team] = ts
        df[f"{side}_rest_days"] = rest_days
    return df
