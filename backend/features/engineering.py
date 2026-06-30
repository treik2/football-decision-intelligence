"""Feature engineering pipeline: computes derived features from raw match data."""
import numpy as np
from typing import Dict
from schemas.match import MatchFeatures


ELO_BASE = 1500.0
_K = 32.0  # Elo K factor


def build_feature_vector(raw: MatchFeatures, home_elo: float = 1500.0, away_elo: float = 1500.0) -> Dict:
    """Convert MatchFeatures + Elo ratings into ML-ready feature dict."""
    elo_diff = home_elo - away_elo
    xg_diff = raw.home_xg - raw.away_xg
    rest_diff = raw.home_rest_days - raw.away_rest_days
    fitness_diff = raw.home_squad_fitness_pct - raw.away_squad_fitness_pct
    form_diff = raw.home_form - raw.away_form
    injury_diff = raw.away_injuries - raw.home_injuries  # positive = home advantage
    corner_rate_home = raw.home_avg_corners
    corner_rate_away = raw.away_avg_corners
    expected_total_goals = raw.home_xg + raw.away_xg
    btts_indicator = float(raw.home_xg > 0.7 and raw.away_xg > 0.7)
    travel_diff = raw.away_travel_km - raw.home_travel_km
    weather_penalty = float(raw.weather_rain_mm > 5.0)  # rainy conditions flag
    motivation_diff = raw.home_motivation - raw.away_motivation

    return {
        "elo_diff": float(elo_diff),
        "xg_diff": float(xg_diff),
        "home_xg": float(raw.home_xg),
        "away_xg": float(raw.away_xg),
        "home_form": float(raw.home_form),
        "away_form": float(raw.away_form),
        "form_diff": float(form_diff),
        "rest_diff": float(rest_diff),
        "home_rest_days": float(raw.home_rest_days),
        "away_rest_days": float(raw.away_rest_days),
        "fitness_diff": float(fitness_diff),
        "injury_diff": float(injury_diff),
        "home_injuries": float(raw.home_injuries),
        "away_injuries": float(raw.away_injuries),
        "corner_rate_home": float(corner_rate_home),
        "corner_rate_away": float(corner_rate_away),
        "expected_total_goals": float(expected_total_goals),
        "btts_indicator": float(btts_indicator),
        "travel_diff": float(travel_diff),
        "weather_penalty": float(weather_penalty),
        "weather_temp": float(raw.weather_temp),
        "weather_rain_mm": float(raw.weather_rain_mm),
        "motivation_diff": float(motivation_diff),
        "home_motivation": float(raw.home_motivation),
        "away_motivation": float(raw.away_motivation),
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
    }


def update_elo(home_elo: float, away_elo: float, home_goals: int, away_goals: int) -> tuple:
    """Update Elo ratings after a match result."""
    if home_goals > away_goals:
        s_home, s_away = 1.0, 0.0
    elif home_goals == away_goals:
        s_home, s_away = 0.5, 0.5
    else:
        s_home, s_away = 0.0, 1.0
    e_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
    e_away = 1.0 - e_home
    new_home = home_elo + _K * (s_home - e_home)
    new_away = away_elo + _K * (s_away - e_away)
    return float(new_home), float(new_away)
