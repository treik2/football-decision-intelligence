"""Odds conversion, implied probability, EV, and Kelly utilities."""
from typing import Dict
import numpy as np


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        raise ValueError(f"Invalid decimal odds: {odds}")
    return 1.0 / odds


def overround(odds_dict: Dict[str, float]) -> float:
    """Compute bookmaker overround (margin) from outcome odds dict."""
    return sum(1.0 / v for v in odds_dict.values()) - 1.0


def remove_margin(odds_dict: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker margin from a set of odds to get fair probabilities."""
    raw_probs = {k: 1.0 / v for k, v in odds_dict.items()}
    total = sum(raw_probs.values())
    return {k: v / total for k, v in raw_probs.items()}


def expected_value(p_model: float, decimal_odds: float) -> float:
    """Compute expected value of a single bet."""
    return p_model * (decimal_odds - 1.0) - (1.0 - p_model)


def kelly_fraction(p_model: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Compute fractional Kelly stake as fraction of bankroll."""
    b = decimal_odds - 1.0
    q = 1.0 - p_model
    if b <= 0:
        return 0.0
    raw_kelly = (p_model * b - q) / b
    if raw_kelly <= 0:
        return 0.0
    return float(raw_kelly * fraction)


def is_value(p_model: float, decimal_odds: float, margin: float = 0.01) -> bool:
    """True if model probability exceeds implied probability by margin."""
    return (p_model - decimal_to_implied(decimal_odds)) > margin


def implied_combined_odds(odds_list) -> float:
    """Compute parlay (accumulator) odds from list of decimal odds."""
    result = 1.0
    for o in odds_list:
        result *= o
    return result
