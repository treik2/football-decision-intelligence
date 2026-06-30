import pytest
from backend.betting.ev import expected_value, fractional_kelly, implied_prob, overround


def test_ev_positive():
    # Model thinks 60% chance, odds are 2.0 (implied 50%)
    ev = expected_value(0.6, 2.0)
    assert ev > 0, "Should be positive EV"


def test_ev_negative():
    # Model thinks 40% chance, odds are 2.0 (implied 50%)
    ev = expected_value(0.4, 2.0)
    assert ev < 0, "Should be negative EV"


def test_kelly_positive():
    k = fractional_kelly(0.6, 2.0, fraction=0.25)
    assert 0 < k < 1


def test_kelly_zero_when_no_edge():
    k = fractional_kelly(0.45, 2.0, fraction=0.25)
    assert k == 0.0


def test_implied_prob():
    assert abs(implied_prob(2.0) - 0.5) < 1e-6
    assert abs(implied_prob(4.0) - 0.25) < 1e-6


def test_overround():
    # 3-way market with typical bookmaker margin
    odds = [1.9, 3.5, 4.2]
    margin = overround(odds)
    assert margin > 0, "Bookmaker should always have positive overround"
