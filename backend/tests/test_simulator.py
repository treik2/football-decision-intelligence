import pytest
from backend.ml.goals import GoalsModel
from backend.simulation.monte_carlo import MonteCarloSimulator


@pytest.fixture
def simulator():
    return MonteCarloSimulator(GoalsModel())


def test_probabilities_sum_to_one(simulator):
    features = {"home_xg": 1.5, "away_xg": 1.0}
    result = simulator.run(features, n=10_000, seed=1)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert abs(total - 1.0) < 0.01


def test_btts_range(simulator):
    features = {"home_xg": 1.5, "away_xg": 1.0}
    result = simulator.run(features, n=10_000, seed=2)
    assert 0.0 <= result["btts"] <= 1.0


def test_over_25_range(simulator):
    features = {"home_xg": 1.5, "away_xg": 1.0}
    result = simulator.run(features, n=10_000, seed=3)
    assert 0.0 <= result["over_2_5"] <= 1.0


def test_over_logic_ordering(simulator):
    features = {"home_xg": 2.0, "away_xg": 1.5}
    result = simulator.run(features, n=50_000, seed=4)
    assert result["over_1_5"] >= result["over_2_5"] >= result["over_3_5"]


def test_top_scores_exist(simulator):
    features = {"home_xg": 1.3, "away_xg": 1.1}
    result = simulator.run(features, n=10_000, seed=5)
    assert len(result["top_correct_scores"]) > 0
