import numpy as np
from typing import Dict


class MonteCarloSimulator:
    """
    Vectorised Monte Carlo match simulator.
    Samples goals, corners, cards, and BTTS for n simulations in one pass.
    """

    def simulate(
        self,
        home_xg: float,
        away_xg: float,
        home_xc: float = 5.2,  # expected corners
        away_xc: float = 4.8,
        home_xy: float = 1.8,  # expected yellow cards
        away_xy: float = 1.8,
        n: int = 100_000,
        seed: int = None,
    ) -> Dict:
        rng = np.random.default_rng(seed)

        home_goals = rng.poisson(home_xg, n)
        away_goals = rng.poisson(away_xg, n)
        total_goals = home_goals + away_goals
        home_corners = rng.poisson(home_xc, n)
        away_corners = rng.poisson(away_xc, n)
        total_corners = home_corners + away_corners
        home_cards = rng.poisson(home_xy, n)
        away_cards = rng.poisson(away_xy, n)
        total_cards = home_cards + away_cards

        outcome = np.sign(home_goals - away_goals)  # 1=home, 0=draw, -1=away

        result = {
            "home_win": float(np.mean(outcome == 1)),
            "draw": float(np.mean(outcome == 0)),
            "away_win": float(np.mean(outcome == -1)),
            "over_0_5": float(np.mean(total_goals > 0.5)),
            "over_1_5": float(np.mean(total_goals > 1.5)),
            "over_2_5": float(np.mean(total_goals > 2.5)),
            "over_3_5": float(np.mean(total_goals > 3.5)),
            "over_4_5": float(np.mean(total_goals > 4.5)),
            "btts": float(np.mean((home_goals > 0) & (away_goals > 0))),
            "btts_no": float(np.mean(~((home_goals > 0) & (away_goals > 0)))),
            "over_8_5_corners": float(np.mean(total_corners > 8.5)),
            "over_9_5_corners": float(np.mean(total_corners > 9.5)),
            "over_10_5_corners": float(np.mean(total_corners > 10.5)),
            "over_3_5_cards": float(np.mean(total_cards > 3.5)),
            "home_clean_sheet": float(np.mean(away_goals == 0)),
            "away_clean_sheet": float(np.mean(home_goals == 0)),
            "n": n,
        }

        # correct score top-10
        pairs = list(zip(home_goals.tolist(), away_goals.tolist()))
        score_counts: Dict[str, int] = {}
        for h, a in pairs:
            key = f"{h}-{a}"
            score_counts[key] = score_counts.get(key, 0) + 1
        result["correct_scores"] = dict(
            sorted(score_counts.items(), key=lambda x: -x[1])[:15]
        )
        result["correct_score_probs"] = {
            k: v / n for k, v in result["correct_scores"].items()
        }

        return result

    def simulate_player(
        self,
        expected_shots: float = 2.5,
        expected_goals: float = 0.35,
        expected_assists: float = 0.15,
        expected_cards: float = 0.12,
        n: int = 100_000,
        seed: int = None,
    ) -> Dict:
        """Simulate player-level markets."""
        rng = np.random.default_rng(seed)
        shots = rng.poisson(expected_shots, n)
        goals = rng.poisson(expected_goals, n)
        assists = rng.poisson(expected_assists, n)
        cards = rng.binomial(1, expected_cards, n)
        return {
            "anytime_scorer": float(np.mean(goals >= 1)),
            "shots_over_1_5": float(np.mean(shots > 1.5)),
            "shots_over_2_5": float(np.mean(shots > 2.5)),
            "shots_over_3_5": float(np.mean(shots > 3.5)),
            "assists_over_0_5": float(np.mean(assists >= 1)),
            "to_be_carded": float(np.mean(cards == 1)),
            "n": n,
        }
