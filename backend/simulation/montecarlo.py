"""Vectorised Monte Carlo match simulator.
Samples goals from Poisson, cards from Poisson, corners from Normal,
then computes market probabilities for all common bet types.
"""
from __future__ import annotations
import numpy as np
from typing import Dict
from backend.ml.goals import GoalsModel


class MonteCarloSimulator:
    def __init__(self, goals_model: GoalsModel):
        self.goals_model = goals_model

    def run(self, features: Dict, n: int = 100_000, seed: int = None) -> Dict:
        rng = np.random.default_rng(seed)
        mu_h, mu_a = self.goals_model.expected_goals(features)

        # --- goals ---
        home_g = rng.poisson(mu_h, size=n)
        away_g = rng.poisson(mu_a, size=n)
        total_g = home_g + away_g

        # --- corners: rough estimate based on team attack/defence ---
        mu_corners = features.get("expected_corners", 10.0)
        corners = rng.normal(mu_corners, 2.5, size=n).clip(0)

        # --- cards ---
        mu_cards = features.get("expected_cards", 3.5)
        cards = rng.poisson(mu_cards, size=n)

        # --- result probabilities ---
        outcomes = np.sign(home_g - away_g)
        home_win = float(np.mean(outcomes == 1))
        draw = float(np.mean(outcomes == 0))
        away_win = float(np.mean(outcomes == -1))

        # --- goals markets ---
        over_0_5 = float(np.mean(total_g > 0.5))
        over_1_5 = float(np.mean(total_g > 1.5))
        over_2_5 = float(np.mean(total_g > 2.5))
        over_3_5 = float(np.mean(total_g > 3.5))
        over_4_5 = float(np.mean(total_g > 4.5))

        # --- btts ---
        btts = float(np.mean((home_g > 0) & (away_g > 0)))

        # --- corners markets ---
        over_8_5_corners = float(np.mean(corners > 8.5))
        over_9_5_corners = float(np.mean(corners > 9.5))
        over_10_5_corners = float(np.mean(corners > 10.5))

        # --- cards markets ---
        over_3_5_cards = float(np.mean(cards > 3.5))
        over_4_5_cards = float(np.mean(cards > 4.5))

        # --- clean sheets ---
        home_cs = float(np.mean(away_g == 0))
        away_cs = float(np.mean(home_g == 0))

        # --- top correct scores ---
        score_pairs = list(zip(home_g.tolist(), away_g.tolist()))
        from collections import Counter
        counts = Counter(score_pairs)
        top_scores = sorted(
            [(f"{h}-{a}", c / n) for (h, a), c in counts.items()],
            key=lambda x: -x[1]
        )[:10]

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_0_5": over_0_5,
            "over_1_5": over_1_5,
            "over_2_5": over_2_5,
            "over_3_5": over_3_5,
            "over_4_5": over_4_5,
            "btts": btts,
            "over_8_5_corners": over_8_5_corners,
            "over_9_5_corners": over_9_5_corners,
            "over_10_5_corners": over_10_5_corners,
            "over_3_5_cards": over_3_5_cards,
            "over_4_5_cards": over_4_5_cards,
            "home_clean_sheet": home_cs,
            "away_clean_sheet": away_cs,
            "top_correct_scores": top_scores,
            "n": n,
        }
