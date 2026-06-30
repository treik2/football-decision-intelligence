import uuid
import numpy as np
from typing import Dict, List, Optional
from backend.betting.ev import expected_value, fractional_kelly
from backend.betting.correlation import correlation_score, is_correlated


class BetEngine:
    """
    Builds and filters bet suggestions from model probabilities and market odds.
    Applies EV filtering, Kelly sizing, and correlation rejection.
    """

    # Minimum thresholds
    MIN_CONFIDENCE = 0.40
    MAX_CORR_SCORE = 0.65

    def build(
        self,
        match_id: str,
        win_probs: Dict[str, float],
        sim_result: Dict,
        odds: Dict[str, float],
        bankroll: float = 1000.0,
        min_ev: float = 0.02,
        kelly_fraction: float = 0.25,
    ) -> List[Dict]:
        candidates = self._build_candidates(win_probs, sim_result, odds)
        value_legs = [
            c for c in candidates
            if c["ev"] >= min_ev and c["confidence"] >= self.MIN_CONFIDENCE
        ]

        suggestions = []
        for leg in value_legs:
            k = fractional_kelly(leg["model_prob"], leg["odds"], fraction=kelly_fraction)
            if k <= 0:
                continue
            suggestions.append({
                "bet_id": str(uuid.uuid4()),
                "market": leg["market"],
                "model_prob": round(leg["model_prob"], 4),
                "implied_prob": round(leg["implied_prob"], 4),
                "ev": round(leg["ev"], 4),
                "odds": leg["odds"],
                "confidence": round(leg["confidence"], 4),
                "kelly_fraction": round(k, 4),
                "stake_advice": round(bankroll * k, 2),
                "type": "single",
            })

        # Try parlay combos of top 2-3 value legs (low correlation only)
        if len(value_legs) >= 2:
            combos = self._safe_combos(value_legs, max_legs=3)
            for combo in combos:
                p_combo = float(np.prod([c["model_prob"] for c in combo]))
                o_combo = float(np.prod([c["odds"] for c in combo]))
                ev_combo = expected_value(p_combo, o_combo)
                k_combo = fractional_kelly(p_combo, o_combo, fraction=kelly_fraction * 0.5)
                if ev_combo >= min_ev and k_combo > 0:
                    suggestions.append({
                        "bet_id": str(uuid.uuid4()),
                        "market": " + ".join(c["market"] for c in combo),
                        "model_prob": round(p_combo, 4),
                        "implied_prob": round(1.0 / o_combo, 4),
                        "ev": round(ev_combo, 4),
                        "odds": round(o_combo, 2),
                        "confidence": round(min(c["confidence"] for c in combo), 4),
                        "kelly_fraction": round(k_combo, 4),
                        "stake_advice": round(bankroll * k_combo, 2),
                        "type": "parlay",
                        "legs": [c["market"] for c in combo],
                    })

        return sorted(suggestions, key=lambda x: x["ev"], reverse=True)

    def _build_candidates(self, win_probs, sim_result, odds) -> List[Dict]:
        market_map = {
            "home_win":  win_probs.get("home", 0.0),
            "draw":      win_probs.get("draw", 0.0),
            "away_win":  win_probs.get("away", 0.0),
            "over_2_5":  sim_result.get("over_2_5", 0.0),
            "under_2_5": 1.0 - sim_result.get("over_2_5", 0.0),
            "over_1_5":  sim_result.get("over_1_5", 0.0),
            "over_3_5":  sim_result.get("over_3_5", 0.0),
            "btts":      sim_result.get("btts", 0.0),
            "btts_no":   sim_result.get("btts_no", 0.0),
            "ah_home":   sim_result.get("ah_home_minus_half", 0.0),
            "ah_away":   sim_result.get("ah_away_plus_half", 0.0),
        }
        candidates = []
        for market, model_prob in market_map.items():
            if market not in odds or model_prob <= 0:
                continue
            odd = float(odds[market])
            impl_p = 1.0 / odd
            ev = expected_value(model_prob, odd)
            conf = self._confidence(market, model_prob, sim_result)
            candidates.append({
                "market": market,
                "model_prob": model_prob,
                "implied_prob": impl_p,
                "odds": odd,
                "ev": ev,
                "confidence": conf,
            })
        return candidates

    def _confidence(self, market: str, prob: float, sim_result: Dict) -> float:
        # Confidence is higher when prob is far from 0.5 (clear signal)
        dist = abs(prob - 0.5)
        base = 0.4 + dist
        # Penalise rare markets
        if market in ("over_4_5", "correct_score"):
            base *= 0.75
        return float(min(base, 0.95))

    def _safe_combos(self, legs: List[Dict], max_legs: int = 3) -> List[List[Dict]]:
        from itertools import combinations
        safe = []
        for r in range(2, max_legs + 1):
            for combo in combinations(legs, r):
                if not is_correlated([c["market"] for c in combo], threshold=self.MAX_CORR_SCORE):
                    safe.append(list(combo))
        return safe
