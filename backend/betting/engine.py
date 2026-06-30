"""Bet Builder Engine.
Finds value legs (model_p > implied_p + margin),
filters by correlation rules, sizes stakes with fractional Kelly.
"""
from __future__ import annotations
import uuid
import numpy as np
from typing import Dict, List, Optional
from backend.schemas.bet import BetResponse, LegSchema
from backend.utils.odds import implied_prob, expected_value, fractional_kelly

# Pairs of outcomes considered highly correlated — reject combos containing both
CORRELATED_PAIRS = [
    {"home_win", "over_3_5"},
    {"away_win", "over_3_5"},
    {"home_win", "btts"},
    {"over_3_5", "over_4_5"},
    {"over_2_5", "over_3_5"},
]


class BetEngine:
    def build_suggestions(
        self,
        model_probs: Dict[str, float],
        odds: Dict[str, float],
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.03,
        max_legs: int = 4,
    ) -> List[BetResponse]:
        results: List[BetResponse] = []

        # 1. Evaluate individual legs
        value_legs: List[LegSchema] = []
        for market, model_p in model_probs.items():
            if market not in odds:
                continue
            odd = odds[market]
            imp = implied_prob(odd)
            ev = expected_value(model_p, odd)
            if ev < min_ev:
                continue
            conf = self._confidence(model_p, ev)
            value_legs.append(LegSchema(
                name=market,
                probability=round(model_p, 4),
                implied_probability=round(imp, 4),
                odds=odd,
                ev=round(ev, 4),
                confidence=round(conf, 3),
            ))

        if not value_legs:
            return []

        # 2. Single-leg bets
        for leg in value_legs:
            kf = fractional_kelly(leg.probability, leg.odds, kelly_fraction)
            results.append(BetResponse(
                bet_id=str(uuid.uuid4()),
                legs=[leg],
                combined_probability=round(leg.probability, 4),
                combined_ev=round(leg.ev, 4),
                kelly_fraction=round(kf, 4),
                stake_advice=round(bankroll * kf, 2),
                rejected=False,
                rejection_reason=None,
            ))

        # 3. Multi-leg combos (up to max_legs)
        if len(value_legs) >= 2:
            for i in range(len(value_legs)):
                for j in range(i + 1, len(value_legs)):
                    legs_combo = [value_legs[i], value_legs[j]]
                    rejected, reason = self._check_correlation(legs_combo)
                    combined_p = np.prod([l.probability for l in legs_combo])
                    combo_odd = np.prod([l.odds for l in legs_combo])
                    combo_ev = expected_value(float(combined_p), float(combo_odd))
                    kf = fractional_kelly(float(combined_p), float(combo_odd), kelly_fraction)
                    results.append(BetResponse(
                        bet_id=str(uuid.uuid4()),
                        legs=legs_combo,
                        combined_probability=round(float(combined_p), 4),
                        combined_ev=round(combo_ev, 4),
                        kelly_fraction=round(kf, 4),
                        stake_advice=round(bankroll * kf, 2) if not rejected else 0.0,
                        rejected=rejected,
                        rejection_reason=reason,
                    ))

        return sorted(results, key=lambda x: -x.combined_ev)

    def _confidence(self, model_p: float, ev: float) -> float:
        base = 0.4 + 0.3 * np.tanh(ev * 5)
        if model_p > 0.7 or model_p < 0.1:
            base *= 0.8
        return float(np.clip(base, 0.1, 0.95))

    def _check_correlation(
        self, legs: List[LegSchema]
    ) -> tuple[bool, Optional[str]]:
        names = {l.name for l in legs}
        for pair in CORRELATED_PAIRS:
            if pair.issubset(names):
                return True, f"Correlated legs: {pair}"
        avg_conf = np.mean([l.confidence for l in legs])
        if avg_conf < 0.3:
            return True, "Avg confidence too low for combo"
        return False, None
