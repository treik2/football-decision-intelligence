"""Walk-forward backtesting for win probability and goals models."""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def rolling_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_factory,
    min_train_rows: int = 500,
    step_rows: int = 100,
) -> pd.DataFrame:
    """
    Walk-forward backtest.
    Trains on rows [0:i] and predicts rows [i:i+step_rows].
    Returns a DataFrame with per-window metrics.
    """
    df = df.sort_values("kickoff_ts").reset_index(drop=True)
    records = []
    n = len(df)

    for i in range(min_train_rows, n, step_rows):
        train = df.iloc[:i]
        test = df.iloc[i:i + step_rows]
        if test.empty:
            break

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test  = test[feature_cols].values
        y_test  = test[target_col].values

        model = model_factory()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        ll = log_loss(y_test, proba, labels=[0, 1, 2])
        brier = float(np.mean([
            brier_score_loss((y_test == c).astype(int), proba[:, c])
            for c in range(proba.shape[1])
        ]))
        records.append({
            "train_end_idx": i,
            "test_start": test["kickoff_ts"].iloc[0],
            "test_end": test["kickoff_ts"].iloc[-1],
            "n_train": i,
            "n_test": len(test),
            "log_loss": ll,
            "brier_score": brier,
        })
        logger.info(f"Window {i}: log_loss={ll:.4f} brier={brier:.4f}")

    return pd.DataFrame(records)


def roi_backtest(
    df: pd.DataFrame,
    pred_col: str,
    odds_col: str,
    result_col: str,
    min_ev: float = 0.03,
    kelly_fraction: float = 0.25,
    bankroll_start: float = 1000.0,
) -> Dict:
    """
    Simulates betting on all rows where EV > min_ev.
    Returns ROI, hit rate, final bankroll, max drawdown.
    """
    bankroll = bankroll_start
    peak = bankroll
    max_dd = 0.0
    bets, wins = 0, 0

    for _, row in df.iterrows():
        p = row[pred_col]
        o = row[odds_col]
        result = row[result_col]  # 1 = bet won, 0 = lost
        ev = p * (o - 1) - (1 - p)
        if ev < min_ev:
            continue
        b = o - 1.0
        q = 1.0 - p
        k = max(0.0, (p * b - q) / b) * kelly_fraction
        stake = bankroll * k
        if result == 1:
            bankroll += stake * b
            wins += 1
        else:
            bankroll -= stake
        bets += 1
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd

    roi = (bankroll - bankroll_start) / bankroll_start
    return {
        "bets": bets,
        "wins": wins,
        "hit_rate": wins / bets if bets > 0 else 0.0,
        "roi": roi,
        "final_bankroll": bankroll,
        "max_drawdown": max_dd,
    }
