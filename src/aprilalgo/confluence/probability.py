"""Historical probability lookup — given a confluence score, what was the win rate?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_historical_probability(
    df: pd.DataFrame,
    forward_return_bars: int = 10,
    profit_target: float = 0.02,
    stop_loss: float = 0.01,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute historical win rates bucketed by confluence score.

    Looks forward *forward_return_bars* bars to determine if the trade would
    have hit the profit target or stop loss first.

    Parameters
    ----------
    df : DataFrame with ``confluence_net`` and ``close`` columns.
    forward_return_bars : How many bars to look ahead for outcome.
    profit_target : Required positive return for a "win".
    stop_loss : Negative return threshold for a "loss".
    bins : Number of confluence score buckets.

    Returns a summary DataFrame with columns:
    ``bin_label``, ``count``, ``win_rate``, ``avg_return``, ``avg_win``, ``avg_loss``
    """
    if "confluence_net" not in df.columns:
        raise ValueError("DataFrame must have a 'confluence_net' column. Run score_confluence first.")

    out = df.copy()
    close = out["close"].values
    n = len(close)

    outcomes = np.full(n, np.nan)
    returns = np.full(n, np.nan)

    for i in range(n - forward_return_bars):
        entry = close[i]
        if entry == 0:
            continue

        future_slice = close[i + 1 : i + 1 + forward_return_bars]
        bar_returns = (future_slice - entry) / entry

        hit_target = np.where(bar_returns >= profit_target)[0]
        hit_stop = np.where(bar_returns <= -stop_loss)[0]

        target_bar = hit_target[0] if len(hit_target) > 0 else float("inf")
        stop_bar = hit_stop[0] if len(hit_stop) > 0 else float("inf")

        if target_bar < stop_bar:
            outcomes[i] = 1.0
            returns[i] = profit_target
        elif stop_bar < target_bar:
            outcomes[i] = 0.0
            returns[i] = -stop_loss
        else:
            final_return = bar_returns[-1] if len(bar_returns) > 0 else 0.0
            outcomes[i] = 1.0 if final_return > 0 else 0.0
            returns[i] = final_return

    out["_outcome"] = outcomes
    out["_return"] = returns

    valid = out.dropna(subset=["_outcome", "confluence_net"])
    if valid.empty:
        return pd.DataFrame(columns=["bin_label", "count", "win_rate", "avg_return", "avg_win", "avg_loss"])

    valid = valid.copy()
    valid["_bin"] = pd.cut(valid["confluence_net"], bins=bins)

    summary = valid.groupby("_bin", observed=False).agg(
        count=("_outcome", "count"),
        win_rate=("_outcome", "mean"),
        avg_return=("_return", "mean"),
    ).reset_index()

    summary.rename(columns={"_bin": "bin_label"}, inplace=True)

    win_returns = valid[valid["_outcome"] == 1.0].groupby("_bin", observed=False)["_return"].mean()
    loss_returns = valid[valid["_outcome"] == 0.0].groupby("_bin", observed=False)["_return"].mean()

    summary["avg_win"] = summary["bin_label"].map(win_returns).fillna(0.0)
    summary["avg_loss"] = summary["bin_label"].map(loss_returns).fillna(0.0)

    return summary
