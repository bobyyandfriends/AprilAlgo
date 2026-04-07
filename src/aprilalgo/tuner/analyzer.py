"""Analyze tuner results — rank combinations and check robustness."""

from __future__ import annotations

import pandas as pd


def analyze_results(
    results_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    top_n: int = 10,
) -> dict:
    """Analyze parameter optimization results.

    Returns a dict with:
    - ``best`` — the single best parameter combination
    - ``top_n`` — DataFrame of the top N combinations
    - ``robustness`` — robustness check for the best combination
    """
    if results_df.empty or metric not in results_df.columns:
        return {"best": {}, "top_n": pd.DataFrame(), "robustness": {}}

    valid = results_df.dropna(subset=[metric]).copy()
    if valid.empty:
        return {"best": {}, "top_n": pd.DataFrame(), "robustness": {}}

    sorted_df = valid.sort_values(metric, ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0].to_dict()
    top = sorted_df.head(top_n)

    robustness = _check_robustness(sorted_df, best_row, metric)

    return {
        "best": best_row,
        "top_n": top,
        "robustness": robustness,
    }


def _check_robustness(
    df: pd.DataFrame,
    best: dict,
    metric: str,
) -> dict:
    """Check if nearby parameters also perform well (not a fragile optimum).

    A robust result means small parameter changes don't destroy performance.
    """
    param_cols = [c for c in df.columns if c not in [
        "combo_id", "error", metric,
        "total_pnl", "total_return_pct", "num_trades", "win_rate_pct",
        "avg_win", "avg_loss", "profit_factor", "max_drawdown_pct",
        "sharpe_ratio", "sortino_ratio",
    ]]

    if not param_cols:
        return {"robust": True, "reason": "No parameters to vary"}

    best_metric = best.get(metric, 0)
    if best_metric == 0:
        return {"robust": False, "reason": "Best metric is zero"}

    neighbors = df.copy()
    for col in param_cols:
        if col in best and isinstance(best[col], (int, float)):
            val = best[col]
            tolerance = max(abs(val) * 0.3, 1)
            neighbors = neighbors[
                (neighbors[col] >= val - tolerance) &
                (neighbors[col] <= val + tolerance)
            ]

    if len(neighbors) < 2:
        return {
            "robust": False,
            "reason": "Too few nearby parameter combinations to assess",
            "neighbor_count": len(neighbors),
        }

    neighbor_mean = neighbors[metric].mean()
    degradation = 1.0 - (neighbor_mean / best_metric) if best_metric != 0 else 0.0

    is_robust = degradation < 0.3

    return {
        "robust": is_robust,
        "best_metric": round(best_metric, 4),
        "neighbor_mean": round(neighbor_mean, 4),
        "degradation_pct": round(degradation * 100, 2),
        "neighbor_count": len(neighbors),
        "reason": "Nearby params perform similarly" if is_robust else "Performance degrades with small param changes",
    }
