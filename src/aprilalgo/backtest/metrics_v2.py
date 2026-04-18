"""Equity-curve-based performance metrics (v2; §AUDIT B2).

The v1 :mod:`aprilalgo.backtest.metrics` computes per-trade returns as
``pnl / initial_capital`` and annualises with ``sqrt(252)`` regardless of trade
frequency. Both assumptions are defensible for short, dense backtests but break
down for longer or low-turnover runs:

* ``pnl / initial_capital`` ignores compounding — a strategy that doubles its
  capital in year 1 still has year 2 returns measured against the starting
  equity, understating Sharpe's denominator once drawdowns start attacking
  the compounded base.
* ``sqrt(252)`` annualisation assumes each observation is a single trading
  day. A weekly rebalancer that produces 50 observations is then annualised
  as if it had 252 — overstating Sharpe by ~``sqrt(252/50)`` ≈ 2.2x.

This module is additive; the older module is unchanged so historical reports
remain reproducible. New code should prefer :func:`compute_metrics_from_equity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reference trading-days-per-year constants for supported sampling frequencies.
# For intraday data we approximate with trading-days * average-intraday-bar count.
_ANNUALISATION_PERIODS: dict[str, float] = {
    "daily": 252.0,
    "weekly": 52.0,
    "monthly": 12.0,
    "hourly": 252.0 * 6.5,
    "minute": 252.0 * 6.5 * 60.0,
}


def infer_periods_per_year(equity: pd.DataFrame) -> float:
    """Infer annualisation factor from the median spacing of ``equity['datetime']``.

    Falls back to ``252`` (daily) when the DataFrame is too short to infer
    or ``datetime`` is missing.
    """
    if "datetime" not in equity.columns or len(equity) < 3:
        return 252.0
    dt = pd.to_datetime(equity["datetime"], errors="coerce").dropna()
    if len(dt) < 3:
        return 252.0
    diffs = dt.diff().dropna()
    if diffs.empty:
        return 252.0
    median_delta = diffs.median()
    if pd.isna(median_delta):
        return 252.0
    seconds = float(median_delta.total_seconds())
    if seconds <= 0.0:
        return 252.0
    days = seconds / 86400.0
    # Map contiguous-calendar spacings to trading-period conventions.
    if days >= 180.0:
        return 1.0
    if days >= 25.0:
        return _ANNUALISATION_PERIODS["monthly"]
    if days >= 5.5:
        return _ANNUALISATION_PERIODS["weekly"]
    if days >= 0.9:
        return _ANNUALISATION_PERIODS["daily"]
    # Intraday: approximate 6.5 trading hours per day.
    bars_per_day = 86400.0 / seconds
    intraday_per_day = min(bars_per_day, 6.5 * 3600.0 / seconds)
    return 252.0 * intraday_per_day


def compute_metrics_from_equity(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    initial_capital: float = 100_000.0,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    """Equity-curve-based metrics with correct compounding + annualisation.

    Parameters
    ----------
    equity
        DataFrame with at least ``datetime`` and ``equity`` columns. Must be
        sorted by ``datetime`` ascending.
    trades
        Optional closed-trades DataFrame (for win-rate / profit-factor style
        metrics that require per-trade resolution).
    initial_capital
        Starting equity; used for total-return_pct and drawdown scaling.
    periods_per_year
        Override the annualisation factor. When ``None`` (the default) it is
        inferred from the median ``datetime`` spacing — weekly bars pick 52,
        daily 252, etc.
    risk_free_rate
        Annualised risk-free rate; subtracted from the mean return before
        Sharpe / Sortino (expressed as a decimal, e.g. ``0.04`` = 4%).
    """
    out: dict[str, float | int] = _empty_metrics()
    if equity.empty or "equity" not in equity.columns:
        return out

    eq = equity["equity"].astype(float).to_numpy()
    if len(eq) < 2:
        out["total_return_pct"] = 0.0
        return out

    ppy = float(periods_per_year) if periods_per_year is not None else infer_periods_per_year(equity)

    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / np.where(running_max == 0, 1.0, running_max)
    max_drawdown = float(drawdowns.max())

    total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] != 0.0 else 0.0

    # Compounded simple returns between bars; geometric compounding is what Sharpe
    # expects as the underlying process, so we feed simple returns into the ratio
    # and derive CAGR from the compounded growth factor.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    rf_per_period = risk_free_rate / ppy if ppy > 0 else 0.0
    excess = r - rf_per_period

    sharpe = _annualised_sharpe(excess, ppy)
    sortino = _annualised_sortino(excess, ppy)

    # CAGR from the compounded growth path; guards against zero / negative equity.
    cagr = 0.0
    if len(r) > 0 and eq[0] > 0.0 and ppy > 0:
        growth = float(eq[-1] / eq[0])
        if growth > 0.0:
            years = len(r) / ppy
            cagr = float(growth ** (1.0 / years) - 1.0) if years > 0 else 0.0

    out.update(
        {
            "total_return_pct": round(total_return * 100.0, 4),
            "cagr_pct": round(cagr * 100.0, 4),
            "max_drawdown_pct": round(max_drawdown * 100.0, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "periods_per_year": round(ppy, 4),
            "n_observations": int(len(r)),
        }
    )

    if trades is not None and not trades.empty and "realized_pnl" in trades.columns:
        pnl = trades["realized_pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        num = int(len(pnl))
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0.0 else float("inf")
        out.update(
            {
                "num_trades": num,
                "total_pnl": round(float(pnl.sum()), 2),
                "win_rate_pct": round(len(wins) / num * 100.0, 4) if num else 0.0,
                "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
                "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
                "profit_factor": round(pf, 4) if np.isfinite(pf) else float("inf"),
            }
        )

    return out


def _annualised_sharpe(excess: np.ndarray, periods_per_year: float) -> float:
    if len(excess) < 2 or periods_per_year <= 0:
        return 0.0
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def _annualised_sortino(excess: np.ndarray, periods_per_year: float) -> float:
    if len(excess) < 2 or periods_per_year <= 0:
        return 0.0
    neg = excess[excess < 0]
    if len(neg) < 2:
        return 0.0 if np.mean(excess) <= 0 else float("inf")
    downside_std = float(np.std(neg, ddof=1))
    if downside_std == 0.0:
        return 0.0 if np.mean(excess) <= 0 else float("inf")
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def _empty_metrics() -> dict[str, float | int]:
    return {
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "num_trades": 0,
        "win_rate_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "periods_per_year": 0.0,
        "n_observations": 0,
    }


__all__ = [
    "compute_metrics_from_equity",
    "infer_periods_per_year",
]
