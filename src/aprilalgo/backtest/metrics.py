"""Performance metrics computed from a DataFrame of closed trades."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    trading_days_per_year: int = 252,
) -> dict:
    """Compute key backtest statistics from *trades*.

    Expected columns: ``realized_pnl``, and optionally ``entry_time`` / ``exit_time``.
    """
    if trades.empty:
        return _empty_metrics()

    pnl = trades["realized_pnl"]
    num = len(pnl)
    total_pnl = pnl.sum()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    # Equity curve for drawdown
    equity = np.empty(num + 1)
    equity[0] = initial_capital
    for i, p in enumerate(pnl):
        equity[i + 1] = equity[i] + p
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.where(running_max == 0, 1, running_max)
    max_drawdown = float(drawdowns.max())

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Approximate Sharpe from per-trade returns
    trade_returns = pnl / initial_capital
    sharpe = _annualized_sharpe(trade_returns, trading_days_per_year)
    sortino = _annualized_sortino(trade_returns, trading_days_per_year)

    return {
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / initial_capital * 100, 2),
        "num_trades": num,
        "win_rate_pct": round(len(wins) / num * 100, 2),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
    }


# ------------------------------------------------------------------
def _annualized_sharpe(returns: pd.Series, days: int) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(days))


def _annualized_sortino(returns: pd.Series, days: int) -> float:
    if len(returns) < 2:
        return 0.0
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0 if returns.mean() <= 0 else float("inf")
    downside_std = neg.std()
    if downside_std == 0:
        return 0.0 if returns.mean() <= 0 else float("inf")
    return float(returns.mean() / downside_std * np.sqrt(days))


def _empty_metrics() -> dict:
    return {
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "num_trades": 0,
        "win_rate_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
    }
