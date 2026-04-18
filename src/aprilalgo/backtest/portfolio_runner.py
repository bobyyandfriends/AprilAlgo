"""Run a strategy across multiple symbols (v0.4).

``initial_capital_total`` with an even split keeps aggregate starting capital
bounded. ``global_risk_cap`` in ``(0, 1]`` scales the pooled budget (only
``cap * global_risk_cap`` is divided across symbols).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aprilalgo.backtest.engine import run_backtest
from aprilalgo.data import load_price_data
from aprilalgo.strategies.base import BaseStrategy


def run_multi_symbol_backtests(
    strategy_for_symbol: Callable[[str], BaseStrategy],
    symbols: list[str],
    timeframe: str = "daily",
    *,
    data_dir: Any = None,
    initial_capital: float = 100_000.0,
    initial_capital_total: float | None = None,
    global_risk_cap: float | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Return ``symbol -> run_backtest`` result dict."""
    out: dict[str, dict[str, Any]] = {}
    n = max(len(symbols), 1)
    risk = 1.0 if global_risk_cap is None else float(global_risk_cap)
    if not (0 < risk <= 1):
        raise ValueError("global_risk_cap must be in (0, 1] when set")

    if initial_capital_total is not None:
        budget = float(initial_capital_total) * risk
        cap = budget / n
    else:
        cap = float(initial_capital) * risk

    for sym in symbols:
        df = load_price_data(sym, timeframe, data_dir=data_dir)
        strat = strategy_for_symbol(sym)
        out[sym] = run_backtest(
            strat,
            df,
            initial_capital=cap,
            commission=commission,
            slippage=slippage,
        )
    return out
