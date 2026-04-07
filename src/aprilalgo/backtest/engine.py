"""Bar-by-bar backtesting engine."""

from __future__ import annotations

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.backtest.metrics import calculate_metrics
from aprilalgo.strategies.base import BaseStrategy


def run_backtest(
    strategy: BaseStrategy,
    price_data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> dict:
    """Run a full backtest of *strategy* on *price_data*.

    Returns a dict with keys ``trades``, ``metrics``, ``equity``, ``strategy``.
    """
    portfolio = Portfolio(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )

    strategy.init(price_data)

    for idx in range(len(price_data)):
        row = price_data.iloc[idx]
        strategy.on_bar(idx, row, portfolio)
        portfolio.record_equity(row["datetime"], row["close"])

    # Force-close any open positions at the last bar
    if portfolio.has_open_position:
        last = price_data.iloc[-1]
        for trade in list(portfolio.open_positions):
            portfolio.close_trade(trade, last["datetime"], last["close"])

    trades_df = portfolio.get_trades_df()
    metrics = calculate_metrics(trades_df, initial_capital=initial_capital)
    equity_df = portfolio.get_equity_df()

    return {
        "trades": trades_df,
        "metrics": metrics,
        "equity": equity_df,
        "strategy": strategy.name,
    }
