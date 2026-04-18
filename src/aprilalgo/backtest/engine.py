"""Bar-by-bar backtesting engine."""

from __future__ import annotations

import pandas as pd

from aprilalgo.backtest.metrics import calculate_metrics
from aprilalgo.backtest.portfolio import Portfolio
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

    loop_df = getattr(strategy, "_backtest_bars_df", None)
    if loop_df is None:
        # Back-compat fallback for external strategies that predate the
        # ``_backtest_bars_df`` contract documented on :class:`BaseStrategy`.
        loop_df = price_data

    if not isinstance(loop_df, pd.DataFrame):
        raise TypeError(
            f"{type(strategy).__name__}._backtest_bars_df must be a DataFrame "
            f"after init(); got {type(loop_df).__name__}"
        )
    if len(loop_df) == 0:
        raise ValueError(
            f"{type(strategy).__name__}.init() produced an empty loop frame "
            "(indicator pipeline may have dropped all warm-up rows)."
        )
    for required in ("datetime", "close"):
        if required not in loop_df.columns:
            raise ValueError(
                f"{type(strategy).__name__}._backtest_bars_df is missing required "
                f"column {required!r}; backtest loop cannot proceed."
            )
    if getattr(strategy, "_backtest_frame_matches_input", True) and len(loop_df) != len(price_data):
        raise ValueError(
            f"{type(strategy).__name__} declared identity loop-frame but produced "
            f"{len(loop_df)} rows from {len(price_data)} input bars. Either fix the "
            "indicator pipeline to preserve row count or set "
            "``_backtest_frame_matches_input = False`` on the strategy class."
        )

    for idx in range(len(loop_df)):
        row = loop_df.iloc[idx]
        strategy.on_bar(idx, row, portfolio)
        portfolio.record_equity(row["datetime"], row["close"])

    # Force-close any open positions at the last bar
    if portfolio.has_open_position:
        last = loop_df.iloc[-1]
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
