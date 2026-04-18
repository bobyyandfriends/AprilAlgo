"""Backtesting engine: bar-by-bar simulation, portfolio tracking, and metrics."""

from aprilalgo.backtest.engine import run_backtest
from aprilalgo.backtest.logger import SignalJsonlLogger, events_to_dataframe
from aprilalgo.backtest.metrics import calculate_metrics
from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.backtest.trade import Trade

__all__ = [
    "SignalJsonlLogger",
    "Trade",
    "Portfolio",
    "calculate_metrics",
    "events_to_dataframe",
    "run_backtest",
]
