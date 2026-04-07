"""Backtesting engine: bar-by-bar simulation, portfolio tracking, and metrics."""

from aprilalgo.backtest.engine import run_backtest
from aprilalgo.backtest.trade import Trade
from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.backtest.metrics import calculate_metrics

__all__ = ["run_backtest", "Trade", "Portfolio", "calculate_metrics"]
