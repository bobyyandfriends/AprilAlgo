"""Tests for the backtesting engine, strategies, and metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from aprilalgo.backtest import calculate_metrics, run_backtest
from aprilalgo.backtest.metrics_v2 import compute_metrics_from_equity
from aprilalgo.data import load_price_data
from aprilalgo.strategies import STRATEGIES, ConfigurableStrategy, DeMarkConfluenceStrategy, RsiSmaStrategy


@pytest.fixture
def price_data():
    return load_price_data("AAPL", "daily")


def _assert_backtest_invariants(result: dict, *, initial_capital: float = 100_000.0) -> None:
    assert "metrics" in result
    assert "trades" in result
    assert "equity" in result
    assert isinstance(result["metrics"], dict)
    eq = result["equity"]
    assert isinstance(eq, pd.DataFrame)
    assert {"time", "equity"} <= set(eq.columns)
    assert eq["equity"].iloc[0] == pytest.approx(initial_capital, rel=0.02)
    tser = pd.to_datetime(eq["time"], errors="coerce")
    assert bool(tser.is_monotonic_increasing)
    trades = result["trades"]
    assert isinstance(trades, pd.DataFrame)
    m = result["metrics"]
    assert int(m["num_trades"]) == len(trades)
    assert 0.0 <= float(m["max_drawdown_pct"]) <= 100.0
    if not trades.empty:
        assert "entry_time" in trades.columns and "exit_time" in trades.columns
        for _, row in trades.iterrows():
            assert row["entry_time"] <= row["exit_time"]
            assert math.isfinite(float(row["realized_pnl"]))
    eq_delta = float(eq["equity"].iloc[-1]) - float(eq["equity"].iloc[0])
    assert float(m["total_pnl"]) == pytest.approx(eq_delta, rel=0.02)
    eq_v2 = eq.rename(columns={"time": "datetime"})
    m2 = compute_metrics_from_equity(eq_v2, trades=trades, initial_capital=initial_capital)
    assert float(m["total_return_pct"]) == pytest.approx(float(m2["total_return_pct"]), abs=0.5)


class TestBacktestEngine:
    def test_rsi_sma_produces_result(self, price_data):
        strat = RsiSmaStrategy(rsi_period=14, sma_period=20, rsi_buy=35, rsi_sell=65)
        result = run_backtest(strat, price_data)
        _assert_backtest_invariants(result)

    def test_demark_confluence_produces_result(self, price_data):
        strat = DeMarkConfluenceStrategy(confluence_threshold=0.1, sma_period=20)
        result = run_backtest(strat, price_data)
        _assert_backtest_invariants(result)

    def test_configurable_strategy_produces_result(self, price_data):
        strat = ConfigurableStrategy(
            indicators=[
                {"name": "rsi", "period": 14},
                {"name": "sma", "period": 20},
                {"name": "demark"},
            ],
            entry_threshold=0.2,
        )
        result = run_backtest(strat, price_data)
        _assert_backtest_invariants(result)
        assert isinstance(result["metrics"]["num_trades"], int)

    def test_configurable_default_indicators(self, price_data):
        strat = ConfigurableStrategy()
        result = run_backtest(strat, price_data)
        _assert_backtest_invariants(result)

    def test_strategies_dict_complete(self):
        assert "rsi_sma" in STRATEGIES
        assert "demark_confluence" in STRATEGIES
        assert "configurable" in STRATEGIES
        assert "ml_xgboost" in STRATEGIES


def test_backtest_equity_bounded_by_price_extremes(price_data):
    strat = RsiSmaStrategy(rsi_period=14, sma_period=20, rsi_buy=35, rsi_sell=65)
    result = run_backtest(strat, price_data)
    eq = result["equity"]
    assert float(eq["equity"].min()) > 0.0


class TestMetrics:
    def test_empty_trades(self):
        m = calculate_metrics(pd.DataFrame())
        assert m["num_trades"] == 0
        assert m["total_pnl"] == 0.0

    def test_metrics_keys(self, price_data):
        strat = RsiSmaStrategy(rsi_period=14, sma_period=20, rsi_buy=35, rsi_sell=65)
        result = run_backtest(strat, price_data)
        m = result["metrics"]
        for key in [
            "total_pnl",
            "total_return_pct",
            "num_trades",
            "win_rate_pct",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "max_drawdown_pct",
            "sharpe_ratio",
            "sortino_ratio",
        ]:
            assert key in m, f"Missing metric: {key}"
