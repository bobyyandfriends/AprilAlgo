"""Multi-symbol runner capital semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aprilalgo.backtest.portfolio_runner import run_multi_symbol_backtests
from aprilalgo.strategies.rsi_sma import RsiSmaStrategy


def test_initial_capital_total_splits_across_symbols(monkeypatch):
    recorded: list[float] = []

    def capture_run(
        strategy,
        price_data,
        initial_capital: float = 100_000.0,
        **kwargs,
    ):
        recorded.append(initial_capital)
        return {
            "trades": pd.DataFrame(),
            "metrics": {"num_trades": 0},
            "equity": pd.DataFrame(),
            "strategy": "stub",
        }

    monkeypatch.setattr(
        "aprilalgo.backtest.portfolio_runner.run_backtest",
        capture_run,
    )
    fixtures = Path(__file__).resolve().parent / "fixtures"
    run_multi_symbol_backtests(
        lambda _s: RsiSmaStrategy(rsi_period=14, sma_period=20),
        ["TEST", "TEST"],
        timeframe="daily",
        data_dir=fixtures,
        initial_capital_total=100_000.0,
        global_risk_cap=1.0,
    )
    assert recorded == [50_000.0, 50_000.0]


def test_global_risk_cap_scales_budget(monkeypatch):
    recorded: list[float] = []

    def capture_run(strategy, price_data, initial_capital: float = 100_000.0, **kwargs):
        recorded.append(initial_capital)
        return {
            "trades": pd.DataFrame(),
            "metrics": {},
            "equity": pd.DataFrame(),
            "strategy": "stub",
        }

    monkeypatch.setattr(
        "aprilalgo.backtest.portfolio_runner.run_backtest",
        capture_run,
    )
    fixtures = Path(__file__).resolve().parent / "fixtures"
    run_multi_symbol_backtests(
        lambda _s: RsiSmaStrategy(rsi_period=14, sma_period=20),
        ["TEST", "TEST"],
        timeframe="daily",
        data_dir=fixtures,
        initial_capital_total=100_000.0,
        global_risk_cap=0.5,
    )
    assert recorded == [25_000.0, 25_000.0]
