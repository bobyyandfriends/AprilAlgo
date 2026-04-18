"""Streamlit runner: tuner run path with mocked price data and TunerRunner."""

from __future__ import annotations

import pandas as pd

import aprilalgo.ui.pages.tuner as tun
from aprilalgo.ui.pages.tuner import render

_df = pd.DataFrame(
    {
        "datetime": pd.date_range("2020-01-01", periods=30, freq="D"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
    }
)


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, progress_callback=None):
        if progress_callback:
            progress_callback(1, 1)
        return pd.DataFrame(
            [
                {
                    "combo_id": 0,
                    "rsi_period": 14,
                    "rsi_buy": 30,
                    "rsi_sell": 70,
                    "sma_period": 20,
                    "sharpe_ratio": 1.2,
                    "total_pnl": 100.0,
                    "total_return_pct": 0.1,
                    "num_trades": 5,
                    "win_rate_pct": 55.0,
                    "avg_win": 10.0,
                    "avg_loss": -5.0,
                    "profit_factor": 1.5,
                    "max_drawdown_pct": 2.0,
                    "sortino_ratio": 0.9,
                }
            ]
        )


tun.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]
tun.load_price_data = lambda s, t: _df.copy()  # type: ignore[method-assign]
tun.TunerRunner = _FakeRunner  # type: ignore[misc, assignment]

render()
