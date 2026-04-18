"""Streamlit runner: dashboard with mocked data + backtest (exercises run path)."""

from __future__ import annotations

import pandas as pd

import aprilalgo.ui.pages.dashboard as dash
from aprilalgo.ui.pages.dashboard import render

_df = pd.DataFrame(
    {
        "datetime": pd.date_range("2020-01-01", periods=20, freq="D"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
    }
)


def _run_backtest(*_a, **_k):
    return {
        "metrics": {
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sortino_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        },
        "trades": pd.DataFrame(),
        "equity": pd.DataFrame({"time": _df["datetime"], "equity": [100_000.0] * len(_df)}),
    }


dash.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]
dash.load_price_data = lambda s, t: _df.copy()  # type: ignore[method-assign]
dash.run_backtest = _run_backtest  # type: ignore[method-assign]

render()
