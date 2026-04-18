"""Streamlit runner: charts page with mocked symbols and OHLCV (no models/)."""

from __future__ import annotations

import pandas as pd

import aprilalgo.ui.pages.charts.page as ch
from aprilalgo.ui.pages.charts.page import render

_df = pd.DataFrame(
    {
        "datetime": pd.date_range("2020-01-01", periods=200, freq="D"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
    }
)

ch.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]
ch.discover_model_dirs = lambda: []  # type: ignore[method-assign]
ch._cached_load_price = lambda s, t: _df.copy()  # type: ignore[method-assign]

render()
