"""Simple Moving Average (SMA)."""

from __future__ import annotations

import pandas as pd


def sma(
    df: pd.DataFrame,
    period: int = 50,
    column: str = "close",
) -> pd.DataFrame:
    """Add SMA value and bull/bear signal columns to *df*.

    Columns added:
    - ``sma_{period}`` — raw SMA value
    - ``sma_{period}_bull`` — True when price is above SMA
    - ``sma_{period}_bear`` — True when price is below SMA
    """
    out = df.copy()
    col_name = f"sma_{period}"
    out[col_name] = out[column].rolling(window=period, min_periods=period).mean()
    out[f"sma_{period}_bull"] = out[column] > out[col_name]
    out[f"sma_{period}_bear"] = out[column] < out[col_name]
    return out
