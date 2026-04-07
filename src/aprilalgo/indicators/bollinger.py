"""Bollinger Bands."""

from __future__ import annotations

import pandas as pd


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """Add Bollinger Bands values and bull/bear signal columns to *df*.

    Columns added (parameterized by period):
    - ``bb_{period}_mid``, ``bb_{period}_upper``, ``bb_{period}_lower``
    - ``bb_{period}_pct`` — %B position within bands (0 = lower, 1 = upper)
    - ``bb_{period}_bull`` — True when price touches/breaks below lower band
    - ``bb_{period}_bear`` — True when price touches/breaks above upper band
    """
    out = df.copy()
    p = f"bb_{period}"
    mid = f"{p}_mid"
    upper = f"{p}_upper"
    lower = f"{p}_lower"

    out[mid] = out[column].rolling(window=period, min_periods=period).mean()
    rolling_std = out[column].rolling(window=period, min_periods=period).std()
    out[upper] = out[mid] + std_dev * rolling_std
    out[lower] = out[mid] - std_dev * rolling_std

    band_width = out[upper] - out[lower]
    out[f"{p}_pct"] = (out[column] - out[lower]) / band_width.replace(0, float("nan"))
    out[f"{p}_bull"] = out[column] <= out[lower]
    out[f"{p}_bear"] = out[column] >= out[upper]
    return out
