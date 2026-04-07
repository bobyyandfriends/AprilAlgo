"""Volume Trend — confirm price moves with volume expansion."""

from __future__ import annotations

import pandas as pd


def volume_trend(
    df: pd.DataFrame,
    vol_period: int = 20,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Add volume-trend values and bull/bear signal columns to *df*.

    Columns added (parameterized by vol_period):
    - ``vol_{vol_period}_sma`` — rolling average volume
    - ``vol_{vol_period}_ratio`` — current volume / vol_sma
    - ``vol_{vol_period}_confirm`` — True when volume is expanding
    - ``vol_{vol_period}_bull`` — volume expands AND price rising
    - ``vol_{vol_period}_bear`` — volume expands AND price falling
    """
    out = df.copy()
    p = f"vol_{vol_period}"
    sma_col = f"{p}_sma"

    out[sma_col] = out["volume"].rolling(window=vol_period, min_periods=vol_period).mean()
    out[f"{p}_ratio"] = out["volume"] / (out[sma_col] + 1e-10)
    out[f"{p}_confirm"] = out[f"{p}_ratio"] >= threshold

    price_up = out["close"] > out["close"].shift(1)
    out[f"{p}_bull"] = out[f"{p}_confirm"] & price_up
    out[f"{p}_bear"] = out[f"{p}_confirm"] & ~price_up
    return out
