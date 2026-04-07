"""Relative Strength Index (RSI)."""

from __future__ import annotations

import pandas as pd


def rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """Add RSI value and bull/bear signal columns to *df*.

    Columns added (parameterized to prevent collision on multiple calls):
    - ``rsi_{period}`` — raw RSI value (0-100)
    - ``rsi_{period}_bull`` — True when RSI < oversold
    - ``rsi_{period}_bear`` — True when RSI > overbought
    """
    out = df.copy()
    delta = out[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    val_col = f"rsi_{period}"
    out[val_col] = 100.0 - (100.0 / (1.0 + rs))
    out[f"rsi_{period}_bull"] = out[val_col] < oversold
    out[f"rsi_{period}_bear"] = out[val_col] > overbought
    return out
