"""Relative Strength Index (RSI)."""

from __future__ import annotations

import pandas as pd


def rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
    oversold: float = 30.0,
    overbought: float = 70.0,
    *,
    mode: str = "sma",
) -> pd.DataFrame:
    """Add RSI value and bull/bear signal columns to *df*.

    Columns added (parameterized to prevent collision on multiple calls):

    - ``rsi_{period}`` — raw RSI value (0-100)
    - ``rsi_{period}_bull`` — True when RSI < oversold
    - ``rsi_{period}_bear`` — True when RSI > overbought

    Parameters
    ----------
    mode
        Smoothing method for the gain / loss averages:

        * ``"sma"`` (default, historical behaviour) uses a simple rolling mean.
          Kept as the default so existing strategy parameters, tuner outputs,
          and backtests remain reproducible.
        * ``"wilder"`` uses Wilder's recursive smoothing
          ``ewm(alpha=1/period, adjust=False)``, matching classic RSI(14) from
          TA-Lib / TradingView. Set this when porting strategies from other
          platforms or when consumers expect textbook RSI values (§AUDIT B15).
    """
    mode_norm = mode.lower()
    if mode_norm not in {"sma", "wilder"}:
        raise ValueError(f"rsi mode must be 'sma' or 'wilder', got {mode!r}")

    out = df.copy()
    delta = out[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    if mode_norm == "wilder":
        # Wilder smoothing is an exponentially weighted moving average with
        # ``alpha = 1/period`` and no bias correction (``adjust=False``).
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    else:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    val_col = f"rsi_{period}"
    out[val_col] = 100.0 - (100.0 / (1.0 + rs))
    out[f"rsi_{period}_bull"] = out[val_col] < oversold
    out[f"rsi_{period}_bear"] = out[val_col] > overbought
    return out
