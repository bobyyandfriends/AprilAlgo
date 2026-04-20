"""Resample OHLCV data to different timeframes."""

from __future__ import annotations

import pandas as pd

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV *df* to a new frequency *rule*.

    Parameters
    ----------
    df : DataFrame with a ``datetime`` column and OHLCV columns.
    rule : pandas offset alias — e.g. ``"5min"``, ``"1h"``, ``"D"``, ``"W"``.

    Returns a new DataFrame sorted by datetime with no NaN bars.
    """
    tmp = df.copy()
    tmp.set_index("datetime", inplace=True)
    resampled = tmp.resample(rule).agg(OHLCV_AGG).dropna()
    resampled.reset_index(inplace=True)
    return resampled
