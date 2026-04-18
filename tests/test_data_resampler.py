"""Tests for aprilalgo.data.resampler."""

from __future__ import annotations

import pandas as pd

from aprilalgo.data.resampler import resample


def test_resample_5min_to_daily() -> None:
    idx = pd.date_range("2020-01-02 09:30", periods=78, freq="5min")
    o = list(range(78))
    h = [x + 1 for x in o]
    l = [x - 1 if x > 0 else 0 for x in o]
    c = list(range(1, 79))
    v = [100] * 78
    df = pd.DataFrame({"datetime": idx, "open": o, "high": h, "low": l, "close": c, "volume": v})
    out = resample(df, "D")
    assert len(out) >= 1
    row = out.iloc[0]
    assert row["open"] == o[0]
    assert row["close"] == c[-1]
    assert row["high"] == max(h)
    assert row["low"] == min(l)
    assert row["volume"] == sum(v)


def test_resample_drops_empty_bars() -> None:
    a = pd.date_range("2020-01-01 09:30", periods=5, freq="h")
    b = pd.date_range("2020-02-01 09:30", periods=5, freq="h")
    dt = list(a) + list(b)
    n = len(dt)
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [10] * n,
        }
    )
    out = resample(df, "D")
    assert "datetime" in out.columns
    assert len(out) <= 10


def test_resample_preserves_datetime_column() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=10, freq="5min"),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
        }
    )
    out = resample(df, "h")
    assert "datetime" in out.columns
