"""Information-driven bar construction tests."""

from __future__ import annotations

import pandas as pd

import pytest

from aprilalgo.data.bars import build_dollar_bars, build_tick_bars, build_volume_bars


def _sample_df(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=i),
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 10 + i,
            }
        )
    return pd.DataFrame(rows)


def test_tick_bars_shape_and_sort():
    df = _sample_df(11)
    out = build_tick_bars(df, threshold=3)
    assert len(out) == 4
    assert out["datetime"].is_monotonic_increasing
    assert {"open", "high", "low", "close", "volume"} <= set(out.columns)


def test_volume_bars_respects_threshold():
    df = _sample_df(12)
    out = build_volume_bars(df, threshold=25)
    assert len(out) > 0
    assert (out["volume"] > 0).all()


def test_dollar_bars_contains_dollar_value():
    df = _sample_df(10)
    out = build_dollar_bars(df, threshold=2000)
    assert "dollar_value" in out.columns
    assert (out["dollar_value"] >= 0).all()


def test_build_bars_rejects_non_positive_threshold() -> None:
    df = _sample_df(5)
    with pytest.raises(ValueError, match="threshold"):
        build_tick_bars(df, threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        build_volume_bars(df, threshold=-1.0)
