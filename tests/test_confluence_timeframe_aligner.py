"""Tests for aprilalgo.confluence.timeframe_aligner."""

from __future__ import annotations

import pandas as pd
import pytest

from aprilalgo.confluence.timeframe_aligner import align_timeframes


def test_duplicates_raise() -> None:
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "close": [1.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate datetimes"):
        align_timeframes(base, {})


def test_non_monotonic_raises() -> None:
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
            "close": [3.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="monotonic"):
        align_timeframes(base, {})


def test_higher_tf_duplicates_raise() -> None:
    base = pd.DataFrame({"datetime": pd.to_datetime(["2020-01-01 09:30", "2020-01-01 09:35"]), "close": [1.0, 2.0]})
    high = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "rsi_bull": [1, 0],
            "rsi_bear": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicate datetimes"):
        align_timeframes(base, {"daily": high})


def test_higher_tf_non_monotonic_raises() -> None:
    base = pd.DataFrame({"datetime": pd.to_datetime(["2020-01-01 09:30", "2020-01-01 09:35"]), "close": [1.0, 2.0]})
    high = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "rsi_bull": [1, 0],
            "rsi_bear": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        align_timeframes(base, {"daily": high})


def test_forward_fill_and_prefix() -> None:
    # align_timeframes uses index join + ffill: higher-TF rows must share exact
    # datetime index labels with base rows for a match; ffill then carries values
    # across finer bars until the next matching higher-TF timestamp.
    d0 = pd.Timestamp("2020-01-01 00:00")
    d1 = pd.Timestamp("2020-01-02 00:00")
    intraday = [d0 + pd.Timedelta(minutes=5 * i) for i in range(1, 6)]
    base = pd.DataFrame(
        {
            "datetime": [d0, *intraday, d1, d1 + pd.Timedelta(hours=9, minutes=30)],
            "close": range(8),
        }
    )
    high = pd.DataFrame(
        {
            "datetime": pd.to_datetime([d0, d1]),
            "rsi_bull": [1, 0],
            "rsi_bear": [0, 1],
        }
    )
    out = align_timeframes(base, {"daily": high})
    assert "daily_rsi_bull" in out.columns
    assert "daily_rsi_bear" in out.columns
    assert out["daily_rsi_bull"].notna().all()
    # After d1, daily signal is bull=0 / bear=1; last intraday row should ffill that.
    assert out["daily_rsi_bull"].iloc[-1] == 0
    assert out["daily_rsi_bear"].iloc[-1] == 1


def test_explicit_signal_cols() -> None:
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:05"]),
            "close": [1.0, 2.0],
        }
    )
    high = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "rsi_14": [50.0, 60.0],
            "rsi_bull": [1, 0],
        }
    )
    out = align_timeframes(base, {"daily": high}, signal_cols=["rsi_14"])
    assert "daily_rsi_14" in out.columns
    assert "daily_rsi_bull" not in out.columns


def test_no_matching_cols_is_noop() -> None:
    base = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:05"]),
            "close": [1.0, 2.0],
        }
    )
    high = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "volume": [1e6, 2e6],
        }
    )
    out = align_timeframes(base, {"daily": high})
    assert list(out.columns) == list(base.columns)
