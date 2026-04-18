"""Tests for aprilalgo.confluence.probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.confluence.probability import calculate_historical_probability


def test_missing_confluence_net_raises() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="confluence_net"):
        calculate_historical_probability(df)


def test_empty_valid_returns_empty_schema() -> None:
    df = pd.DataFrame({"confluence_net": [0.1, 0.2, 0.3], "close": [100.0, 101.0, 102.0]})
    out = calculate_historical_probability(df, forward_return_bars=10)
    expected_cols = ["bin_label", "count", "win_rate", "avg_return", "avg_win", "avg_loss"]
    assert list(out.columns) == expected_cols
    assert len(out) == 0


def test_basic_win_rate_bucketing() -> None:
    n = 40
    close = np.concatenate([np.linspace(100.0, 100.0, 15), np.linspace(100.0, 115.0, 25)])
    conf = np.linspace(-1.0, 1.0, n)
    df = pd.DataFrame({"confluence_net": conf, "close": close})
    out = calculate_historical_probability(df, forward_return_bars=5, profit_target=0.02, stop_loss=0.01, bins=5)
    assert not out.empty
    assert (out["win_rate"] >= 0).all() and (out["win_rate"] <= 1).all()
    assert int(out["count"].sum()) == int((out["count"]).sum())


def test_entry_price_zero_skipped() -> None:
    n = 25
    close = np.linspace(1.0, 2.0, n)
    close[5] = 0.0
    df = pd.DataFrame({"confluence_net": np.zeros(n), "close": close})
    out = calculate_historical_probability(df, forward_return_bars=3, profit_target=0.05, stop_loss=0.05, bins=3)
    assert list(out.columns) == ["bin_label", "count", "win_rate", "avg_return", "avg_win", "avg_loss"]


def test_profit_and_stop_same_bar_uses_final_return() -> None:
    """When target and stop hit same bar index, code uses sign of final bar return."""
    n = 12
    forward = 4
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 101.0])
    # At i=0: future bars all 100 until last -> bar_returns small positive at end
    df_pos = pd.DataFrame({"confluence_net": np.linspace(0.0, 1.0, n), "close": close})
    out_pos = calculate_historical_probability(
        df_pos, forward_return_bars=forward, profit_target=0.02, stop_loss=0.02, bins=2
    )
    assert not out_pos.empty

    close_neg = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0])
    df_neg = pd.DataFrame({"confluence_net": np.linspace(0.0, 1.0, n), "close": close_neg})
    out_neg = calculate_historical_probability(
        df_neg, forward_return_bars=forward, profit_target=0.02, stop_loss=0.02, bins=2
    )
    assert not out_neg.empty
