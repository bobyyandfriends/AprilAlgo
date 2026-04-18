"""Tests for the confluence scoring engine."""

import pytest
import pandas as pd
from aprilalgo.data import load_price_data
from aprilalgo.indicators import rsi, sma, bollinger_bands, volume_trend, demark, super_smoother
from aprilalgo.confluence import score_confluence


@pytest.fixture
def price_data():
    return load_price_data("AAPL", "daily")


@pytest.fixture
def enriched_data(price_data):
    df = rsi(price_data, period=14)
    df = sma(df, period=20)
    df = bollinger_bands(df, period=20)
    df = volume_trend(df)
    df = demark(df)
    df = super_smoother(df, period=10)
    return df


class TestScoreConfluence:

    def test_adds_required_columns(self, enriched_data):
        scored = score_confluence(enriched_data)
        for col in ["confluence_net", "confluence_direction", "bull_count",
                     "bear_count", "bull_total", "bear_total"]:
            assert col in scored.columns, f"Missing column: {col}"

    def test_confluence_net_range(self, enriched_data):
        scored = score_confluence(enriched_data)
        valid = scored["confluence_net"].dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_direction_values(self, enriched_data):
        scored = score_confluence(enriched_data)
        valid_directions = {"LONG", "SHORT", "NEUTRAL"}
        actual = set(scored["confluence_direction"].unique())
        assert actual.issubset(valid_directions)

    def test_auto_detects_bull_bear(self, enriched_data):
        scored = score_confluence(enriched_data)
        bull_cols = [c for c in enriched_data.columns if c.endswith("_bull")]
        assert scored["bull_total"].max() == len(bull_cols)

    def test_empty_signals_returns_neutral(self):
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 900]})
        scored = score_confluence(df)
        assert (scored["confluence_net"] == 0.0).all()
        assert (scored["confluence_direction"] == "NEUTRAL").all()
