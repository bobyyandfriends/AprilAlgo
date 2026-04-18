"""Tests for ML feature matrix construction (no OHLCV in X)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.indicators import IndicatorRegistry, rsi
from aprilalgo.labels.triple_barrier import apply_triple_barrier
from aprilalgo.meta.regime import add_vol_regime
from aprilalgo.ml.features import (
    DEFAULT_EXCLUDED_FROM_FEATURES,
    align_features_and_labels,
    build_feature_matrix,
    extract_feature_matrix,
    feature_column_names,
)


@pytest.fixture
def price_data():
    from aprilalgo.data import load_price_data

    return load_price_data("AAPL", "daily")


class TestFeatureColumnNames:
    def test_excludes_ohlcv_and_time(self):
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
                "timestamp": ["2020-01-01"],
                "rsi_14": [50.0],
            }
        )
        names = feature_column_names(df)
        assert names == ["rsi_14"]
        assert set(DEFAULT_EXCLUDED_FROM_FEATURES) >= {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "realized_vol",
        }

    def test_extra_exclude(self):
        df = pd.DataFrame({"close": [1.0], "rsi_14": [50.0], "extra": [1.0]})
        assert feature_column_names(
            df, extra_exclude=frozenset({"extra"})
        ) == ["rsi_14"]


class TestExtractFeatureMatrix:
    def test_strips_ohlcv_from_enriched(self, price_data):
        enriched = rsi(price_data, period=14)
        X = extract_feature_matrix(enriched)
        for c in ("open", "high", "low", "close", "volume"):
            assert c not in X.columns
        assert "rsi_14" in X.columns
        assert len(X) == len(enriched)

    def test_bool_to_float(self, price_data):
        enriched = rsi(price_data, period=14)
        X = extract_feature_matrix(enriched, convert_bool=True)
        assert str(X["rsi_14_bull"].dtype) == "float64"

    def test_empty_features_returns_empty_frame_with_index(self):
        df = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=[0])
        X = extract_feature_matrix(df)
        assert X.empty
        assert list(X.index) == [0]


class TestBuildFeatureMatrix:
    def test_from_indicator_config(self, price_data):
        X = build_feature_matrix(
            price_data,
            indicator_config=[{"name": "rsi", "period": 14}],
        )
        for c in ("open", "high", "low", "close", "volume"):
            assert c not in X.columns
        assert "rsi_14" in X.columns

    def test_from_registry(self, price_data):
        reg = IndicatorRegistry().add_by_name("rsi", period=14)
        X = build_feature_matrix(price_data, registry=reg)
        assert "rsi_14" in X.columns
        assert "close" not in X.columns

    def test_both_config_and_registry_rejected(self, price_data):
        reg = IndicatorRegistry().add_by_name("rsi", period=14)
        with pytest.raises(ValueError, match="at most one"):
            build_feature_matrix(
                price_data,
                indicator_config=[{"name": "rsi", "period": 14}],
                registry=reg,
            )

    def test_enriched_only_path(self, price_data):
        enriched = rsi(price_data, period=14)
        X = build_feature_matrix(enriched)
        assert "rsi_14" in X.columns
        assert "close" not in X.columns

    def test_regime_inclusion_rules(self, price_data):
        df = add_vol_regime(price_data, window=10, n_buckets=3)
        X = build_feature_matrix(
            df,
            indicator_config=[
                {"name": "rsi", "period": 14},
                {"name": "sma", "period": 20},
            ],
        )
        assert "vol_regime" in X.columns
        assert "realized_vol" not in X.columns


class TestAlignFeaturesAndLabels:
    def test_drops_nan_labels(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        y = pd.Series([1.0, np.nan, -1.0], index=[0, 1, 2])
        X2, y2 = align_features_and_labels(X, y)
        assert len(X2) == 2
        assert list(y2.values) == [1.0, -1.0]

    def test_with_triple_barrier(self, price_data):
        enriched = rsi(price_data, period=14)
        tb = apply_triple_barrier(
            price_data, upper_pct=0.02, lower_pct=0.02, vertical_bars=5
        )
        X = extract_feature_matrix(enriched)
        Xa, ya = align_features_and_labels(X, tb.label, dropna_features=True)
        assert len(Xa) <= len(X)
        assert len(Xa) == len(ya)
        assert ya.notna().all()
