"""Tests for all indicator functions — column naming, no collision, basic correctness."""

from __future__ import annotations

import pytest
import pandas as pd

from aprilalgo.data import load_price_data
from aprilalgo.indicators import (
    rsi,
    sma,
    bollinger_bands,
    volume_trend,
    demark,
    hurst,
    super_smoother,
    roofing_filter,
    decycler,
    tmi,
    pv_sequences,
    IndicatorRegistry,
    get_catalog,
)


@pytest.fixture
def price_data():
    return load_price_data("AAPL", "daily")


class TestParameterizedColumnNames:
    """Verify that column names include parameters to prevent collision."""

    def test_rsi_columns(self, price_data):
        r = rsi(price_data, period=14)
        assert "rsi_14" in r.columns
        assert "rsi_14_bull" in r.columns
        assert "rsi_14_bear" in r.columns

    def test_rsi_no_collision(self, price_data):
        r = rsi(price_data, period=14)
        r = rsi(r, period=7)
        assert "rsi_14" in r.columns
        assert "rsi_7" in r.columns
        assert "rsi_14_bull" in r.columns
        assert "rsi_7_bull" in r.columns

    def test_sma_columns(self, price_data):
        r = sma(price_data, period=20)
        assert "sma_20" in r.columns
        assert "sma_20_bull" in r.columns
        assert "sma_20_bear" in r.columns

    def test_sma_no_collision(self, price_data):
        r = sma(price_data, period=20)
        r = sma(r, period=50)
        assert "sma_20" in r.columns
        assert "sma_50" in r.columns
        assert "sma_20_bull" in r.columns
        assert "sma_50_bull" in r.columns

    def test_bollinger_columns(self, price_data):
        r = bollinger_bands(price_data, period=20)
        assert "bb_20_mid" in r.columns
        assert "bb_20_upper" in r.columns
        assert "bb_20_lower" in r.columns
        assert "bb_20_pct" in r.columns
        assert "bb_20_bull" in r.columns
        assert "bb_20_bear" in r.columns

    def test_volume_trend_columns(self, price_data):
        r = volume_trend(price_data, vol_period=20)
        assert "vol_20_sma" in r.columns
        assert "vol_20_bull" in r.columns
        assert "vol_20_bear" in r.columns

    def test_demark_columns(self, price_data):
        r = demark(price_data)
        assert "td_bull" in r.columns
        assert "td_bear" in r.columns
        assert "td_buy_setup" in r.columns

    def test_super_smoother_columns(self, price_data):
        r = super_smoother(price_data, period=10)
        assert "ss_10" in r.columns
        assert "ss_10_bull" in r.columns
        assert "ss_10_bear" in r.columns

    def test_super_smoother_no_collision(self, price_data):
        r = super_smoother(price_data, period=10)
        r = super_smoother(r, period=20)
        assert "ss_10_bull" in r.columns
        assert "ss_20_bull" in r.columns

    def test_roofing_filter_columns(self, price_data):
        r = roofing_filter(price_data, hp_period=48, lp_period=10)
        assert "roof_48_10" in r.columns
        assert "roof_48_10_bull" in r.columns
        assert "roof_48_10_bear" in r.columns

    def test_decycler_columns(self, price_data):
        r = decycler(price_data, period=125)
        assert "decycler_125" in r.columns
        assert "decycler_125_bull" in r.columns
        assert "decycler_125_bear" in r.columns

    def test_tmi_columns(self, price_data):
        r = tmi(price_data, period=14, smooth=5)
        assert "tmi_14" in r.columns
        assert "tmi_14_bull" in r.columns
        assert "tmi_14_bear" in r.columns

    def test_pv_sequences_columns(self, price_data):
        r = pv_sequences(price_data, streak_threshold=3)
        assert "pv_bull" in r.columns
        assert "pv_bear" in r.columns
        assert "pv_state" in r.columns

    def test_hurst_columns(self, price_data):
        r = hurst(price_data, windows=[100])
        assert "hurst_100" in r.columns
        assert "hurst_bull" in r.columns
        assert "hurst_bear" in r.columns


class TestDescriptorCatalog:
    def test_catalog_has_all_indicators(self):
        catalog = get_catalog()
        names = set(catalog.keys())
        # Keep in sync with aprilalgo.indicators.descriptor._build_catalog
        assert names == {
            "rsi",
            "sma",
            "bollinger_bands",
            "volume_trend",
            "demark",
            "hurst",
            "super_smoother",
            "roofing_filter",
            "decycler",
            "tmi",
            "pv_sequences",
            "demark_counts",
            "ml_proba",
            "shap_local",
        }

    def test_each_spec_callable(self, price_data):
        catalog = get_catalog()
        for spec in catalog.values():
            try:
                out = spec(price_data)
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
                if spec.name in {"ml_proba", "shap_local"}:
                    pytest.skip(f"{spec.name} requires trained model or artifacts: {e}")
                raise
            assert isinstance(out, pd.DataFrame)
            assert len(out) == len(price_data)
            for c in price_data.columns:
                assert c in out.columns, f"{spec.name}: missing input column {c}"
            if "_identity_passthrough" in spec.output_columns:
                assert list(out.columns) == list(price_data.columns)
            else:
                new_cols = set(out.columns) - set(price_data.columns)
                assert len(new_cols) >= 1, f"{spec.name}: expected new indicator columns"
            for oc in spec.output_columns:
                if oc == "_identity_passthrough":
                    continue
                assert oc in out.columns, f"{spec.name}: documented output {oc!r} missing"

    def test_catalog_output_columns_documented(self):
        catalog = get_catalog()
        for spec in catalog.values():
            assert isinstance(spec.output_columns, tuple)
            assert len(spec.output_columns) >= 1
            assert all(isinstance(s, str) and s for s in spec.output_columns)

    def test_each_spec_has_params(self):
        catalog = get_catalog()
        for spec in catalog.values():
            assert len(spec.params) >= 1


class TestRegistryPipeline:
    def test_add_raw(self, price_data):
        reg = IndicatorRegistry()
        reg.add(rsi, period=14)
        r = reg.apply(price_data)
        assert "rsi_14" in r.columns

    def test_add_by_name(self, price_data):
        reg = IndicatorRegistry()
        reg.add_by_name("rsi", period=14)
        r = reg.apply(price_data)
        assert "rsi_14" in r.columns

    def test_from_config(self, price_data):
        reg = IndicatorRegistry.from_config([{"name": "rsi", "period": 14}])
        r = reg.apply(price_data)
        assert "rsi_14" in r.columns
