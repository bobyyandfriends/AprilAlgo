"""Unit tests for charts page helpers (no Streamlit session)."""

from __future__ import annotations

import pandas as pd

from aprilalgo.indicators.descriptor import get_catalog
from aprilalgo.ui.pages.charts import page as chart_page


def test_apply_indicators_skips_ml_pseudo_except_demark() -> None:
    catalog = get_catalog()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
        }
    )
    selected = ["ml_proba", "shap_local", "sma"]
    params = {"sma": {"period": 3}}
    out = chart_page._apply_indicators(df, selected, params, catalog)
    assert "sma_3" in out.columns


def test_apply_indicators_demark_counts_runs() -> None:
    catalog = get_catalog()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=30, freq="D"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
        }
    )
    out = chart_page._apply_indicators(
        df,
        ["demark_counts"],
        {"demark_counts": {"min_count": 4}},
        catalog,
    )
    assert len(out) == len(df)
