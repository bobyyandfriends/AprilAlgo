"""Unit tests for chart layer helpers (no Streamlit session)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aprilalgo.ui.pages.charts.layers import demark_counts, ml_proba, overlays, panels, shap_local


def test_render_overlay_sma_adds_trace() -> None:
    fig = make_subplots(rows=1, cols=1)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="D"),
            "close": [100.0, 101, 102, 103, 104],
            "sma_20": [99.0, 100, 101, 102, 103],
        }
    )
    n0 = len(fig.data)
    overlays.render_overlay(fig, df, "sma", {"period": 20}, price_row=1)
    assert len(fig.data) == n0 + 1
    assert isinstance(fig, go.Figure)


def test_render_panel_rsi() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="D"),
            "rsi_14": [30.0, 40, 50, 60, 70],
        }
    )
    n0 = len(fig.data)
    panels.render_panel(fig, df, "rsi", {"period": 14}, row=2)
    assert len(fig.data) > n0


def test_render_ml_proba_panel_area_mode() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=5, freq="D"), "close": range(5)})
    proba = pd.DataFrame({"proba_1": [0.2, 0.4, 0.6, 0.55, 0.3]})
    ml_proba.render_ml_proba_panel(fig, df, proba, row=2, threshold=0.55, mode="area")
    assert any(getattr(t, "mode", None) == "lines" for t in fig.data)


def test_render_shap_stack_panel_empty() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=3, freq="D"), "close": [1, 2, 3]})
    out = shap_local.render_shap_stack_panel(fig, df, pd.DataFrame(), row=2, top_k=3)
    assert out == []


def test_render_shap_stack_panel_nonempty() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=3, freq="D"), "close": [1, 2, 3]})
    sm = pd.DataFrame({"a": [0.1, -0.2, 0.05], "b": [-0.05, 0.1, 0.02]})
    names = shap_local.render_shap_stack_panel(fig, df, sm, row=2, top_k=2)
    assert isinstance(names, list)
    assert len(fig.data) >= 1


def test_shade_price_band_by_proba_calls_vrect() -> None:
    fig = make_subplots(rows=1, cols=1)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=6, freq="D"),
            "close": [100.0, 101, 102, 103, 104, 105],
        }
    )
    proba = pd.DataFrame({"proba_1": [0.1, 0.7, 0.8, 0.2, 0.1, 0.6]})
    calls = {"n": 0}
    orig = fig.add_vrect

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    fig.add_vrect = _spy  # type: ignore[method-assign]
    ml_proba.shade_price_band_by_proba(fig, df, proba, threshold=0.55, max_spans=10)
    assert calls["n"] >= 1


def test_render_demark_counts_no_td_columns_no_crash() -> None:
    fig = go.Figure()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="D"),
            "close": [100.0] * 5,
        }
    )
    demark_counts.render_demark_counts(fig, df, min_count=1, price_row=1)
    assert fig.layout.annotations is None or len(fig.layout.annotations) == 0


def test_render_ml_proba_panel_stacked_multiclass() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=4, freq="D"), "close": range(4)})
    proba = pd.DataFrame(
        {
            "proba_-1.0": [0.1, 0.2, 0.15, 0.1],
            "proba_0.0": [0.3, 0.3, 0.35, 0.4],
            "proba_1.0": [0.6, 0.5, 0.5, 0.5],
        }
    )
    n0 = len(fig.data)
    ml_proba.render_ml_proba_panel(fig, df, proba, row=2, mode="stacked")
    assert len(fig.data) > n0
