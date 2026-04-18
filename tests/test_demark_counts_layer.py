"""Unit tests for DeMark count annotations on Plotly figures."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aprilalgo.ui.pages.charts.layers import demark_counts


def test_render_demark_counts_no_columns_noop() -> None:
    fig = go.Figure()
    df = pd.DataFrame({"datetime": [1, 2], "low": [1.0, 1.0], "high": [1.1, 1.1]})
    n_ann = len(fig.layout.annotations) if fig.layout.annotations else 0
    demark_counts.render_demark_counts(fig, df, min_count=1, show_countdown=True)
    assert len(fig.layout.annotations or []) == n_ann


def test_render_demark_counts_adds_annotations() -> None:
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Candlestick(x=[1, 2], open=[1, 1], high=[1.1, 1.1], low=[0.9, 0.9], close=[1, 1]), row=1, col=1)
    df = pd.DataFrame(
        {
            "datetime": [1, 2],
            "low": [1.0, 1.0],
            "high": [1.1, 1.1],
            "td_buy_setup": [5, 0],
            "td_sell_setup": [0, 6],
            "td_buy_countdown": [4, 0],
            "td_sell_countdown": [0, 5],
        }
    )
    demark_counts.render_demark_counts(fig, df, min_count=4, show_countdown=True, price_row=1)
    assert len(fig.layout.annotations or []) >= 2


def test_render_demark_counts_show_countdown_false_skips_cd() -> None:
    fig = go.Figure()
    df = pd.DataFrame(
        {
            "datetime": [1],
            "low": [1.0],
            "high": [1.2],
            "td_buy_setup": [9],
            "td_buy_countdown": [13],
        }
    )
    demark_counts.render_demark_counts(fig, df, min_count=1, show_countdown=False, price_row=1)
    ann_texts = [a.text for a in (fig.layout.annotations or [])]
    assert "9" in ann_texts
    assert not any(t == "13" for t in ann_texts)


def test_render_demark_counts_yref_second_row() -> None:
    fig = make_subplots(rows=2, cols=1)
    df = pd.DataFrame(
        {
            "datetime": [1],
            "low": [10.0],
            "high": [11.0],
            "td_sell_setup": [7],
        }
    )
    demark_counts.render_demark_counts(fig, df, min_count=1, show_countdown=False, price_row=2)
    assert fig.layout.annotations
    assert fig.layout.annotations[0].yref == "y2"
