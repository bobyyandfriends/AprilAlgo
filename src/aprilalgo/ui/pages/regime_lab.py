"""Streamlit: volatility regime overlay (v0.4)."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from aprilalgo.data import load_price_data
from aprilalgo.meta.regime import add_vol_regime

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def render() -> None:
    st.title("Regime lab")
    sym = st.text_input("Symbol", value="TEST")
    tf = st.text_input("Timeframe", value="daily")
    use_fixture = st.checkbox("Load from tests/fixtures (TEST ticker)", value=True)
    window = st.slider("Realized vol window", 5, 60, 20)
    buckets = st.slider("Quantile buckets", 2, 5, 3)

    data_dir = _PROJECT_ROOT / "tests" / "fixtures" if use_fixture else None
    try:
        df = load_price_data(sym, tf, data_dir=data_dir)
    except Exception as e:
        st.error(f"Load failed: {e}")
        return

    enriched = add_vol_regime(df, window=window, n_buckets=buckets)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], name="close", yaxis="y"))
    fig.add_trace(
        go.Scatter(
            x=enriched["datetime"],
            y=enriched["vol_regime"],
            name="regime",
            yaxis="y2",
            mode="markers",
        )
    )
    fig.update_layout(
        yaxis=dict(title="price"),
        yaxis2=dict(title="regime", overlaying="y", side="right"),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(enriched.tail(20), use_container_width=True)
