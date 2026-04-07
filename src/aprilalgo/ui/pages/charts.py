"""Charts page — interactive candlestick chart with indicator overlays and trade markers.

Indicator options, parameter sliders, and function calls are all driven
by the descriptor catalog — adding a new indicator requires zero changes here.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aprilalgo.data import load_price_data
from aprilalgo.indicators.descriptor import get_catalog, IndicatorSpec
from aprilalgo.confluence.scorer import score_confluence
from aprilalgo.ui.helpers import discover_symbols


def render() -> None:
    st.header("Price Charts & Indicators")

    available = discover_symbols()
    if not available:
        st.warning("No data found in the `data/` directory. Fetch some data first.")
        return

    catalog = get_catalog()

    # --- sidebar controls ---
    with st.sidebar:
        st.subheader("Chart Settings")
        timeframes = list(available.keys())
        tf = st.selectbox("Timeframe", timeframes,
                          index=timeframes.index("daily") if "daily" in timeframes else 0)
        symbols = available.get(tf, [])
        symbol = st.selectbox("Symbol", symbols,
                              index=symbols.index("AAPL") if "AAPL" in symbols else 0)

        st.subheader("Indicators")
        selected = st.multiselect(
            "Overlay",
            list(catalog.keys()),
            format_func=lambda k: catalog[k].display_name,
            default=["sma", "rsi"],
        )

        st.subheader("Parameters")
        ind_params: dict[str, dict] = {}
        for ind_name in selected:
            spec = catalog[ind_name]
            ind_params[ind_name] = {}
            for p in spec.params:
                key = f"chart_{ind_name}_{p.name}"
                if isinstance(p.default, float):
                    ind_params[ind_name][p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        float(p.min_val), float(p.max_val), float(p.default), float(p.step),
                        key=key,
                    )
                else:
                    ind_params[ind_name][p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        int(p.min_val), int(p.max_val), int(p.default), int(p.step),
                        key=key,
                    )

        show_confluence = st.checkbox("Show Confluence Score", value=True)

    # --- load and enrich data ---
    try:
        df = load_price_data(symbol, tf)
    except FileNotFoundError:
        st.error(f"Could not load {symbol} {tf} data.")
        return

    for ind_name in selected:
        spec = catalog[ind_name]
        df = spec(df, **ind_params.get(ind_name, {}))

    if show_confluence:
        df = score_confluence(df)

    _draw_chart(df, symbol, tf, selected, ind_params, show_confluence, catalog)


def _draw_chart(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    selected: list[str],
    ind_params: dict[str, dict],
    show_confluence: bool,
    catalog: dict[str, IndicatorSpec],
) -> None:
    """Build the Plotly figure with candlestick + overlays + sub-panels."""

    overlays = [s for s in selected if catalog[s].overlay]
    panels = [s for s in selected if not catalog[s].overlay]

    sub_names: list[str] = [catalog[s].display_name for s in panels]
    sub_names.append("Volume")
    if show_confluence and "confluence_net" in df.columns:
        sub_names.append("Confluence")

    n_rows = 1 + len(sub_names)
    row_heights = [0.5] + [0.5 / max(len(sub_names), 1)] * len(sub_names)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[f"{symbol} ({tf.upper()})"] + sub_names,
    )

    # --- row 1: candlestick ---
    fig.add_trace(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # --- overlay rendering ---
    for ind_name in overlays:
        params = ind_params.get(ind_name, {})
        _render_overlay(fig, df, ind_name, params)

    # --- sub-panel rendering ---
    current_row = 2
    for ind_name in panels:
        params = ind_params.get(ind_name, {})
        _render_panel(fig, df, ind_name, params, current_row, catalog)
        current_row += 1

    # Volume (always shown)
    if "volume" in df.columns:
        vol_colors = [
            "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["volume"], name="Volume",
            marker_color=vol_colors, opacity=0.6,
        ), row=current_row, col=1)
        current_row += 1

    # Confluence
    if show_confluence and "confluence_net" in df.columns:
        conf_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["confluence_net"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df["datetime"], y=df["confluence_net"], name="Confluence Net",
            marker_color=conf_colors,
        ), row=current_row, col=1)
        fig.add_hline(y=0, line_color="white", line_width=0.5, row=current_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=200 + 250 * n_rows,
        margin=dict(l=60, r=30, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Data", expanded=False):
        display_cols = ["datetime", "open", "high", "low", "close", "volume"]
        extra = [c for c in df.columns if c not in display_cols]
        st.dataframe(df[display_cols + extra].tail(100), use_container_width=True, height=400)


# ---------------------------------------------------------------------------
# Overlay renderers (drawn on the candlestick chart, row=1)
# ---------------------------------------------------------------------------

_OVERLAY_COLORS = {
    "sma": "#ffa726", "super_smoother": "#ab47bc",
    "decycler": "#66bb6a", "bollinger_bands": "#90caf9",
}


def _render_overlay(fig: go.Figure, df: pd.DataFrame, name: str, params: dict) -> None:
    if name == "sma":
        col = f"sma_{params.get('period', 20)}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col], name=col.upper(),
                line=dict(color=_OVERLAY_COLORS["sma"], width=1.5),
            ), row=1, col=1)

    elif name == "bollinger_bands":
        p = params.get("period", 20)
        upper, lower, mid = f"bb_{p}_upper", f"bb_{p}_lower", f"bb_{p}_mid"
        if upper in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[upper], name="BB Upper",
                line=dict(color="#90caf9", width=1, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[lower], name="BB Lower",
                line=dict(color="#90caf9", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(144,202,249,0.08)",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[mid], name="BB Mid",
                line=dict(color="#64b5f6", width=1),
            ), row=1, col=1)

    elif name == "super_smoother":
        col = f"ss_{params.get('period', 10)}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col], name="Super Smoother",
                line=dict(color=_OVERLAY_COLORS["super_smoother"], width=1.5),
            ), row=1, col=1)

    elif name == "decycler":
        col = f"decycler_{params.get('period', 125)}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col], name="Decycler",
                line=dict(color=_OVERLAY_COLORS["decycler"], width=1.5),
            ), row=1, col=1)

    elif name == "demark":
        if "td_bull" in df.columns:
            buys = df[df["td_bull"] == True]
            sells = df[df["td_bear"] == True]
            if len(buys):
                fig.add_trace(go.Scatter(
                    x=buys["datetime"], y=buys["low"] * 0.998,
                    mode="markers", name="TD Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                ), row=1, col=1)
            if len(sells):
                fig.add_trace(go.Scatter(
                    x=sells["datetime"], y=sells["high"] * 1.002,
                    mode="markers", name="TD Sell",
                    marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                ), row=1, col=1)


# ---------------------------------------------------------------------------
# Panel renderers (drawn in sub-panels below the candlestick)
# ---------------------------------------------------------------------------

_PANEL_COLORS = {
    "rsi": "#7e57c2", "tmi": None, "hurst": "#ff7043",
    "roofing_filter": "#42a5f5", "volume_trend": None, "pv_sequences": None,
}


def _render_panel(
    fig: go.Figure, df: pd.DataFrame, name: str, params: dict,
    row: int, catalog: dict,
) -> None:
    if name == "rsi":
        col = f"rsi_{params.get('period', 14)}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col], name="RSI",
                line=dict(color=_PANEL_COLORS["rsi"], width=1.5),
            ), row=row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.4)", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.4)", row=row, col=1)
            fig.update_yaxes(range=[0, 100], row=row, col=1)

    elif name == "tmi":
        col = f"tmi_{params.get('period', 14)}"
        if col in df.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[col].fillna(0)]
            fig.add_trace(go.Bar(
                x=df["datetime"], y=df[col], name="TMI", marker_color=colors,
            ), row=row, col=1)
            fig.add_hline(y=0, line_color="white", line_width=0.5, row=row, col=1)

    elif name == "hurst":
        w = params.get("window", 100)
        col = f"hurst_{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["datetime"], y=df[col], name=f"Hurst({w})",
                line=dict(color=_PANEL_COLORS["hurst"], width=1.5),
            ), row=row, col=1)
            fig.add_hline(y=0.55, line_dash="dash", line_color="rgba(38,166,154,0.4)", row=row, col=1)
            fig.add_hline(y=0.45, line_dash="dash", line_color="rgba(239,83,80,0.4)", row=row, col=1)
            fig.update_yaxes(range=[0, 1], row=row, col=1)

    elif name == "roofing_filter":
        hp = params.get("hp_period", 48)
        lp = params.get("lp_period", 10)
        col = f"roof_{hp}_{lp}"
        if col in df.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[col].fillna(0)]
            fig.add_trace(go.Bar(
                x=df["datetime"], y=df[col], name="Roofing", marker_color=colors,
            ), row=row, col=1)
            fig.add_hline(y=0, line_color="white", line_width=0.5, row=row, col=1)

    elif name == "volume_trend":
        vp = params.get("vol_period", 20)
        ratio_col = f"vol_{vp}_ratio"
        if ratio_col in df.columns:
            colors = ["#26a69a" if v >= 1.5 else "#78909c" for v in df[ratio_col].fillna(0)]
            fig.add_trace(go.Bar(
                x=df["datetime"], y=df[ratio_col], name="Vol Ratio", marker_color=colors,
            ), row=row, col=1)
            fig.add_hline(y=1.5, line_dash="dash", line_color="rgba(255,167,38,0.5)", row=row, col=1)

    elif name == "pv_sequences":
        if "pv_state" in df.columns:
            state_colors = {0: "#26a69a", 1: "#a5d6a7", 2: "#ef5350", 3: "#ef9a9a"}
            colors = [state_colors.get(s, "#78909c") for s in df["pv_state"].fillna(3)]
            fig.add_trace(go.Bar(
                x=df["datetime"], y=df["pv_streak"], name="PV Streak", marker_color=colors,
            ), row=row, col=1)
