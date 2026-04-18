"""Signal Feed page — shows per-bar confluence signals with bull/bear breakdown."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from aprilalgo.confluence.scorer import score_confluence
from aprilalgo.data import load_price_data
from aprilalgo.indicators.descriptor import get_catalog
from aprilalgo.ui.helpers import discover_symbols


def render() -> None:
    st.header("Signal Feed")

    available = discover_symbols()
    if not available:
        st.warning("No data found.")
        return

    with st.sidebar:
        st.subheader("Signal Settings")
        timeframes = list(available.keys())
        tf = st.selectbox(
            "Timeframe",
            timeframes,
            index=timeframes.index("daily") if "daily" in timeframes else 0,
            key="sig_tf",
        )
        symbols = available.get(tf, [])
        symbol = st.selectbox(
            "Symbol",
            symbols,
            index=symbols.index("AAPL") if "AAPL" in symbols else 0,
            key="sig_sym",
        )
        n_rows = st.slider("Show last N bars", 10, 200, 50, key="sig_n")
        min_conf = st.slider("Min |confluence_net|", 0.0, 1.0, 0.0, 0.05, key="sig_min")

    try:
        df = load_price_data(symbol, tf)
    except FileNotFoundError:
        st.error(f"Data not found for {symbol} {tf}")
        return

    df = _enrich_all(df)
    df = score_confluence(df)

    tail = df.tail(n_rows).copy()
    if min_conf > 0:
        tail = tail[tail["confluence_net"].abs() >= min_conf]

    if tail.empty:
        st.info("No signals meet the filter criteria.")
        return

    bull_cols = [c for c in tail.columns if c.endswith("_bull")]
    bear_cols = [c for c in tail.columns if c.endswith("_bear")]

    def _safe_int(val) -> int:
        # Indicator warm-up rows propagate NaN into ``bull_count`` / ``bear_count``;
        # ``.get(..., 0)`` only covers missing keys, not NaN values.
        try:
            if val is None or pd.isna(val):
                return 0
            return int(val)
        except (TypeError, ValueError):
            return 0

    for _, row in tail.iloc[::-1].iterrows():
        direction = row.get("confluence_direction", "NEUTRAL")
        net = row.get("confluence_net", 0.0)
        if pd.isna(net):
            continue

        dt_val = row.get("datetime")
        if pd.isna(dt_val):
            continue
        dt_str = pd.Timestamp(dt_val).strftime("%Y-%m-%d %H:%M")

        color = "#26a69a" if direction == "LONG" else "#ef5350" if direction == "SHORT" else "#78909c"
        arrow = "\u25b2" if direction == "LONG" else "\u25bc" if direction == "SHORT" else "\u25cf"

        bull_n = _safe_int(row.get("bull_count", 0))
        bull_t = _safe_int(row.get("bull_total", 0))
        bear_n = _safe_int(row.get("bear_count", 0))
        bear_t = _safe_int(row.get("bear_total", 0))
        close_val = row.get("close", float("nan"))
        close_str = f"${float(close_val):.2f}" if pd.notna(close_val) else "—"

        st.markdown(
            f"""<div style="border-left: 4px solid {color}; padding: 8px 14px; margin-bottom: 8px;
            background: rgba(255,255,255,0.03); border-radius: 0 6px 6px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.1em;"><b>{arrow} {direction}</b>
                &nbsp; {symbol} &nbsp;
                <span style="color:#aaa;">{dt_str}</span></span>
                <span style="font-size:1.3em; font-weight:bold; color:{color};">
                    Net: {net:+.2f}
                </span>
            </div>
            <div style="color:#bbb; font-size:0.85em; margin-top:4px;">
                Close: {close_str} &nbsp;|&nbsp;
                Bull: {bull_n}/{bull_t} &nbsp;|&nbsp;
                Bear: {bear_n}/{bear_t}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Signal Breakdown")
    show_cols = (
        [
            "datetime",
            "close",
            "confluence_net",
            "confluence_direction",
            "bull_count",
            "bull_total",
            "bear_count",
            "bear_total",
        ]
        + bull_cols
        + bear_cols
    )
    show_cols = [c for c in show_cols if c in tail.columns]
    st.dataframe(
        tail[show_cols].iloc[::-1].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


def _enrich_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every registered indicator with default params."""
    catalog = get_catalog()
    for spec in catalog.values():
        df = spec(df)
    return df
