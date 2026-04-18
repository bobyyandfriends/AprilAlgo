"""Dashboard page — run backtest, display metrics, equity curve, and trade log."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aprilalgo.backtest import run_backtest
from aprilalgo.data import load_price_data
from aprilalgo.indicators.descriptor import get_catalog
from aprilalgo.strategies import STRATEGIES
from aprilalgo.ui.helpers import METRIC_DISPLAY_NAMES, discover_symbols, format_metric


def render() -> None:
    st.header("Backtest Dashboard")

    available = discover_symbols()
    if not available:
        st.warning("No data found.")
        return

    with st.sidebar:
        st.subheader("Backtest Settings")
        timeframes = list(available.keys())
        tf = st.selectbox(
            "Timeframe",
            timeframes,
            index=timeframes.index("daily") if "daily" in timeframes else 0,
            key="dash_tf",
        )
        symbols = available.get(tf, [])
        symbol = st.selectbox(
            "Symbol",
            symbols,
            index=symbols.index("AAPL") if "AAPL" in symbols else 0,
            key="dash_sym",
        )

        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), key="dash_strat")
        capital = st.number_input("Initial Capital ($)", value=100_000, step=10_000, key="dash_cap")

        st.subheader("Strategy Parameters")
        strat_params = _strategy_params(strategy_name)

        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure settings in the sidebar and click **Run Backtest** to begin.")
        return

    try:
        df = load_price_data(symbol, tf)
    except FileNotFoundError:
        st.error(f"Data not found for {symbol} {tf}")
        return

    with st.spinner(f"Running {strategy_name} on {symbol} ({tf})..."):
        strat_cls = STRATEGIES[strategy_name]
        strategy = strat_cls(**strat_params)
        result = run_backtest(strategy, df, initial_capital=capital)

    metrics = result["metrics"]
    trades_df = result["trades"]
    equity_df = result["equity"]

    # --- metric cards ---
    st.subheader("Performance Summary")
    cols = st.columns(5)
    for i, key in enumerate(["total_pnl", "total_return_pct", "num_trades", "win_rate_pct", "sharpe_ratio"]):
        with cols[i]:
            st.metric(METRIC_DISPLAY_NAMES.get(key, key), format_metric(key, metrics.get(key, 0)))
    cols2 = st.columns(5)
    for i, key in enumerate(["profit_factor", "max_drawdown_pct", "sortino_ratio", "avg_win", "avg_loss"]):
        with cols2[i]:
            st.metric(METRIC_DISPLAY_NAMES.get(key, key), format_metric(key, metrics.get(key, 0)))

    # --- equity curve ---
    st.subheader("Equity Curve")
    if not equity_df.empty and "time" in equity_df.columns:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=equity_df["time"],
                y=equity_df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#42a5f5", width=2),
                fill="tozeroy",
                fillcolor="rgba(66,165,245,0.1)",
            )
        )
        fig_eq.add_hline(y=capital, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_eq.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=60, r=30, t=20, b=30),
            yaxis_title="Equity ($)",
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No equity data to display.")

    # --- trades on chart ---
    st.subheader("Trades on Chart")
    if not trades_df.empty:
        fig_t = go.Figure()
        fig_t.add_trace(
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            )
        )
        entries = trades_df[trades_df["entry_time"].notna()]
        exits = trades_df[trades_df["exit_time"].notna()]
        fig_t.add_trace(
            go.Scatter(
                x=entries["entry_time"],
                y=entries["entry_price"],
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#26a69a",
                    line=dict(width=1, color="white"),
                ),
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=exits["exit_time"],
                y=exits["exit_price"],
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ef5350",
                    line=dict(width=1, color="white"),
                ),
            )
        )
        fig_t.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=60, r=30, t=20, b=30),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # --- trade log ---
    st.subheader("Trade Log")
    if not trades_df.empty:
        display = trades_df.copy()
        for col in ["entry_price", "exit_price", "realized_pnl"]:
            if col in display.columns:
                display[col] = display[col].round(2)

        def _color_pnl(val):
            if pd.isna(val):
                return ""
            return "color: #26a69a" if val > 0 else "color: #ef5350" if val < 0 else ""

        styled = display.style.map(_color_pnl, subset=["realized_pnl"])
        st.dataframe(styled, use_container_width=True, height=300)
    else:
        st.info("No trades generated by this strategy on the selected data.")


def _strategy_params(strategy_name: str) -> dict:
    """Render strategy-specific parameter controls and return values."""
    params: dict = {}

    if strategy_name == "rsi_sma":
        params["rsi_period"] = st.slider("RSI Period", 5, 30, 14, key="d_rsi_p")
        params["sma_period"] = st.slider("SMA Period", 5, 200, 50, key="d_sma_p")
        params["rsi_buy"] = st.slider("RSI Buy Threshold", 10, 50, 30, key="d_rsi_b")
        params["rsi_sell"] = st.slider("RSI Sell Threshold", 50, 90, 70, key="d_rsi_s")

    elif strategy_name == "demark_confluence":
        params["rsi_period"] = st.slider("RSI Period", 5, 30, 14, key="d_dm_rsi")
        params["sma_period"] = st.slider("SMA Period", 5, 200, 50, key="d_dm_sma")
        params["bb_period"] = st.slider("BB Period", 5, 50, 20, key="d_dm_bb")
        params["ss_period"] = st.slider("SS Period", 5, 50, 10, key="d_dm_ss")
        params["confluence_threshold"] = st.slider("Confluence Threshold", 0.0, 1.0, 0.3, 0.05, key="d_dm_ct")
        params["stop_loss_pct"] = st.slider("Stop Loss %", 0.01, 0.10, 0.03, 0.005, key="d_dm_sl")

    elif strategy_name == "configurable":
        catalog = get_catalog()
        selected_inds = st.multiselect(
            "Indicators",
            list(catalog.keys()),
            format_func=lambda k: catalog[k].display_name,
            default=["rsi", "sma", "demark"],
            key="d_cfg_inds",
        )
        indicators = []
        for ind_name in selected_inds:
            spec = catalog[ind_name]
            cfg: dict = {"name": ind_name}
            for p in spec.params:
                key = f"d_cfg_{ind_name}_{p.name}"
                if isinstance(p.default, float):
                    cfg[p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        float(p.min_val),
                        float(p.max_val),
                        float(p.default),
                        float(p.step),
                        key=key,
                    )
                else:
                    cfg[p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        int(p.min_val),
                        int(p.max_val),
                        int(p.default),
                        int(p.step),
                        key=key,
                    )
            indicators.append(cfg)

        params["indicators"] = indicators
        params["entry_threshold"] = st.slider("Entry Threshold", 0.0, 1.0, 0.3, 0.05, key="d_cfg_et")
        params["exit_threshold"] = st.slider("Exit Threshold", -1.0, 0.0, -0.1, 0.05, key="d_cfg_xt")
        params["stop_loss_pct"] = st.slider("Stop Loss %", 0.01, 0.10, 0.03, 0.005, key="d_cfg_sl")

    return params
