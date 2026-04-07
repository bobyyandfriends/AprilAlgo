"""Parameter Tuner page — run optimization sweeps and view results."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from aprilalgo.data import load_price_data
from aprilalgo.strategies import STRATEGIES
from aprilalgo.tuner import ParameterGrid, TunerRunner, analyze_results
from aprilalgo.indicators.descriptor import get_catalog
from aprilalgo.ui.helpers import discover_symbols, format_metric, METRIC_DISPLAY_NAMES


def render() -> None:
    st.header("Parameter Tuner")

    available = discover_symbols()
    if not available:
        st.warning("No data found.")
        return

    with st.sidebar:
        st.subheader("Tuner Settings")
        timeframes = list(available.keys())
        tf = st.selectbox("Timeframe", timeframes,
                          index=timeframes.index("daily") if "daily" in timeframes else 0,
                          key="tune_tf")
        symbols = available.get(tf, [])
        symbol = st.selectbox("Symbol", symbols,
                              index=symbols.index("AAPL") if "AAPL" in symbols else 0,
                              key="tune_sym")

        strategy_name = st.selectbox("Strategy",
                                     [k for k in STRATEGIES if k != "configurable"],
                                     key="tune_strat")
        optimize_metric = st.selectbox("Optimize For", list(METRIC_DISPLAY_NAMES.keys()),
                                       format_func=lambda k: METRIC_DISPLAY_NAMES[k],
                                       index=list(METRIC_DISPLAY_NAMES.keys()).index("sharpe_ratio"),
                                       key="tune_metric")

    st.subheader("Parameter Ranges")
    st.caption("Define the values to sweep. The tuner will test every combination.")

    grid = ParameterGrid()

    if strategy_name == "rsi_sma":
        col1, col2 = st.columns(2)
        with col1:
            rsi_vals = st.multiselect("RSI Period", [8, 10, 12, 14, 16, 18, 20],
                                      default=[12, 14, 16], key="t_rsi")
            rsi_buy_vals = st.multiselect("RSI Buy", [20, 25, 30, 35, 40],
                                          default=[25, 30, 35], key="t_rsib")
        with col2:
            sma_vals = st.multiselect("SMA Period", [10, 15, 20, 30, 50, 100, 150, 200],
                                      default=[20, 50], key="t_sma")
            rsi_sell_vals = st.multiselect("RSI Sell", [60, 65, 70, 75, 80],
                                           default=[65, 70, 75], key="t_rsis")
        if rsi_vals:
            grid.add("rsi", rsi_period=rsi_vals, rsi_buy=rsi_buy_vals or [30],
                     rsi_sell=rsi_sell_vals or [70])
        if sma_vals:
            grid.add("sma", sma_period=sma_vals)

    elif strategy_name == "demark_confluence":
        col1, col2 = st.columns(2)
        with col1:
            rsi_vals = st.multiselect("RSI Period", [10, 12, 14, 16],
                                      default=[14], key="t_dm_rsi")
            sma_vals = st.multiselect("SMA Period", [15, 20, 30, 50, 100],
                                      default=[20, 50], key="t_dm_sma")
            bb_vals = st.multiselect("BB Period", [10, 15, 20, 25],
                                     default=[20], key="t_dm_bb")
        with col2:
            ss_vals = st.multiselect("SS Period", [5, 8, 10, 15, 20],
                                     default=[10], key="t_dm_ss")
            ct_vals = st.multiselect("Conf Threshold", [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
                                     default=[0.1, 0.2, 0.3], key="t_dm_ct")
            sl_vals = st.multiselect("Stop Loss %", [0.02, 0.03, 0.05, 0.07],
                                     default=[0.03, 0.05], key="t_dm_sl")
        if rsi_vals:
            grid.add("rsi", rsi_period=rsi_vals)
        if sma_vals:
            grid.add("sma", sma_period=sma_vals)
        if bb_vals:
            grid.add("bb", bb_period=bb_vals)
        if ss_vals:
            grid.add("ss", ss_period=ss_vals)
        if ct_vals:
            grid.add("conf", confluence_threshold=ct_vals)
        if sl_vals:
            grid.add("sl", stop_loss_pct=sl_vals)

    total = grid.total_combinations
    st.info(f"**{total}** parameter combinations to test")

    if total > 500:
        st.warning("Large grids can take several minutes. Consider reducing parameter ranges.")

    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)
    if not run_btn:
        return

    try:
        df = load_price_data(symbol, tf)
    except FileNotFoundError:
        st.error(f"Data not found for {symbol} {tf}")
        return

    strat_cls = STRATEGIES[strategy_name]
    runner = TunerRunner(strat_cls, df, grid, metric=optimize_metric)

    progress_bar = st.progress(0, text="Running optimizations...")

    def on_progress(current: int, total: int) -> None:
        progress_bar.progress(current / total, text=f"Combo {current}/{total}")

    results_df = runner.run(progress_callback=on_progress)
    progress_bar.empty()

    if results_df.empty:
        st.error("No results generated.")
        return

    analysis = analyze_results(results_df, metric=optimize_metric)

    # --- best result ---
    st.subheader("Best Combination")
    best = analysis["best"]
    if best:
        metric_cols = list(METRIC_DISPLAY_NAMES.keys())
        param_cols = [c for c in best if c not in metric_cols and c not in ("combo_id", "error")]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Parameters**")
            for p in param_cols:
                st.write(f"- `{p}` = **{best[p]}**")
        with col2:
            st.markdown("**Metrics**")
            for k in metric_cols:
                if k in best:
                    st.write(f"- {METRIC_DISPLAY_NAMES[k]}: **{format_metric(k, best[k])}**")

    # --- robustness ---
    rob = analysis["robustness"]
    if rob:
        is_robust = rob.get("robust", False)
        badge = "Robust" if is_robust else "Fragile"
        badge_color = "#26a69a" if is_robust else "#ef5350"
        st.markdown(
            f"**Robustness:** <span style='color:{badge_color}; font-weight:bold;'>{badge}</span> "
            f"— {rob.get('reason', 'N/A')} "
            f"(neighbors: {rob.get('neighbor_count', 0)}, "
            f"degradation: {rob.get('degradation_pct', 0):.1f}%)",
            unsafe_allow_html=True,
        )

    # --- results table ---
    st.subheader("All Results")
    top_df = analysis.get("top_n")
    if top_df is not None and not top_df.empty:
        st.dataframe(top_df, use_container_width=True, height=400)

    # --- scatter ---
    st.subheader("Parameter Impact")
    numeric_params = [c for c in results_df.columns
                      if c not in list(METRIC_DISPLAY_NAMES.keys()) + ["combo_id", "error"]
                      and pd.api.types.is_numeric_dtype(results_df[c])]

    if len(numeric_params) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_param = st.selectbox("X axis", numeric_params, index=0, key="tune_x")
        with c2:
            y_param = st.selectbox("Y axis", numeric_params,
                                   index=min(1, len(numeric_params) - 1), key="tune_y")
        fig = px.scatter(
            results_df.dropna(subset=[optimize_metric]),
            x=x_param, y=y_param, color=optimize_metric,
            size=results_df[optimize_metric].clip(lower=0).fillna(0) + 0.01,
            color_continuous_scale="RdYlGn",
            hover_data=list(METRIC_DISPLAY_NAMES.keys()),
            template="plotly_dark", height=450,
        )
        fig.update_layout(margin=dict(l=60, r=30, t=20, b=30))
        st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_params) == 1:
        fig = px.bar(
            results_df.dropna(subset=[optimize_metric]).sort_values(optimize_metric, ascending=False),
            x=numeric_params[0], y=optimize_metric,
            color=optimize_metric, color_continuous_scale="RdYlGn",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
