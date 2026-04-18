"""Charts page — Streamlit layout, sidebar controls, and tab orchestration.

The page is driven by three groups of controls in the sidebar:

1. *Chart settings* — timeframe / symbol / N-bars-to-render.
2. *Indicators* — the catalog-driven multiselect with auto-generated parameter
   sliders (this includes the ML / SHAP pseudo-indicators).
3. *Model* — optional model-bundle picker that unlocks the ML and SHAP tabs.

The main column renders three tabs:

* **Price & DeMark** — candles, overlays, panels, DeMark integer counts.
* **ML Overlay** — candles + OOF probability sub-panel + price-band shading.
* **SHAP Explainer** — candles + stacked SHAP bars + side panel with global /
  local SHAP.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from aprilalgo.confluence.scorer import score_confluence
from aprilalgo.data import load_price_data
from aprilalgo.indicators.descriptor import IndicatorSpec, get_catalog
from aprilalgo.ui.helpers import discover_symbols
from aprilalgo.ui.pages.charts.data.ml_artifacts import (
    align_shap_to_price_rows,
    build_feature_frame_for_chart,
    build_proba_frame_for_df,
    bundle_meta,
    compute_shap_tables,
    discover_model_dirs,
    load_bundle,
    project_relative,
    shap_values_to_wide,
)
from aprilalgo.ui.pages.charts.figure import (
    ChartLayout,
    build_figure,
    row_index_for,
    streamlit_chart_config,
)
from aprilalgo.ui.pages.charts.layers.demark_counts import render_demark_counts
from aprilalgo.ui.pages.charts.layers.ml_proba import (
    render_ml_proba_panel,
    shade_price_band_by_proba,
)
from aprilalgo.ui.pages.charts.layers.overlays import render_overlay
from aprilalgo.ui.pages.charts.layers.panels import render_panel
from aprilalgo.ui.pages.charts.layers.shap_local import (
    align_datetime_to_sample_idx,
    render_global_importance,
    render_local_bar,
    render_shap_stack_panel,
)

# UI-only pseudo-indicators that are routed through dedicated renderers rather
# than the generic overlay / panel dispatcher.
_ML_PSEUDO = {"demark_counts", "ml_proba", "shap_local"}

# Hard ceiling for simultaneous candlestick-row overlays (warning shown beyond).
_MAX_OVERLAYS = 5

_DEFAULT_BARS = 1000
_MAX_BARS = 5000


@st.cache_data(show_spinner=False, ttl=600)
def _cached_load_price(symbol: str, timeframe: str) -> pd.DataFrame:
    return load_price_data(symbol, timeframe)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_score_confluence(df: pd.DataFrame) -> pd.DataFrame:
    return score_confluence(df)


def _apply_indicators(
    df: pd.DataFrame,
    selected: list[str],
    ind_params: dict[str, dict[str, Any]],
    catalog: dict[str, IndicatorSpec],
) -> pd.DataFrame:
    """Run the configured indicator functions on *df* in catalog order."""
    out = df
    for name in selected:
        if name in _ML_PSEUDO and name != "demark_counts":
            # ML pseudo-indicators add no columns — skip.
            continue
        spec = catalog[name]
        out = spec(out, **ind_params.get(name, {}))
    return out


def _sidebar_controls(
    catalog: dict[str, IndicatorSpec],
    available: dict[str, list[str]],
) -> dict[str, Any]:
    """Draw the sidebar and return the user selections as a dict."""
    with st.sidebar:
        st.subheader("Chart Settings")

        timeframes = list(available.keys())
        tf_index = timeframes.index("daily") if "daily" in timeframes else 0
        tf = st.selectbox("Timeframe", timeframes, index=tf_index, key="chart_tf")

        symbols = available.get(tf, [])
        sym_index = symbols.index("AAPL") if "AAPL" in symbols else 0
        symbol = st.selectbox("Symbol", symbols, index=sym_index, key="chart_symbol")

        n_bars = st.slider(
            "Bars to render",
            min_value=100,
            max_value=_MAX_BARS,
            value=_DEFAULT_BARS,
            step=100,
            help="Limit the render window — 5k bars × many overlays gets slow.",
        )

        st.subheader("Indicators")
        selected = st.multiselect(
            "Overlay",
            list(catalog.keys()),
            format_func=lambda k: catalog[k].display_name,
            default=["sma", "rsi"],
            key="chart_indicators",
        )

        overlays_selected = [s for s in selected if catalog[s].overlay]
        if len(overlays_selected) > _MAX_OVERLAYS:
            st.warning(
                f"{len(overlays_selected)} price-row overlays selected — "
                f"rendering only the first {_MAX_OVERLAYS} to keep the chart readable.",
                icon="⚠️",
            )

        st.subheader("Parameters")
        ind_params: dict[str, dict[str, Any]] = {}
        for name in selected:
            spec = catalog[name]
            ind_params[name] = {}
            for p in spec.params:
                key = f"chart_{name}_{p.name}"
                if isinstance(p.default, float):
                    ind_params[name][p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        float(p.min_val),
                        float(p.max_val),
                        float(p.default),
                        float(p.step),
                        key=key,
                    )
                else:
                    ind_params[name][p.name] = st.slider(
                        f"{spec.display_name} — {p.display_name}",
                        int(p.min_val),
                        int(p.max_val),
                        int(p.default),
                        int(p.step),
                        key=key,
                    )

        show_confluence = st.checkbox("Show Confluence Score", value=True)

        st.subheader("Model")
        model_dirs = discover_model_dirs()
        model_options = ["(none)"] + [project_relative(p) for p in model_dirs]
        model_choice = st.selectbox(
            "Bundle",
            model_options,
            index=0,
            help="Required for ML Probability and SHAP tabs.",
            key="chart_model_dir",
        )
        selected_model = None if model_choice == "(none)" else model_choice

        dark_mode = st.checkbox(
            "Dark mode",
            value=st.session_state.get("chart_dark_mode", True),
            key="chart_dark_mode",
        )

    return {
        "timeframe": tf,
        "symbol": symbol,
        "n_bars": int(n_bars),
        "selected": selected,
        "overlays_selected": overlays_selected[:_MAX_OVERLAYS],
        "panels_selected": [s for s in selected if not catalog[s].overlay and s not in _ML_PSEUDO],
        "ind_params": ind_params,
        "show_confluence": show_confluence,
        "model_dir": selected_model,
        "dark_mode": dark_mode,
    }


def _live_header(df: pd.DataFrame, proba: pd.DataFrame | None, show_confluence: bool) -> None:
    """Render the 3-metric header (last price / ML prob / confluence)."""
    if df.empty:
        return
    last = df.iloc[-1]
    prev_close = df["close"].iloc[-2] if len(df) >= 2 else float(last["close"])
    change = float(last["close"]) - float(prev_close)
    pct = (change / prev_close) * 100 if prev_close else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Last", f"{float(last['close']):,.2f}", f"{change:+.2f} ({pct:+.2f}%)")
    with c2:
        if proba is not None and not proba.empty:
            pos_col = next(
                (c for c in proba.columns if c.lower().startswith(("oof_proba_", "proba_"))),
                None,
            )
            if pos_col is not None:
                vals = proba[pos_col].dropna()
                last_p = float(vals.iloc[-1]) if len(vals) else float("nan")
                st.metric("P(+)", f"{last_p:.2f}" if last_p == last_p else "n/a")
            else:
                st.metric("P(+)", "n/a")
        else:
            st.metric("P(+)", "n/a")
    with c3:
        if show_confluence and "confluence_net" in df.columns:
            val = float(df["confluence_net"].iloc[-1])
            st.metric("Confluence", f"{val:+.1f}")
        else:
            st.metric("Confluence", "off")


def _draw_indicator_layers(
    fig,
    df: pd.DataFrame,
    *,
    layout: ChartLayout,
    overlays_selected: list[str],
    panels_selected: list[str],
    ind_params: dict[str, dict[str, Any]],
) -> None:
    """Apply the catalog-driven overlay & panel renderers to *fig*."""
    for name in overlays_selected:
        render_overlay(fig, df, name, ind_params.get(name, {}))
    for i, name in enumerate(panels_selected):
        panel_row = row_index_for(layout, name) if name in layout.panel_indicators else 2 + i
        render_panel(fig, df, name, ind_params.get(name, {}), row=panel_row)


def _draw_demark_counts(fig, df: pd.DataFrame, ind_params: dict[str, dict[str, Any]]) -> None:
    """Render TD Sequential integer annotations when ``demark_counts`` is selected."""
    params = ind_params.get("demark_counts", {})
    min_count = int(params.get("min_count", 4))
    render_demark_counts(fig, df, min_count=min_count, show_countdown=True, price_row=1)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_tab_price(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    overlays_selected: list[str],
    panels_selected: list[str],
    ind_params: dict[str, dict[str, Any]],
    show_confluence: bool,
    show_demark_counts: bool,
    dark_mode: bool,
    catalog: dict[str, IndicatorSpec],
) -> None:
    layout = ChartLayout(
        include_volume=True,
        panel_indicators=list(panels_selected),
        include_confluence=show_confluence and "confluence_net" in df.columns,
        include_ml_proba=False,
        include_shap=False,
    )
    panel_display = {n: catalog[n].display_name for n in panels_selected}
    fig = build_figure(
        df,
        symbol=symbol,
        timeframe=timeframe,
        layout=layout,
        dark_mode=dark_mode,
        panel_display=panel_display,
    )
    _draw_indicator_layers(
        fig,
        df,
        layout=layout,
        overlays_selected=overlays_selected,
        panels_selected=panels_selected,
        ind_params=ind_params,
    )
    if show_demark_counts:
        _draw_demark_counts(fig, df, ind_params)
    st.plotly_chart(fig, use_container_width=True, config=streamlit_chart_config())


def _render_tab_ml(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    overlays_selected: list[str],
    panels_selected: list[str],
    ind_params: dict[str, dict[str, Any]],
    show_demark_counts: bool,
    dark_mode: bool,
    catalog: dict[str, IndicatorSpec],
    proba: pd.DataFrame | None,
    threshold: float,
    show_price_shading: bool,
    stacked_view: bool,
) -> None:
    if proba is None or proba.empty:
        st.info(
            "No OOF probability file found for the selected model. "
            "Run `uv run python -m aprilalgo.cli oof --config <your.yaml>` "
            "to produce `oof_primary.csv` under the model directory."
        )
        return

    layout = ChartLayout(
        include_volume=True,
        panel_indicators=list(panels_selected),
        include_confluence=False,
        include_ml_proba=True,
        include_shap=False,
    )
    panel_display = {n: catalog[n].display_name for n in panels_selected}
    fig = build_figure(
        df,
        symbol=symbol,
        timeframe=timeframe,
        layout=layout,
        dark_mode=dark_mode,
        panel_display=panel_display,
    )
    _draw_indicator_layers(
        fig,
        df,
        layout=layout,
        overlays_selected=overlays_selected,
        panels_selected=panels_selected,
        ind_params=ind_params,
    )
    if show_demark_counts:
        _draw_demark_counts(fig, df, ind_params)

    proba_row = row_index_for(layout, "ml_proba")
    render_ml_proba_panel(
        fig,
        df,
        proba,
        row=proba_row,
        threshold=threshold,
        mode="stacked" if stacked_view else "area",
    )
    if show_price_shading:
        shade_price_band_by_proba(fig, df, proba, price_row=1, threshold=threshold, max_spans=40)
    st.plotly_chart(fig, use_container_width=True, config=streamlit_chart_config())


def _render_tab_shap(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    overlays_selected: list[str],
    panels_selected: list[str],
    ind_params: dict[str, dict[str, Any]],
    show_demark_counts: bool,
    dark_mode: bool,
    catalog: dict[str, IndicatorSpec],
    shap_wide: pd.DataFrame | None,
    importance: pd.DataFrame | None,
    top_k: int,
) -> None:
    if shap_wide is None or shap_wide.empty:
        st.info(
            "SHAP values could not be computed. Make sure `shap` is installed "
            "and the selected model bundle's feature columns are present in "
            "the current chart configuration."
        )
        if importance is not None and not importance.empty:
            render_global_importance(importance, top_k=15, dark_mode=dark_mode)
        return

    layout = ChartLayout(
        include_volume=True,
        panel_indicators=list(panels_selected),
        include_confluence=False,
        include_ml_proba=False,
        include_shap=True,
    )
    panel_display = {n: catalog[n].display_name for n in panels_selected}

    chart_col, side_col = st.columns([3, 1])
    with chart_col:
        fig = build_figure(
            df,
            symbol=symbol,
            timeframe=timeframe,
            layout=layout,
            dark_mode=dark_mode,
            panel_display=panel_display,
        )
        _draw_indicator_layers(
            fig,
            df,
            layout=layout,
            overlays_selected=overlays_selected,
            panels_selected=panels_selected,
            ind_params=ind_params,
        )
        if show_demark_counts:
            _draw_demark_counts(fig, df, ind_params)

        shap_row = row_index_for(layout, "shap")
        render_shap_stack_panel(
            fig,
            df,
            shap_wide,
            row=shap_row,
            top_k=int(top_k),
        )

        try:  # optional click-to-select
            from streamlit_plotly_events import plotly_events

            events = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                override_height=fig.layout.height,
            )
            selected_dt = pd.Timestamp(events[0]["x"]) if events and "x" in events[0] else None
        except ImportError:
            st.plotly_chart(fig, use_container_width=True, config=streamlit_chart_config())
            dt_options = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            if dt_options:
                pick = st.select_slider(
                    "Inspect bar",
                    options=dt_options,
                    value=dt_options[-1],
                    key="shap_bar_pick",
                )
                selected_dt = pd.to_datetime(pick)
            else:
                selected_dt = None

    with side_col:
        if importance is not None and not importance.empty:
            render_global_importance(importance, top_k=15, dark_mode=dark_mode)
        sample_idx = align_datetime_to_sample_idx(shap_wide, df, selected_dt)
        if sample_idx is None and len(shap_wide):
            sample_idx = len(shap_wide) - 1
        if sample_idx is not None:
            render_local_bar(shap_wide, sample_idx, top_k=15, dark_mode=dark_mode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render() -> None:
    st.header("Price Charts & Indicators")

    available = discover_symbols()
    if not available:
        st.warning("No data found in the `data/` directory. Fetch some data first.")
        return

    catalog = get_catalog()
    ctrl = _sidebar_controls(catalog, available)

    try:
        df = _cached_load_price(ctrl["symbol"], ctrl["timeframe"])
    except FileNotFoundError:
        st.error(f"Could not load {ctrl['symbol']} {ctrl['timeframe']} data.")
        return

    df = _apply_indicators(df, ctrl["selected"], ctrl["ind_params"], catalog)
    if ctrl["show_confluence"]:
        df = _cached_score_confluence(df)

    n = min(int(ctrl["n_bars"]), len(df))
    df_view = df.tail(n).reset_index(drop=True)

    # --- Model artifacts (only loaded if the user picked a bundle) ---
    proba_frame: pd.DataFrame | None = None
    shap_wide: pd.DataFrame | None = None
    shap_importance: pd.DataFrame | None = None
    bundle_sym: str | None = None

    if ctrl["model_dir"]:
        meta = bundle_meta(ctrl["model_dir"])
        bundle_sym = str(meta.get("symbol")) if meta else None
        if bundle_sym and bundle_sym.upper() != str(ctrl["symbol"]).upper():
            st.info(
                f"Selected model was trained on **{bundle_sym}** — "
                f"currently viewing **{ctrl['symbol']}**. Probabilities / SHAP "
                f"may not align.",
                icon="ℹ️",
            )
        sig = (
            ctrl["symbol"],
            ctrl["timeframe"],
            int(df_view["datetime"].iloc[0].value) if len(df_view) else 0,
            int(df_view["datetime"].iloc[-1].value) if len(df_view) else 0,
        )
        proba_frame = build_proba_frame_for_df(ctrl["model_dir"], df_len=len(df_view), df_signature=sig)

        try:
            bundle = load_bundle(ctrl["model_dir"])
        except FileNotFoundError:
            bundle = None

        if bundle is not None:
            ind_cfg = bundle.indicator_config
            feature_frame, valid_mask = build_feature_frame_for_chart(df_view, ind_cfg)
            if valid_mask.any():
                X_valid = feature_frame.loc[valid_mask].reset_index(drop=True)
                shap_tables = compute_shap_tables(
                    ctrl["model_dir"],
                    tuple(X_valid.columns),
                    X_valid,
                    max_samples=500,
                )
                if shap_tables is not None:
                    shap_long = shap_tables.get("values")
                    shap_importance = shap_tables.get("importance")
                    wide = shap_values_to_wide(shap_long) if shap_long is not None else None
                    shap_wide = align_shap_to_price_rows(wide, valid_mask) if wide is not None else None

    # --- Header ---
    _live_header(df_view, proba_frame, ctrl["show_confluence"])

    show_demark_counts = "demark_counts" in ctrl["selected"]
    include_ml = "ml_proba" in ctrl["selected"] and ctrl["model_dir"] is not None
    include_shap = "shap_local" in ctrl["selected"] and ctrl["model_dir"] is not None

    tab_price, tab_ml, tab_shap = st.tabs(["Price & DeMark", "ML Overlay", "SHAP Explainer"])

    with tab_price:
        _render_tab_price(
            df_view,
            symbol=ctrl["symbol"],
            timeframe=ctrl["timeframe"],
            overlays_selected=ctrl["overlays_selected"],
            panels_selected=ctrl["panels_selected"],
            ind_params=ctrl["ind_params"],
            show_confluence=ctrl["show_confluence"],
            show_demark_counts=show_demark_counts,
            dark_mode=ctrl["dark_mode"],
            catalog=catalog,
        )

    with tab_ml:
        if not ctrl["model_dir"]:
            st.info("Select a model bundle in the sidebar to enable the ML overlay.")
        elif not include_ml:
            st.info("Enable **ML Probability** in the Indicators sidebar to render this tab.")
        else:
            mp = ctrl["ind_params"].get("ml_proba", {})
            threshold = float(mp.get("threshold", 0.55))
            col_a, col_b = st.columns([1, 1])
            with col_a:
                stacked_view = st.checkbox("Stacked multi-class view", value=False, key="ml_stacked_view")
            with col_b:
                show_price_shading = st.checkbox("Shade price band above threshold", value=True, key="ml_price_shade")
            _render_tab_ml(
                df_view,
                symbol=ctrl["symbol"],
                timeframe=ctrl["timeframe"],
                overlays_selected=ctrl["overlays_selected"],
                panels_selected=ctrl["panels_selected"],
                ind_params=ctrl["ind_params"],
                show_demark_counts=show_demark_counts,
                dark_mode=ctrl["dark_mode"],
                catalog=catalog,
                proba=proba_frame,
                threshold=threshold,
                show_price_shading=show_price_shading,
                stacked_view=stacked_view,
            )

    with tab_shap:
        if not ctrl["model_dir"]:
            st.info("Select a model bundle in the sidebar to enable the SHAP explainer.")
        elif not include_shap:
            st.info("Enable **SHAP Contributions** in the Indicators sidebar to render this tab.")
        else:
            sp = ctrl["ind_params"].get("shap_local", {})
            top_k = int(sp.get("top_k", 5))
            _render_tab_shap(
                df_view,
                symbol=ctrl["symbol"],
                timeframe=ctrl["timeframe"],
                overlays_selected=ctrl["overlays_selected"],
                panels_selected=ctrl["panels_selected"],
                ind_params=ctrl["ind_params"],
                show_demark_counts=show_demark_counts,
                dark_mode=ctrl["dark_mode"],
                catalog=catalog,
                shap_wide=shap_wide,
                importance=shap_importance,
                top_k=top_k,
            )

    with st.expander("Raw Data", expanded=False):
        base_cols = ["datetime", "open", "high", "low", "close", "volume"]
        extra = [c for c in df_view.columns if c not in base_cols]
        st.dataframe(
            df_view[base_cols + extra].tail(100),
            use_container_width=True,
            height=400,
        )
