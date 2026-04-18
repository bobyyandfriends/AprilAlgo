"""SHAP visualization layers.

Two complementary views:

* **In-timeline stacked bars** (:func:`render_shap_stack_panel`) — signed
  contribution per bar, stacked across the globally top-K features (colour
  stability matters; per-bar top-K would shuffle the palette on pan).
* **Companion panels** (:func:`render_shap_side_panel`) — horizontal bar for
  the global mean ``|SHAP|`` ranking and a local waterfall-style bar chart
  for a single selected bar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_POS_COLOR = "#26a69a"
_NEG_COLOR = "#ef5350"
_OTHER_COLOR = "#78909c"

_DEFAULT_PALETTE = (
    "#42a5f5",
    "#ab47bc",
    "#ffa726",
    "#26a69a",
    "#ef5350",
    "#66bb6a",
    "#7e57c2",
    "#ff7043",
    "#29b6f6",
    "#d4e157",
)


def _top_k_features_by_mean_abs(shap_matrix: pd.DataFrame, k: int) -> list[str]:
    means = shap_matrix.abs().mean(axis=0).sort_values(ascending=False)
    return [str(c) for c in means.head(int(k)).index]


def _feature_palette(features: Sequence[str]) -> dict[str, str]:
    return {f: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, f in enumerate(features)}


def render_shap_stack_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    shap_matrix: pd.DataFrame,
    *,
    row: int,
    top_k: int = 5,
    aggregate_remainder: bool = True,
) -> list[str]:
    """Draw a signed stacked-bar SHAP sub-panel at *row*.

    Returns the list of features that were rendered (useful for the side
    panel's legend / palette reuse).

    Parameters
    ----------
    fig
        Target figure (with the subplot already reserved for SHAP).
    df
        Price frame — supplies the x-axis timestamps.
    shap_matrix
        Wide ``(n_samples, n_features)`` dataframe aligned row-for-row with
        *df*. Rows beyond ``len(shap_matrix)`` are dropped from the x axis.
    row
        Target subplot row.
    top_k
        Number of globally most-important features to stack individually.
    aggregate_remainder
        When True, remaining features are summed into a grey "Other" band so
        the panel still reflects the full SHAP contribution.
    """
    if shap_matrix is None or shap_matrix.empty:
        return []

    n = min(len(df), len(shap_matrix))
    if n <= 0:
        return []

    x = df["datetime"].iloc[:n].reset_index(drop=True)
    sm = shap_matrix.iloc[:n].reset_index(drop=True)

    features = _top_k_features_by_mean_abs(sm, top_k)
    palette = _feature_palette(features)

    for feat in features:
        fig.add_trace(
            go.Bar(
                x=x,
                y=sm[feat],
                name=feat,
                marker_color=palette[feat],
                marker_line_width=0,
                hovertemplate=f"{feat}: %{{y:+.3f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    if aggregate_remainder:
        others = [c for c in sm.columns if c not in features]
        if others:
            remainder = sm[others].sum(axis=1)
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=remainder,
                    name="Other",
                    marker_color=_OTHER_COLOR,
                    marker_line_width=0,
                    opacity=0.6,
                    hovertemplate="Other: %{y:+.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, row=row, col=1, line_color="rgba(255,255,255,0.3)", line_width=0.5)
    return features


def render_global_importance(
    importance: pd.DataFrame,
    *,
    top_k: int = 15,
    dark_mode: bool = True,
) -> None:
    """Render a horizontal global-importance bar chart into the active Streamlit column."""
    if importance is None or importance.empty:
        st.caption("No SHAP importance available.")
        return

    col = "mean_abs_shap" if "mean_abs_shap" in importance.columns else importance.columns[-1]
    imp = importance[["feature", col]].dropna().sort_values(col, ascending=True).tail(int(top_k))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=imp[col],
            y=imp["feature"],
            orientation="h",
            marker_color=_POS_COLOR,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=28 * len(imp) + 60,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Global |SHAP| importance",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_local_bar(
    shap_matrix: pd.DataFrame,
    sample_idx: int,
    *,
    top_k: int = 15,
    dark_mode: bool = True,
) -> None:
    """Render a signed horizontal bar of the SHAP values for one row."""
    if shap_matrix is None or shap_matrix.empty:
        st.caption("Select a bar to see its local SHAP breakdown.")
        return
    if sample_idx < 0 or sample_idx >= len(shap_matrix):
        st.caption("Selected bar is outside the SHAP range.")
        return

    row = shap_matrix.iloc[int(sample_idx)]
    ranked = row.reindex(row.abs().sort_values(ascending=False).index).head(int(top_k))
    ranked = ranked.iloc[::-1]  # bottom-to-top for plotly horizontal bars
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ranked.to_numpy()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranked.to_numpy(),
            y=[str(i) for i in ranked.index],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=28 * len(ranked) + 60,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Local SHAP (row {sample_idx})",
        showlegend=False,
    )
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=0.5)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def align_datetime_to_sample_idx(
    shap_matrix: pd.DataFrame,
    df: pd.DataFrame,
    selected_dt: pd.Timestamp | None,
) -> int | None:
    """Map a selected price-chart timestamp to a row index within *shap_matrix*.

    Returns ``None`` if no SHAP row covers the timestamp (e.g. warm-up bars).
    """
    if selected_dt is None or shap_matrix is None or shap_matrix.empty:
        return None
    n = min(len(df), len(shap_matrix))
    if n <= 0:
        return None
    x = df["datetime"].iloc[:n].reset_index(drop=True)
    diffs = (x - pd.Timestamp(selected_dt)).abs()
    idx = int(np.argmin(diffs.to_numpy()))
    return idx
