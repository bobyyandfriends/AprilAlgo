"""Sub-panel renderers — traces drawn below the candlestick, in their own subplot row."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_PANEL_COLORS = {
    "rsi": "#7e57c2",
    "hurst": "#ff7043",
    "roofing_filter": "#42a5f5",
}


def render_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    name: str,
    params: dict,
    *,
    row: int,
) -> None:
    """Dispatch to the right sub-panel renderer for indicator ``name`` at subplot *row*."""
    if name == "rsi":
        col = f"rsi_{params.get('period', 14)}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name="RSI",
                    line=dict(color=_PANEL_COLORS["rsi"], width=1.5),
                ),
                row=row,
                col=1,
            )
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="rgba(239,83,80,0.4)",
                row=row,
                col=1,
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="rgba(38,166,154,0.4)",
                row=row,
                col=1,
            )
            fig.update_yaxes(range=[0, 100], row=row, col=1)

    elif name == "tmi":
        col = f"tmi_{params.get('period', 14)}"
        if col in df.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[col].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=df["datetime"],
                    y=df[col],
                    name="TMI",
                    marker_color=colors,
                ),
                row=row,
                col=1,
            )
            fig.add_hline(y=0, line_color="white", line_width=0.5, row=row, col=1)

    elif name == "hurst":
        w = params.get("window", 100)
        col = f"hurst_{w}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name=f"Hurst({w})",
                    line=dict(color=_PANEL_COLORS["hurst"], width=1.5),
                ),
                row=row,
                col=1,
            )
            fig.add_hline(
                y=0.55,
                line_dash="dash",
                line_color="rgba(38,166,154,0.4)",
                row=row,
                col=1,
            )
            fig.add_hline(
                y=0.45,
                line_dash="dash",
                line_color="rgba(239,83,80,0.4)",
                row=row,
                col=1,
            )
            fig.update_yaxes(range=[0, 1], row=row, col=1)

    elif name == "roofing_filter":
        hp = params.get("hp_period", 48)
        lp = params.get("lp_period", 10)
        col = f"roof_{hp}_{lp}"
        if col in df.columns:
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[col].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=df["datetime"],
                    y=df[col],
                    name="Roofing",
                    marker_color=colors,
                ),
                row=row,
                col=1,
            )
            fig.add_hline(y=0, line_color="white", line_width=0.5, row=row, col=1)

    elif name == "volume_trend":
        vp = params.get("vol_period", 20)
        ratio_col = f"vol_{vp}_ratio"
        if ratio_col in df.columns:
            colors = ["#26a69a" if v >= 1.5 else "#78909c" for v in df[ratio_col].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=df["datetime"],
                    y=df[ratio_col],
                    name="Vol Ratio",
                    marker_color=colors,
                ),
                row=row,
                col=1,
            )
            fig.add_hline(
                y=1.5,
                line_dash="dash",
                line_color="rgba(255,167,38,0.5)",
                row=row,
                col=1,
            )

    elif name == "pv_sequences":
        if "pv_state" in df.columns:
            state_colors = {0: "#26a69a", 1: "#a5d6a7", 2: "#ef5350", 3: "#ef9a9a"}
            colors = [state_colors.get(s, "#78909c") for s in df["pv_state"].fillna(3)]
            fig.add_trace(
                go.Bar(
                    x=df["datetime"],
                    y=df["pv_streak"],
                    name="PV Streak",
                    marker_color=colors,
                ),
                row=row,
                col=1,
            )
