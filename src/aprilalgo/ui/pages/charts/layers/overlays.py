"""Overlay renderers — traces drawn directly on the candlestick row.

Each renderer is keyed by indicator name (matching ``IndicatorSpec.name``) and
consumes pre-computed columns produced by the indicator pipeline. Renderers
are no-ops when the expected column is absent.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_OVERLAY_COLORS = {
    "sma": "#ffa726",
    "super_smoother": "#ab47bc",
    "decycler": "#66bb6a",
    "bollinger_bands": "#90caf9",
}


def render_overlay(
    fig: go.Figure,
    df: pd.DataFrame,
    name: str,
    params: dict,
    *,
    price_row: int = 1,
) -> None:
    """Dispatch to the right overlay renderer by indicator ``name``."""
    if name == "sma":
        col = f"sma_{params.get('period', 20)}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name=col.upper(),
                    line=dict(color=_OVERLAY_COLORS["sma"], width=1.5),
                ),
                row=price_row,
                col=1,
            )

    elif name == "bollinger_bands":
        p = params.get("period", 20)
        upper, lower, mid = f"bb_{p}_upper", f"bb_{p}_lower", f"bb_{p}_mid"
        if upper in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[upper],
                    name="BB Upper",
                    line=dict(color="#90caf9", width=1, dash="dot"),
                ),
                row=price_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[lower],
                    name="BB Lower",
                    line=dict(color="#90caf9", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(144,202,249,0.08)",
                ),
                row=price_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[mid],
                    name="BB Mid",
                    line=dict(color="#64b5f6", width=1),
                ),
                row=price_row,
                col=1,
            )

    elif name == "super_smoother":
        col = f"ss_{params.get('period', 10)}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name="Super Smoother",
                    line=dict(color=_OVERLAY_COLORS["super_smoother"], width=1.5),
                ),
                row=price_row,
                col=1,
            )

    elif name == "decycler":
        col = f"decycler_{params.get('period', 125)}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name="Decycler",
                    line=dict(color=_OVERLAY_COLORS["decycler"], width=1.5),
                ),
                row=price_row,
                col=1,
            )

    elif name == "demark":
        # Completion triangles (9/13). Per-bar integers are rendered by
        # ``layers.demark_counts.render_demark_counts``.
        if "td_bull" in df.columns:
            buys = df[df["td_bull"]]
            sells = df[df["td_bear"]] if "td_bear" in df.columns else df.iloc[0:0]
            if len(buys):
                fig.add_trace(
                    go.Scatter(
                        x=buys["datetime"],
                        y=buys["low"] * 0.998,
                        mode="markers",
                        name="TD Buy",
                        marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                    ),
                    row=price_row,
                    col=1,
                )
            if len(sells):
                fig.add_trace(
                    go.Scatter(
                        x=sells["datetime"],
                        y=sells["high"] * 1.002,
                        mode="markers",
                        name="TD Sell",
                        marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                    ),
                    row=price_row,
                    col=1,
                )
