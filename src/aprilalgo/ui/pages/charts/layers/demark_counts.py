"""DeMark TD Sequential integer annotations.

Renders the running Setup (1-9) and Countdown (1-13) counts as small numeric
labels just above/below each bar. Completion triangles remain the
responsibility of :mod:`layers.overlays` (under the ``"demark"`` branch).

Numbers are emitted as *annotations* rather than scatter traces to avoid
bloating the trace list — Plotly slows noticeably beyond ~100 traces and we
would easily exceed that with one marker-and-label trace per bar.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_BUY_SETUP_COLOR = "#26a69a"
_BUY_CD_COLOR = "#00867a"
_SELL_SETUP_COLOR = "#ef5350"
_SELL_CD_COLOR = "#c0392b"

_PRICE_OFFSET = 0.003  # fraction of price used to nudge labels off the wick


def render_demark_counts(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    min_count: int = 4,
    show_countdown: bool = True,
    price_row: int = 1,
) -> None:
    """Attach TD Setup / Countdown integer annotations to *fig* on ``price_row``.

    Parameters
    ----------
    fig
        Target Plotly figure (already has the candlestick drawn on ``price_row``).
    df
        Price frame enriched with DeMark columns (``td_buy_setup``,
        ``td_sell_setup``, ``td_buy_countdown``, ``td_sell_countdown``).
        Missing columns are silently skipped.
    min_count
        Suppress labels below this count to reduce clutter. Default 4.
    show_countdown
        When True, also render the 1-13 TD countdown in a darker shade.
    price_row
        Subplot row index of the price chart (``yref`` target). Default 1.
    """
    has_cols = {"td_buy_setup", "td_sell_setup", "td_buy_countdown", "td_sell_countdown"}
    if not has_cols.intersection(df.columns):
        return

    yref = "y" if price_row == 1 else f"y{price_row}"

    if "td_buy_setup" in df.columns:
        sel = df[df["td_buy_setup"] >= min_count]
        for _, r in sel.iterrows():
            fig.add_annotation(
                x=r["datetime"],
                y=float(r["low"]) * (1.0 - _PRICE_OFFSET),
                text=str(int(r["td_buy_setup"])),
                showarrow=False,
                font=dict(size=9, color=_BUY_SETUP_COLOR),
                xref="x",
                yref=yref,
            )

    if "td_sell_setup" in df.columns:
        sel = df[df["td_sell_setup"] >= min_count]
        for _, r in sel.iterrows():
            fig.add_annotation(
                x=r["datetime"],
                y=float(r["high"]) * (1.0 + _PRICE_OFFSET),
                text=str(int(r["td_sell_setup"])),
                showarrow=False,
                font=dict(size=9, color=_SELL_SETUP_COLOR),
                xref="x",
                yref=yref,
            )

    if not show_countdown:
        return

    if "td_buy_countdown" in df.columns:
        sel = df[df["td_buy_countdown"] >= min_count]
        for _, r in sel.iterrows():
            fig.add_annotation(
                x=r["datetime"],
                y=float(r["low"]) * (1.0 - _PRICE_OFFSET * 2),
                text=str(int(r["td_buy_countdown"])),
                showarrow=False,
                font=dict(size=9, color=_BUY_CD_COLOR),
                xref="x",
                yref=yref,
            )

    if "td_sell_countdown" in df.columns:
        sel = df[df["td_sell_countdown"] >= min_count]
        for _, r in sel.iterrows():
            fig.add_annotation(
                x=r["datetime"],
                y=float(r["high"]) * (1.0 + _PRICE_OFFSET * 2),
                text=str(int(r["td_sell_countdown"])),
                showarrow=False,
                font=dict(size=9, color=_SELL_CD_COLOR),
                xref="x",
                yref=yref,
            )
