"""Plotly figure builder for the charts page.

Assembles the candlestick + volume + optional ML/SHAP sub-panels into a single
figure with:

* shared x-axis across all rows
* crosshair / unified hover across subplots
* ``uirevision`` keyed on ``(symbol, timeframe)`` so zoom & pan survive reruns
* weekend range-breaks on daily charts (same trick used by the QuantHedgeFund
  live-chart renderer in ``dashboard/app.py``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(slots=True)
class ChartLayout:
    """Declarative description of subplot rows.

    Attributes
    ----------
    include_volume
        Render a volume bar sub-panel.
    panel_indicators
        Names of sub-panel indicators (RSI, TMI, Hurst, ...) to render.
    include_confluence
        Append a confluence-net sub-panel (requires ``confluence_net`` column).
    include_ml_proba
        Append an ML probability sub-panel (caller must provide the frame).
    include_shap
        Append a SHAP stacked-bars sub-panel (caller must provide the matrix).
    """

    include_volume: bool = True
    panel_indicators: list[str] = field(default_factory=list)
    include_confluence: bool = False
    include_ml_proba: bool = False
    include_shap: bool = False

    def subplot_titles(self, symbol: str, tf: str, panel_display: dict[str, str]) -> list[str]:
        titles = [f"{symbol} ({tf.upper()})"]
        for name in self.panel_indicators:
            titles.append(panel_display.get(name, name.upper()))
        if self.include_volume:
            titles.append("Volume")
        if self.include_confluence:
            titles.append("Confluence")
        if self.include_ml_proba:
            titles.append("ML Probability")
        if self.include_shap:
            titles.append("SHAP Contribution")
        return titles

    def row_heights(self) -> list[float]:
        extras = (
            len(self.panel_indicators)
            + int(self.include_volume)
            + int(self.include_confluence)
            + int(self.include_ml_proba)
            + int(self.include_shap)
        )
        if extras == 0:
            return [1.0]
        # Price takes 55%; remaining 45% is split across the extra rows,
        # giving ML / SHAP slightly more than Volume / panels.
        weights: list[float] = []
        for name in self.panel_indicators:
            weights.append(0.6)
            _ = name
        if self.include_volume:
            weights.append(0.55)
        if self.include_confluence:
            weights.append(0.55)
        if self.include_ml_proba:
            weights.append(0.85)
        if self.include_shap:
            weights.append(1.0)
        total = sum(weights)
        # reserve 55% for price
        extras_total = 0.45
        scaled = [w / total * extras_total for w in weights]
        return [0.55] + scaled


def build_figure(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    layout: ChartLayout,
    dark_mode: bool = True,
    panel_display: dict[str, str] | None = None,
    apply_weekend_breaks: bool = True,
) -> go.Figure:
    """Create a subplot figure with the candlestick already drawn on row 1.

    Caller layers (overlays / panels / demark_counts / ml_proba / shap) are
    expected to draw into the returned figure using the row indices returned
    by :func:`row_index_for`.
    """
    panel_display = panel_display or {}
    titles = layout.subplot_titles(symbol, timeframe, panel_display)
    heights = layout.row_heights()
    n_rows = len(titles)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=titles,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    if layout.include_volume and "volume" in df.columns:
        vol_row = row_index_for(layout, "volume")
        vol_colors = [
            "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["volume"],
                name="Volume",
                marker_color=vol_colors,
                marker_line_width=0,
                opacity=0.7,
            ),
            row=vol_row,
            col=1,
        )

    if layout.include_confluence and "confluence_net" in df.columns:
        conf_row = row_index_for(layout, "confluence")
        conf_colors = [
            "#26a69a" if v >= 0 else "#ef5350" for v in df["confluence_net"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["confluence_net"],
                name="Confluence Net",
                marker_color=conf_colors,
                marker_line_width=0,
            ),
            row=conf_row,
            col=1,
        )
        fig.add_hline(
            y=0, line_color="rgba(255,255,255,0.4)", line_width=0.5, row=conf_row, col=1
        )

    apply_layout(
        fig,
        n_rows=n_rows,
        symbol=symbol,
        timeframe=timeframe,
        dark_mode=dark_mode,
        apply_weekend_breaks=apply_weekend_breaks,
    )
    return fig


def row_index_for(layout: ChartLayout, slot: str) -> int:
    """Return the subplot row index for a named slot.

    Slot names: ``"price"``, one of the panel indicator names, ``"volume"``,
    ``"confluence"``, ``"ml_proba"``, ``"shap"``. Raises ``KeyError`` if the
    slot is not enabled in *layout*.
    """
    if slot == "price":
        return 1
    order: list[str] = []
    order.extend(layout.panel_indicators)
    if layout.include_volume:
        order.append("volume")
    if layout.include_confluence:
        order.append("confluence")
    if layout.include_ml_proba:
        order.append("ml_proba")
    if layout.include_shap:
        order.append("shap")
    try:
        offset = order.index(slot)
    except ValueError as e:
        raise KeyError(f"Slot {slot!r} is not enabled in the current layout.") from e
    return 2 + offset


def apply_layout(
    fig: go.Figure,
    *,
    n_rows: int,
    symbol: str,
    timeframe: str,
    dark_mode: bool,
    apply_weekend_breaks: bool,
) -> None:
    """Apply the QuantHedgeFund-inspired interactivity baseline.

    * ``uirevision`` keyed on ``(symbol, timeframe)`` keeps zoom / pan across
      Streamlit reruns.
    * ``dragmode='pan'`` + ``hovermode='x unified'`` + ``showspikes`` give the
      crosshair-linked-across-subplots experience.
    * Weekend range-breaks remove Sat/Sun gaps on daily intraday charts.
    """
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        uirevision=f"{symbol}-{timeframe}",
        dragmode="pan",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=40, b=30),
        height=260 + 200 * max(n_rows - 1, 0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
        bargap=0.1,
    )

    spike_color = "rgba(255,255,255,0.3)" if dark_mode else "rgba(0,0,0,0.3)"
    grid_color = "rgba(255,255,255,0.06)" if dark_mode else "rgba(0,0,0,0.06)"
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=spike_color,
        spikethickness=1,
        showgrid=True,
        gridcolor=grid_color,
    )
    fig.update_yaxes(
        showspikes=False,
        showgrid=True,
        gridcolor=grid_color,
    )

    if apply_weekend_breaks:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])


def streamlit_chart_config() -> dict[str, Any]:
    """Canonical ``config`` dict for :func:`streamlit.plotly_chart`."""
    return {
        "scrollZoom": True,
        "doubleClick": "reset",
        "displayModeBar": True,
        "displaylogo": False,
    }
