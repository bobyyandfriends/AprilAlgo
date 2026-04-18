"""XGBoost probability visualization layer.

Two binding modes share one sub-panel row:

1. **Area chart** (default) — filled scatter of ``P(positive)`` across time
   with a dashed threshold line.
2. **Multi-class stacked** — for three-class ``{-1, 0, 1}`` models, three
   stackgroup scatters summing to 1.0 (red / grey / green).

Optionally shades the price row with a translucent ``add_vrect`` for
contiguous spans where ``P(positive) >= threshold`` — subtle by design so it
does not fight the candlesticks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_POS_COLOR = "#26a69a"
_NEG_COLOR = "#ef5350"
_NEU_COLOR = "#78909c"


def _positive_proba_col(proba: pd.DataFrame) -> str | None:
    """Pick the column that carries the class-1 probability.

    Accepts ``oof_proba_1`` / ``oof_proba_1.0`` / ``proba_1``. Falls back to
    the last numeric probability column when unambiguous.
    """
    candidates = [
        "oof_proba_1",
        "oof_proba_1.0",
        "proba_1",
        "proba_1.0",
        "p_pos",
    ]
    for c in candidates:
        if c in proba.columns:
            return c
    proba_cols = [c for c in proba.columns if c.lower().startswith(("oof_proba_", "proba_"))]
    if not proba_cols:
        return None
    return proba_cols[-1]


def _proba_class_columns(proba: pd.DataFrame) -> list[tuple[float, str]]:
    """Return ``(class_value, column_name)`` pairs sorted by class, low→high."""
    out: list[tuple[float, str]] = []
    for c in proba.columns:
        low = c.lower()
        if not (low.startswith("oof_proba_") or low.startswith("proba_")):
            continue
        raw = c.split("_", maxsplit=2)[-1]
        try:
            out.append((float(raw), c))
        except ValueError:
            continue
    out.sort(key=lambda t: t[0])
    return out


def render_ml_proba_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    proba: pd.DataFrame,
    *,
    row: int,
    threshold: float = 0.55,
    mode: str = "area",
) -> None:
    """Render the probability sub-panel at subplot *row*.

    Parameters
    ----------
    fig
        Figure with the subplot grid already allocated.
    df
        Price frame whose ``datetime`` column defines the x-axis.
    proba
        Probability frame aligned row-for-row with *df* (NaN padding is fine).
        Must carry at least one ``(oof_)?proba_<c>`` column.
    row
        Target subplot row for the panel.
    threshold
        Dashed-line threshold drawn on the panel (and used for price-band
        shading by :func:`shade_price_band_by_proba`).
    mode
        ``"area"`` (default) or ``"stacked"`` for the multi-class view.
    """
    pos_col = _positive_proba_col(proba)
    if pos_col is None or len(proba) == 0:
        return

    x = df["datetime"].iloc[: len(proba)]

    if mode == "stacked":
        class_cols = _proba_class_columns(proba)
        if len(class_cols) >= 2:
            palette = {
                -1.0: _NEG_COLOR,
                0.0: _NEU_COLOR,
                1.0: _POS_COLOR,
            }
            for cls_val, col in class_cols:
                color = palette.get(cls_val, _NEU_COLOR)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=proba[col],
                        name=f"P({cls_val:g})",
                        mode="lines",
                        line=dict(width=0.5, color=color),
                        stackgroup="ml_proba",
                        fillcolor=_rgba(color, 0.55),
                        hovertemplate=f"P({cls_val:g})=%{{y:.2f}}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(range=[0, 1], row=row, col=1)
            return

    # default: area chart of positive class
    fig.add_trace(
        go.Scatter(
            x=x,
            y=proba[pos_col],
            name="P(+)",
            mode="lines",
            line=dict(color=_POS_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor=_rgba(_POS_COLOR, 0.25),
            hovertemplate="P(+)=%{y:.2f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        row=row,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255,255,255,0.15)",
        row=row,
        col=1,
    )
    fig.update_yaxes(range=[0, 1], row=row, col=1)


def shade_price_band_by_proba(
    fig: go.Figure,
    df: pd.DataFrame,
    proba: pd.DataFrame,
    *,
    price_row: int = 1,
    threshold: float = 0.55,
    max_spans: int = 40,
) -> None:
    """Shade ``price_row`` where ``P(positive) >= threshold`` using ``add_vrect``.

    Capped at *max_spans* spans to keep the figure light. Only the longest
    contiguous runs are kept when the cap is exceeded.
    """
    pos_col = _positive_proba_col(proba)
    if pos_col is None:
        return
    p = np.asarray(proba[pos_col].to_numpy(), dtype=np.float64)
    if p.size == 0:
        return
    mask = np.where(np.isnan(p), False, p >= float(threshold))
    if not mask.any():
        return

    x = df["datetime"].iloc[: len(p)].reset_index(drop=True)

    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        x0 = x.iloc[i]
        x1 = x.iloc[j - 1]
        spans.append((x0, x1, j - i))
        i = j

    if len(spans) > max_spans:
        spans = sorted(spans, key=lambda t: t[2], reverse=True)[:max_spans]

    yref = "y" if price_row == 1 else f"y{price_row}"
    for x0, x1, _ in spans:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=_rgba(_POS_COLOR, 0.06),
            line_width=0,
            layer="below",
            row=price_row,
            col=1,
            annotation=None,
            yref=yref,
        )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"
