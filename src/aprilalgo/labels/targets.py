"""Unified triple-barrier target columns for ML (multiclass, binary, t0/t1).

Wraps :func:`aprilalgo.labels.triple_barrier.apply_triple_barrier` and
:func:`aprilalgo.labels.triple_barrier.label_inclusive_end_ix`. See
``docs/DATA_SCHEMA.md`` §5a.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from aprilalgo.labels.triple_barrier import (
    LABEL_STOP_LOSS,
    LABEL_TAKE_PROFIT,
    LABEL_VERTICAL_TIMEOUT,
    BothHitPolicy,
    TripleBarrierResult,
    apply_triple_barrier,
    label_inclusive_end_ix,
)

BarrierHit = Literal["take_profit", "stop_loss", "vertical_timeout"]


def barrier_hit_name(label: float) -> str | None:
    """Map multiclass code to string; NaN / non-finite -> None."""
    if not np.isfinite(label):
        return None
    if int(label) == LABEL_TAKE_PROFIT:
        return "take_profit"
    if int(label) == LABEL_STOP_LOSS:
        return "stop_loss"
    if int(label) == LABEL_VERTICAL_TIMEOUT:
        return "vertical_timeout"
    return None


def build_triple_barrier_targets(
    df: pd.DataFrame,
    *,
    upper_pct: float,
    lower_pct: float,
    vertical_bars: int,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    both_hit_policy: BothHitPolicy = "stop_loss_first",
) -> pd.DataFrame:
    """Return a frame aligned to *df*.index with ML label columns.

    Columns
    -------
    label_multiclass
        ``1`` = TP first, ``-1`` = SL first, ``0`` = vertical timeout, ``NaN`` =
        insufficient future bars (same as triple-barrier).
    label_binary
        ``1`` if take-profit hit first, ``0`` otherwise (SL or vertical), ``NaN``
        where multiclass is NaN. Use for "hit upper barrier first" classification.

        **Semantics (§AUDIT B11):** this is strictly *TP-vs-rest*; the negative
        class collapses stop-loss and vertical-timeout into a single label. A
        strategy thresholding ``P(class=1)`` therefore cannot distinguish
        "neutral, uncertain" from "active downside". If directional context
        matters (e.g. gating shorts), prefer ``label_multiclass`` and consume
        the take-profit column of the resulting probability matrix.
    label_t0
        Integer **position** ``0..n-1`` of the decision bar (row index in *df*).
    label_t1
        Inclusive end **position** of the label-formation window for purged CV.
    barrier_hit
        ``take_profit`` | ``stop_loss`` | ``vertical_timeout`` or ``NaN``.
    entry_price
        Barrier anchor (close at decision bar) from triple-barrier result.
    barrier_hit_offset
        Bars from decision until event (from triple-barrier).
    """
    tb = apply_triple_barrier(
        df,
        upper_pct=upper_pct,
        lower_pct=lower_pct,
        vertical_bars=vertical_bars,
        close_col=close_col,
        high_col=high_col,
        low_col=low_col,
        both_hit_policy=both_hit_policy,
    )
    n = len(df)
    t0 = pd.Series(np.arange(n, dtype=np.int64), index=df.index, name="label_t0")
    t1 = label_inclusive_end_ix(tb.barrier_hit_offset)
    t1.name = "label_t1"

    mc = tb.label
    valid = mc.notna()
    lb_vals = np.where(
        valid.to_numpy(),
        (mc == LABEL_TAKE_PROFIT).to_numpy(dtype=np.float64),
        np.nan,
    )
    lb = pd.Series(lb_vals, index=df.index, name="label_binary")

    hits = mc.map(barrier_hit_name)
    hits.name = "barrier_hit"

    out = pd.DataFrame(
        {
            "label_multiclass": mc,
            "label_binary": lb,
            "label_t0": t0,
            "label_t1": t1,
            "barrier_hit": hits,
            "entry_price": tb.entry_price,
            "barrier_hit_offset": tb.barrier_hit_offset,
        },
        index=df.index,
    )
    return out


def targets_from_triple_barrier_result(
    df_index: pd.Index,
    tb: TripleBarrierResult,
) -> pd.DataFrame:
    """Build the same target columns from an existing :class:`TripleBarrierResult`."""
    n = len(df_index)
    t0 = pd.Series(np.arange(n, dtype=np.int64), index=df_index, name="label_t0")
    t1 = label_inclusive_end_ix(tb.barrier_hit_offset)
    t1.name = "label_t1"
    mc = tb.label
    valid = mc.notna()
    lb_vals = np.where(
        valid.to_numpy(),
        (mc == LABEL_TAKE_PROFIT).to_numpy(dtype=np.float64),
        np.nan,
    )
    lb = pd.Series(lb_vals, index=df_index, name="label_binary")
    hits = mc.map(barrier_hit_name)
    hits.name = "barrier_hit"
    return pd.DataFrame(
        {
            "label_multiclass": mc,
            "label_binary": lb,
            "label_t0": t0,
            "label_t1": t1,
            "barrier_hit": hits,
            "entry_price": tb.entry_price,
            "barrier_hit_offset": tb.barrier_hit_offset,
        },
        index=df_index,
    )
