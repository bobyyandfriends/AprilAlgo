"""Triple-barrier labeling for supervised learning (López de Prado style).

Barriers are fixed at decision bar *t* from the entry price; only bars *t+1…* may
touch them. See docs/TRIPLE_BARRIER_MATH.md for definitions and tie-breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# Multiclass / signed labels: TP hit first, SL hit first, or vertical (time) exit
LABEL_TAKE_PROFIT = 1
LABEL_STOP_LOSS = -1
LABEL_VERTICAL_TIMEOUT = 0


BothHitPolicy = Literal["stop_loss_first", "take_profit_first"]


@dataclass(frozen=True, slots=True)
class TripleBarrierResult:
    """Outputs from :func:`apply_triple_barrier`, aligned to ``df`` index."""

    label: pd.Series
    """``1`` = upper (profit) barrier first, ``-1`` = lower (stop) first, ``0`` = timeout."""
    barrier_hit_offset: pd.Series
    """Bars after decision row until first hit, or ``NaN`` if unknown / invalid row."""
    entry_price: pd.Series
    """Price at which barriers are anchored (default: ``close`` at decision row)."""


def apply_triple_barrier(
    df: pd.DataFrame,
    *,
    upper_pct: float,
    lower_pct: float,
    vertical_bars: int,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    both_hit_policy: BothHitPolicy = "stop_loss_first",
) -> TripleBarrierResult:
    """
    Assign a triple-barrier label per row using only future OHLC after each row.

    At decision index ``i``, entry = ``close[i]``. Upper and lower barriers are
    ``entry * (1 + upper_pct)`` and ``entry * (1 - lower_pct)``. Starting at bar
    ``i+1``, scan up to ``vertical_bars`` bars; the first touched barrier wins.
    If neither is touched within that window, label is ``LABEL_VERTICAL_TIMEOUT``.

    Rows where fewer than ``vertical_bars`` future bars exist get ``NaN`` labels
    (insufficient horizon to evaluate the vertical barrier).

    Parameters
    ----------
    df
        Must be sorted in chronological order. Requires ``high``, ``low``, ``close``.
    upper_pct
        Profit barrier as a fraction of entry (e.g. ``0.02`` for +2%).
    lower_pct
        Stop barrier as a fraction of entry (e.g. ``0.01`` for -1%).
    vertical_bars
        Maximum number of **future** bars (not counting the decision bar) to scan.
    both_hit_policy
        If upper and lower are both touched on the **same** bar, which counts as
        first (see docs/TRIPLE_BARRIER_MATH.md).

    Returns
    -------
    TripleBarrierResult
        Series aligned to ``df.index``.
    """
    _require_columns(df, close_col, high_col, low_col)
    if vertical_bars < 1:
        raise ValueError("vertical_bars must be >= 1")
    if upper_pct <= 0 or lower_pct <= 0:
        raise ValueError("upper_pct and lower_pct must be positive")

    n = len(df)
    close = df[close_col].to_numpy(dtype=np.float64, copy=False)
    high = df[high_col].to_numpy(dtype=np.float64, copy=False)
    low = df[low_col].to_numpy(dtype=np.float64, copy=False)

    labels = np.full(n, np.nan, dtype=np.float64)
    offsets = np.full(n, np.nan, dtype=np.float64)
    entries = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i + vertical_bars >= n:
            # Not enough future bars to complete the vertical window
            continue

        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue

        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)
        entries[i] = entry

        decided = False
        for k in range(1, vertical_bars + 1):
            j = i + k
            hi = float(high[j])
            lo = float(low[j])
            hit_upper = hi >= upper
            hit_lower = lo <= lower

            if hit_upper and hit_lower:
                if both_hit_policy == "stop_loss_first":
                    labels[i] = LABEL_STOP_LOSS
                else:
                    labels[i] = LABEL_TAKE_PROFIT
                offsets[i] = k
                decided = True
                break
            if hit_lower:
                labels[i] = LABEL_STOP_LOSS
                offsets[i] = k
                decided = True
                break
            if hit_upper:
                labels[i] = LABEL_TAKE_PROFIT
                offsets[i] = k
                decided = True
                break

        if not decided:
            labels[i] = LABEL_VERTICAL_TIMEOUT
            offsets[i] = vertical_bars

    idx = df.index
    return TripleBarrierResult(
        label=pd.Series(labels, index=idx, name="triple_barrier_label"),
        barrier_hit_offset=pd.Series(offsets, index=idx, name="barrier_hit_offset"),
        entry_price=pd.Series(entries, index=idx, name="entry_price"),
    )


def label_inclusive_end_ix(barrier_hit_offset: pd.Series) -> pd.Series:
    """Inclusive integer **position** of the last bar used in the label window.

    For decision at integer position ``i`` with offset ``k`` (bars until event),
    the last bar index involved is ``i + k``. Rows with NaN offset get NaN.

    Use with :class:`aprilalgo.ml.cv.PurgedKFold` as ``sample_t1`` (aligned, 0..n-1).
    """
    n = len(barrier_hit_offset)
    pos = np.arange(n, dtype=np.float64)
    off = barrier_hit_offset.to_numpy(dtype=np.float64, copy=False)
    t1 = pos + off
    t1[~np.isfinite(off)] = np.nan
    return pd.Series(t1, index=barrier_hit_offset.index, name="label_t1")


def _require_columns(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
