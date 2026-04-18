"""John Ehlers cycle indicators — Super Smoother, Roofing Filter, Decycler.

These indicators separate trend from cycle components in price data.
Reference: "Cybernetic Analysis for Stocks and Futures" by John Ehlers.

NaN handling (§AUDIT B17)
-------------------------
All three filters are recursive (``result[i]`` depends on ``result[i-1]`` and
``result[i-2]``). A single NaN in the source column — from a resampling gap,
a chained upstream indicator, or a bad tick — would otherwise poison the
entire downstream tail. :func:`_forward_fill_source` forward-fills the
source array *for the recursion only*; filter output is still returned for
every row, and NaN originally present in the leading prefix stays NaN
(no phantom readings before the first real price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_fill_source(src: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ``(filled, first_valid_ix)``.

    Forward-fills interior NaN gaps with the last valid price so the IIR
    recursion stays finite. The returned ``first_valid_ix`` is the index of
    the first non-NaN input; earlier positions remain ``NaN`` in the final
    output so we don't synthesise values before the series begins.
    """
    n = src.size
    if n == 0:
        return src.copy(), n
    finite = np.isfinite(src)
    if not finite.any():
        return src.copy(), n
    first_valid = int(np.argmax(finite))
    filled = src.copy()
    last_valid = filled[first_valid]
    for i in range(first_valid, n):
        if np.isfinite(filled[i]):
            last_valid = filled[i]
        else:
            filled[i] = last_valid
    return filled, first_valid


def super_smoother(
    df: pd.DataFrame,
    period: int = 10,
    column: str = "close",
) -> pd.DataFrame:
    """Add Super Smoother filter and bull/bear signals to *df*.

    Columns added:
    - ``ss_{period}`` — smoothed value
    - ``ss_{period}_bull`` — True when price is above the smoother
    - ``ss_{period}_bear`` — True when price is below the smoother
    """
    out = df.copy()
    src_raw = out[column].values.astype(float)
    src, first_valid = _forward_fill_source(src_raw)
    n = len(src)
    result = np.full(n, np.nan)

    a = np.exp(-np.sqrt(2) * np.pi / period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b
    c3 = -(a * a)
    c1 = 1 - c2 - c3

    if first_valid < n:
        result[first_valid] = src[first_valid]
    if first_valid + 1 < n:
        result[first_valid + 1] = src[first_valid + 1]

    for i in range(first_valid + 2, n):
        result[i] = c1 * (src[i] + src[i - 1]) / 2 + c2 * result[i - 1] + c3 * result[i - 2]

    col_name = f"ss_{period}"
    out[col_name] = result
    out[f"ss_{period}_bull"] = out[column] > out[col_name]
    out[f"ss_{period}_bear"] = out[column] < out[col_name]
    return out


def roofing_filter(
    df: pd.DataFrame,
    hp_period: int = 48,
    lp_period: int = 10,
    column: str = "close",
) -> pd.DataFrame:
    """Add Roofing Filter (bandpass) and bull/bear signals to *df*.

    Columns added:
    - ``roof_{hp_period}_{lp_period}`` — filtered cycle component
    - ``roof_{hp_period}_{lp_period}_bull`` — True when cycle is positive
    - ``roof_{hp_period}_{lp_period}_bear`` — True when cycle is negative
    """
    out = df.copy()
    src_raw = out[column].values.astype(float)
    src, first_valid = _forward_fill_source(src_raw)
    n = len(src)

    alpha_hp = (np.cos(2 * np.pi / hp_period) + np.sin(2 * np.pi / hp_period) - 1) / np.cos(2 * np.pi / hp_period)
    hp = np.zeros(n)
    for i in range(first_valid + 2, n):
        hp[i] = (
            (1 - alpha_hp / 2) * (1 - alpha_hp / 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
            + 2 * (1 - alpha_hp) * hp[i - 1]
            - (1 - alpha_hp) * (1 - alpha_hp) * hp[i - 2]
        )

    a = np.exp(-np.sqrt(2) * np.pi / lp_period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / lp_period)
    c2 = b
    c3 = -(a * a)
    c1 = 1 - c2 - c3

    filt = np.full(n, np.nan)
    if first_valid + 2 < n:
        # Seed the first two post-warm-up samples at 0 (Ehlers convention) so
        # the recursion has a well-defined starting point.
        filt[first_valid] = 0.0
        filt[first_valid + 1] = 0.0
        for i in range(first_valid + 2, n):
            filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    col_name = f"roof_{hp_period}_{lp_period}"
    out[col_name] = filt
    out[f"{col_name}_bull"] = filt > 0
    out[f"{col_name}_bear"] = filt < 0
    return out


def decycler(
    df: pd.DataFrame,
    period: int = 125,
    column: str = "close",
) -> pd.DataFrame:
    """Add Ehlers Decycler (trend extraction) and bull/bear signals to *df*.

    Columns added:
    - ``decycler_{period}`` — extracted trend line
    - ``decycler_{period}_bull`` — True when price is above the trend line
    - ``decycler_{period}_bear`` — True when price is below the trend line
    """
    out = df.copy()
    src_raw = out[column].values.astype(float)
    src, first_valid = _forward_fill_source(src_raw)
    n = len(src)

    alpha = (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1) / np.cos(2 * np.pi / period)

    hp = np.zeros(n)
    for i in range(first_valid + 1, n):
        hp[i] = (1 - alpha / 2) * (src[i] - src[i - 1]) + (1 - alpha) * hp[i - 1]

    dec = src - hp
    # Restore NaN in the leading prefix so the output doesn't synthesise trend
    # values for bars before the first real price.
    if first_valid > 0:
        dec[:first_valid] = np.nan
    col_name = f"decycler_{period}"
    out[col_name] = dec
    out[f"decycler_{period}_bull"] = out[column] > dec
    out[f"decycler_{period}_bear"] = out[column] < dec
    return out
