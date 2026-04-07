"""John Ehlers cycle indicators — Super Smoother, Roofing Filter, Decycler.

These indicators separate trend from cycle components in price data.
Reference: "Cybernetic Analysis for Stocks and Futures" by John Ehlers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    src = out[column].values.astype(float)
    n = len(src)
    result = np.full(n, np.nan)

    a = np.exp(-np.sqrt(2) * np.pi / period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b
    c3 = -(a * a)
    c1 = 1 - c2 - c3

    if n > 1:
        result[0] = src[0]
        result[1] = src[1]

    for i in range(2, n):
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
    src = out[column].values.astype(float)
    n = len(src)

    alpha_hp = (np.cos(2 * np.pi / hp_period) +
                np.sin(2 * np.pi / hp_period) - 1) / np.cos(2 * np.pi / hp_period)
    hp = np.zeros(n)
    for i in range(2, n):
        hp[i] = ((1 - alpha_hp / 2) * (1 - alpha_hp / 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
                 + 2 * (1 - alpha_hp) * hp[i - 1]
                 - (1 - alpha_hp) * (1 - alpha_hp) * hp[i - 2])

    a = np.exp(-np.sqrt(2) * np.pi / lp_period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / lp_period)
    c2 = b
    c3 = -(a * a)
    c1 = 1 - c2 - c3

    filt = np.zeros(n)
    for i in range(2, n):
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
    src = out[column].values.astype(float)
    n = len(src)

    alpha = (np.cos(2 * np.pi / period) +
             np.sin(2 * np.pi / period) - 1) / np.cos(2 * np.pi / period)

    hp = np.zeros(n)
    for i in range(1, n):
        hp[i] = ((1 - alpha / 2) * (src[i] - src[i - 1])
                 + (1 - alpha) * hp[i - 1])

    dec = src - hp
    col_name = f"decycler_{period}"
    out[col_name] = dec
    out[f"decycler_{period}_bull"] = src > dec
    out[f"decycler_{period}_bear"] = src < dec
    return out
