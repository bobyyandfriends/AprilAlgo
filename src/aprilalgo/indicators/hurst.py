"""Hurst Exponent — detect trend persistence vs mean-reversion.

H > 0.5 → trending (positive feedback / momentum)
H ≈ 0.5 → random walk
H < 0.5 → mean-reverting (negative feedback)

Uses Rescaled Range (R/S) analysis computed over rolling windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hurst_rs(series: np.ndarray) -> float:
    """Compute Hurst exponent via R/S analysis on a 1-D array."""
    n = len(series)
    if n < 20:
        return np.nan

    max_k = min(n // 2, 256)
    sizes = []
    rs_values = []

    k = 10
    while k <= max_k:
        num_chunks = n // k
        if num_chunks < 1:
            break

        rs_list = []
        for chunk_idx in range(num_chunks):
            chunk = series[chunk_idx * k : (chunk_idx + 1) * k]
            mean = chunk.mean()
            deviations = chunk - mean
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(k)
            rs_values.append(np.mean(rs_list))

        k = int(k * 1.5) if k < 50 else k + 20

    if len(sizes) < 3:
        return np.nan

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def hurst(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    column: str = "close",
    trend_threshold: float = 0.55,
    revert_threshold: float = 0.45,
) -> pd.DataFrame:
    """Add Hurst exponent columns and bull/bear signals to *df*.

    Columns added (for each window w):

    - ``hurst_{w}`` — rolling Hurst exponent
    - ``hurst_bull`` — True when a majority of windows are trending (H > ``trend_threshold``)
      **and** the last close is higher than the previous close (strong uptrend).
    - ``hurst_bear`` — True when a majority of windows are trending (H > ``trend_threshold``)
      **and** the last close is lower than the previous close (strong downtrend).
    - ``hurst_mean_revert`` — True when a majority of windows have H < ``revert_threshold``
      (useful standalone context for both sides; does **not** drive bull / bear directly).

    The bull/bear interpretation is deliberately asymmetric with the
    mean-revert signal: trending markets reinforce the current direction, so
    we gate bull / bear on both the Hurst verdict *and* the last-bar price
    direction; mean-reversion is exposed as an independent flag rather than
    folded into bear (§AUDIT B16).
    """
    if windows is None:
        windows = [50, 100, 200]

    out = df.copy()
    prices = out[column].values
    n = len(prices)

    hurst_cols = []
    for w in windows:
        col_name = f"hurst_{w}"
        hurst_vals = np.full(n, np.nan)
        for i in range(w, n):
            hurst_vals[i] = _hurst_rs(prices[i - w : i])
        out[col_name] = hurst_vals
        hurst_cols.append(col_name)

    hurst_matrix = out[hurst_cols]
    trending = (hurst_matrix > trend_threshold).sum(axis=1)
    reverting = (hurst_matrix < revert_threshold).sum(axis=1)
    majority = len(windows) / 2.0

    price_rising = out[column] > out[column].shift(1)

    out["hurst_bull"] = (trending > majority) & price_rising
    out["hurst_bear"] = (trending > majority) & ~price_rising
    out["hurst_mean_revert"] = reverting > majority
    return out
