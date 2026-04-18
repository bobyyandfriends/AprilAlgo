"""Turn Measurement Index (TMI) — curvature-based trend change detection.

TMI measures the rate of change of the rate of change (second derivative)
of a smoothed price series. When TMI crosses zero, it signals a potential
trend reversal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tmi(
    df: pd.DataFrame,
    period: int = 14,
    smooth: int = 5,
    column: str = "close",
) -> pd.DataFrame:
    """Add Turn Measurement Index and bull/bear signals to *df*.

    Columns added:
    - ``tmi_{period}`` — raw TMI value (second derivative of smoothed price)
    - ``tmi_{period}_bull`` — True when TMI crosses from negative to positive
    - ``tmi_{period}_bear`` — True when TMI crosses from positive to negative
    """
    out = df.copy()
    prices = out[column].values.astype(float)
    n = len(prices)

    if n == 0:
        col_name = f"tmi_{period}"
        out[col_name] = np.array([], dtype=float)
        out[f"tmi_{period}_bull"] = np.array([], dtype=bool)
        out[f"tmi_{period}_bear"] = np.array([], dtype=bool)
        return out

    smoothed = pd.Series(prices).rolling(window=smooth, min_periods=1).mean().values

    roc = np.zeros(n)
    for i in range(period, n):
        roc[i] = (smoothed[i] - smoothed[i - period]) / (smoothed[i - period] + 1e-10)

    tmi_vals = np.zeros(n)
    for i in range(1, n):
        tmi_vals[i] = roc[i] - roc[i - 1]

    tmi_smooth = pd.Series(tmi_vals).rolling(window=smooth, min_periods=1).mean().values

    col_name = f"tmi_{period}"
    out[col_name] = tmi_smooth

    prev = np.roll(tmi_smooth, 1)
    prev[0] = 0.0
    out[f"tmi_{period}_bull"] = (tmi_smooth > 0) & (prev <= 0)
    out[f"tmi_{period}_bear"] = (tmi_smooth < 0) & (prev >= 0)
    return out
