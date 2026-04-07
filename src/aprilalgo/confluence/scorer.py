"""Confluence scoring — tally bull/bear signals across indicators and timeframes."""

from __future__ import annotations

import pandas as pd


def score_confluence(
    df: pd.DataFrame,
    bull_cols: list[str] | None = None,
    bear_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add confluence scores to *df*.

    Auto-detects ``*_bull`` and ``*_bear`` columns if not specified.

    Columns added:
    - ``bull_count`` — number of bullish signals active
    - ``bear_count`` — number of bearish signals active
    - ``bull_total`` — total possible bull signals (non-NaN)
    - ``bear_total`` — total possible bear signals (non-NaN)
    - ``confluence_bull`` — bull_count / bull_total (0.0 - 1.0)
    - ``confluence_bear`` — bear_count / bear_total (0.0 - 1.0)
    - ``confluence_net`` — confluence_bull - confluence_bear (-1.0 to 1.0)
    - ``confluence_direction`` — "LONG", "SHORT", or "NEUTRAL"
    """
    out = df.copy()

    if bull_cols is None:
        bull_cols = [c for c in out.columns if c.endswith("_bull")]
    if bear_cols is None:
        bear_cols = [c for c in out.columns if c.endswith("_bear")]

    if not bull_cols and not bear_cols:
        out["confluence_net"] = 0.0
        out["confluence_direction"] = "NEUTRAL"
        return out

    bull_df = out[bull_cols].astype(float)
    bear_df = out[bear_cols].astype(float)

    out["bull_count"] = bull_df.sum(axis=1)
    out["bear_count"] = bear_df.sum(axis=1)
    out["bull_total"] = bull_df.notna().sum(axis=1)
    out["bear_total"] = bear_df.notna().sum(axis=1)

    out["confluence_bull"] = out["bull_count"] / out["bull_total"].replace(0, 1)
    out["confluence_bear"] = out["bear_count"] / out["bear_total"].replace(0, 1)
    out["confluence_net"] = out["confluence_bull"] - out["confluence_bear"]

    out["confluence_direction"] = "NEUTRAL"
    out.loc[out["confluence_net"] > 0.1, "confluence_direction"] = "LONG"
    out.loc[out["confluence_net"] < -0.1, "confluence_direction"] = "SHORT"

    return out
