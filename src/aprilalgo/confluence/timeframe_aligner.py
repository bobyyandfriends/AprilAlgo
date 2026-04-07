"""Align indicator DataFrames from multiple timeframes into a single view.

Uses forward-fill only (no look-ahead): higher-timeframe values persist
until the next bar on that timeframe is available.
"""

from __future__ import annotations

import pandas as pd


def align_timeframes(
    base_df: pd.DataFrame,
    higher_dfs: dict[str, pd.DataFrame],
    signal_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge higher-timeframe signals onto the base (finest) timeframe.

    Parameters
    ----------
    base_df : The finest-granularity DataFrame (e.g. 5-min bars with indicators).
              Must have a ``datetime`` column.
    higher_dfs : Mapping of timeframe label → DataFrame (e.g. {"daily": df_daily}).
                 Each must have a ``datetime`` column and the same indicator columns.
    signal_cols : Which columns to pull from higher timeframes. If None, auto-detect
                  columns ending in ``_bull`` or ``_bear``.

    Returns a copy of base_df with additional columns prefixed by timeframe label
    (e.g. ``daily_rsi_bull``, ``daily_sma_bear``).
    """
    out = base_df.copy()
    out.set_index("datetime", inplace=True)

    for tf_label, tf_df in higher_dfs.items():
        hdf = tf_df.copy()
        hdf.set_index("datetime", inplace=True)

        if signal_cols is None:
            cols = [c for c in hdf.columns if c.endswith("_bull") or c.endswith("_bear")]
        else:
            cols = [c for c in signal_cols if c in hdf.columns]

        if not cols:
            continue

        hdf = hdf[cols]
        hdf = hdf.rename(columns={c: f"{tf_label}_{c}" for c in cols})

        out = out.join(hdf, how="left")
        for col in hdf.columns:
            out[col] = out[col].ffill()

    out.reset_index(inplace=True)
    return out
