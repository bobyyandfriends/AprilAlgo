"""Volatility regime buckets from realized vol (v0.4).

Rule-based first; optional HMM can be added behind a flag later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, *, window: int = 20) -> pd.Series:
    """Rolling std of log returns (causal)."""
    lr = np.log(close.astype(float)).diff()
    return lr.rolling(window, min_periods=1).std()


def add_vol_regime(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    window: int = 20,
    n_buckets: int = 3,
    out_col: str = "vol_regime",
    use_hmm: bool = False,
    hmm_states: int | None = None,
) -> pd.DataFrame:
    """Quantile buckets on realized vol, or optional Gaussian HMM on log returns.

    HMM path requires ``hmmlearn``. NaN regimes only apply to the quantile path
    where realized vol is undefined.
    """
    out = df.copy()
    vol = realized_vol(out[close_col], window=window)
    out["realized_vol"] = vol

    if use_hmm:
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "add_vol_regime(use_hmm=True) requires the hmmlearn package"
            ) from e
        k = int(hmm_states or n_buckets)
        lr = (
            np.log(out[close_col].astype(float))
            .diff()
            .fillna(0.0)
            .to_numpy()
            .reshape(-1, 1)
        )
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            random_state=0,
            n_iter=100,
        )
        model.fit(lr)
        out[out_col] = model.predict(lr).astype(float)
        return out

    regimes = pd.Series(np.nan, index=out.index, dtype=np.float64)
    valid = vol.notna()
    if valid.sum() == 0:
        out[out_col] = regimes
        return out
    q = pd.qcut(vol.loc[valid], q=n_buckets, labels=False, duplicates="drop")
    regimes.loc[valid] = q.astype(float)
    out[out_col] = regimes
    return out
