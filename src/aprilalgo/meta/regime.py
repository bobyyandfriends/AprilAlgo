"""Volatility regime buckets from realized vol (v0.4).

Rule-based first; optional HMM can be added behind a flag later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, *, window: int = 20) -> pd.Series:
    """Rolling std of log returns (causal).

    The first ``window - 1`` rows are ``NaN`` because a std computed from a
    single observation is meaningless (§AUDIT B14). Non-positive closes or NaN
    diffs propagate through as ``NaN`` rather than ``-inf`` from ``log(0)``.
    """
    close_f = close.astype(float)
    # ``np.log`` silently returns ``-inf`` for a zero close and ``NaN`` for a
    # negative one; neither is a legal input to ``GaussianHMM.fit`` downstream
    # (§AUDIT B14). Coerce non-positive closes to NaN so the diff/rolling chain
    # produces NaN rather than ``-inf``.
    safe_close = close_f.where(close_f > 0.0)
    lr = np.log(safe_close).diff()
    # Require at least half the window of observations before emitting a std;
    # this keeps early-warmup rows NaN instead of producing one-sample stds.
    min_p = max(2, window // 2)
    return lr.rolling(window, min_periods=min_p).std()


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
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError("add_vol_regime(use_hmm=True) requires the hmmlearn package") from e
        k = int(hmm_states or n_buckets)
        # ``np.log`` emits ``-inf`` for a zero close and ``NaN`` for a negative
        # one; ``GaussianHMM.fit`` errors opaquely on either. Mask non-positive
        # closes before diff-logging and fill the resulting NaNs with 0.0 so
        # the input matrix is always finite (§AUDIT B14).
        close_f = out[close_col].astype(float)
        safe_close = close_f.where(close_f > 0.0)
        lr_s = np.log(safe_close).diff()
        lr = lr_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy().reshape(-1, 1)
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
        out.attrs["vol_regime_buckets_actual"] = 0
        return out
    q = pd.qcut(vol.loc[valid], q=n_buckets, labels=False, duplicates="drop")
    regimes.loc[valid] = q.astype(float)
    out[out_col] = regimes
    # ``pd.qcut(..., duplicates="drop")`` silently collapses buckets when the
    # realized-vol distribution has too few distinct quantile edges. Expose the
    # *actual* bucket count via ``df.attrs`` so downstream routing code
    # (``bundles[str(i)] for i in range(n_buckets)``) can detect a mismatch and
    # degrade gracefully instead of raising ``KeyError`` (§AUDIT B14).
    actual_buckets = int(q.nunique(dropna=True))
    out.attrs["vol_regime_buckets_actual"] = actual_buckets
    if actual_buckets < n_buckets:
        import logging

        logging.getLogger(__name__).warning(
            "add_vol_regime: requested %d buckets but qcut emitted %d "
            "(realized-vol distribution has insufficient distinct quantile edges); "
            "downstream per-regime routing must use df.attrs['vol_regime_buckets_actual'].",
            n_buckets,
            actual_buckets,
        )
    return out
