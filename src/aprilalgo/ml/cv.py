"""Purged k-fold cross-validation (López de Prado).

Removes training samples whose label-formation interval overlaps any test sample's
interval, and optionally drops an embargo window after each test block to limit
serial correlation leakage. See ARCHITECTURE.md §4.6.

Requires per-row ``sample_t0`` / ``sample_t1`` (inclusive integer bar indices, 0..n-1).
For triple-barrier labels, use :func:`aprilalgo.labels.triple_barrier.label_inclusive_end_ix`.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd


def _intervals_overlap(t0_a: float, t1_a: float, t0_b: float, t1_b: float) -> bool:
    """Inclusive integer intervals [t0_a, t1_a] and [t0_b, t1_b]."""
    return max(t0_a, t0_b) <= min(t1_a, t1_b)


class PurgedKFold:
    """Chronological k-fold with purging and optional embargo.

    Each fold uses one contiguous block of indices as the test set. Training
    indices are all remaining samples that (1) do not overlap any test sample's
    ``[t0, t1]`` label window, and (2) do not start in the embargo zone after the
    test block's latest ``t1``.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        *,
        sample_t0: np.ndarray | pd.Series | None = None,
        sample_t1: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``train_indices``, ``test_indices`` (integer positions).

        Parameters
        ----------
        X
            Length ``n`` — any sequence; only ``len(X)`` is used unless *y* is
            given with different length (then *y*'s length wins for alignment checks).
        sample_t0
            Start index of each sample's feature time (default: ``0..n-1``).
        sample_t1
            End index (inclusive) of each sample's label-formation window — required.
        """
        n = len(X)
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if sample_t1 is None:
            raise ValueError(
                "sample_t1 is required for purged k-fold (inclusive end index per row)"
            )

        t0 = np.arange(n, dtype=np.int64) if sample_t0 is None else np.asarray(sample_t0)
        t1 = np.asarray(sample_t1, dtype=np.float64)

        if t0.shape[0] != n or t1.shape[0] != n:
            raise ValueError("sample_t0 and sample_t1 must have length len(X)")

        if np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t0)):
            raise ValueError("sample_t0 and sample_t1 must be finite (drop NaN rows first)")

        t0_i = t0.astype(np.int64, copy=False)
        t1_i = t1.astype(np.int64, copy=False)

        if np.any(t1_i < t0_i):
            raise ValueError("sample_t1 must be >= sample_t0 for every row")

        folds = np.array_split(np.arange(n, dtype=np.int64), self.n_splits)

        for test_idx in folds:
            if test_idx.size == 0:
                continue
            train_cand = np.setdiff1d(np.arange(n, dtype=np.int64), test_idx)
            train_idx = _purge_train(train_cand, test_idx, t0_i, t1_i)
            train_idx = _embargo_train(train_idx, test_idx, t0_i, t1_i, self.embargo)
            yield train_idx, test_idx


def _purge_train(
    train_cand: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
) -> np.ndarray:
    keep: list[int] = []
    for j in train_cand:
        purge = False
        for k in test_idx:
            if _intervals_overlap(int(t0[j]), int(t1[j]), int(t0[k]), int(t1[k])):
                purge = True
                break
        if not purge:
            keep.append(int(j))
    return np.array(keep, dtype=np.int64)


def _embargo_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    embargo: int,
) -> np.ndarray:
    if embargo == 0 or train_idx.size == 0:
        return train_idx
    max_t1_test = int(np.max(t1[test_idx]))
    keep: list[int] = []
    for j in train_idx:
        t0j = int(t0[j])
        # Drop training starts in (max_t1_test, max_t1_test + embargo]
        if max_t1_test < t0j <= max_t1_test + embargo:
            continue
        keep.append(int(j))
    return np.array(keep, dtype=np.int64)


def learning_matrix(
    ohlcv: pd.DataFrame,
    *,
    indicator_config: list[dict[str, Any]],
    upper_pct: float,
    lower_pct: float,
    vertical_bars: int,
    extra_exclude: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Build ``X``, ``y``, ``sample_t0``, ``sample_t1`` for :class:`PurgedKFold`.

    Uses a positional ``0..n-1`` index on a copy of *ohlcv* so ``t0``/``t1`` match
    rows of *X* after dropping NaNs. ``t0[i]`` is the bar index of sample *i*;
    ``t1[i]`` is the inclusive end bar index for that sample's triple-barrier label.
    """
    from aprilalgo.labels.triple_barrier import apply_triple_barrier, label_inclusive_end_ix
    from aprilalgo.ml.features import build_feature_matrix

    df = ohlcv.reset_index(drop=True)
    n = len(df)
    tb = apply_triple_barrier(
        df,
        upper_pct=upper_pct,
        lower_pct=lower_pct,
        vertical_bars=vertical_bars,
    )
    t0_full = np.arange(n, dtype=np.int64)
    t1_full = label_inclusive_end_ix(tb.barrier_hit_offset).to_numpy(dtype=np.float64)

    extra: dict[str, Any] = {}
    if extra_exclude is not None:
        extra["extra_exclude"] = extra_exclude

    X_raw = build_feature_matrix(df, indicator_config=indicator_config, **extra)
    y = tb.label
    mask_arr = (
        y.notna().to_numpy()
        & np.isfinite(t1_full)
        & X_raw.notna().all(axis=1).to_numpy()
    )
    X = X_raw.loc[mask_arr].reset_index(drop=True)
    y = y.loc[mask_arr].reset_index(drop=True)
    t0 = t0_full[mask_arr]
    t1 = t1_full[mask_arr].astype(np.int64)
    return X, y, t0, t1
