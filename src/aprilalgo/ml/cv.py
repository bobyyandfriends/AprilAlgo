"""Purged k-fold cross-validation (López de Prado).

Removes training samples whose label-formation interval overlaps any test sample's
interval, and optionally drops an embargo window after each test block to limit
serial correlation leakage. See ARCHITECTURE.md §4.6.

Requires per-row ``sample_t0`` / ``sample_t1`` (inclusive integer bar indices, 0..n-1).
For triple-barrier labels, use :func:`aprilalgo.labels.triple_barrier.label_inclusive_end_ix`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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

    Parameters
    ----------
    n_splits
        Number of folds (``>= 2``).
    embargo
        One-sided embargo distance applied *after* each test block (drops training
        rows whose ``t0`` falls in ``(max_t1_test, max_t1_test + embargo]``).
    symmetric_embargo
        When ``True`` the embargo is also applied *before* the test block, dropping
        training rows whose ``t1`` falls in ``[min_t0_test - embargo, min_t0_test)``.
        This matches AFML §7's symmetric embargo prescription (covers feature-level
        serial correlation that leaks *into* the test block). Defaults to ``False``
        for backward compatibility — existing tuner results remain reproducible.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 0,
        *,
        symmetric_embargo: bool = False,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits = n_splits
        self.embargo = embargo
        self.symmetric_embargo = symmetric_embargo

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
            raise ValueError("sample_t1 is required for purged k-fold (inclusive end index per row)")

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
            train_idx = _embargo_train(
                train_idx,
                test_idx,
                t0_i,
                t1_i,
                self.embargo,
                symmetric=self.symmetric_embargo,
            )
            yield train_idx, test_idx


def _purge_train(
    train_cand: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
) -> np.ndarray:
    """Drop candidate training rows whose ``[t0,t1]`` overlaps any test row's interval.

    Vectorised O(|test| + |train|) implementation: a training row ``j`` is kept iff
    its window ``[t0[j], t1[j]]`` does **not** intersect the union of the test-row
    intervals, which is itself a union of closed intervals. Rather than build the
    union explicitly (which can be sparse), we note that test rows' ``[t0,t1]`` form
    contiguous runs when sorted; checking overlap against ``[min_t0_test, max_t1_test]``
    handles the common single-run case in one pass, and we fall back to a merge-style
    scan for the (rare) multi-run case.
    """
    if train_cand.size == 0 or test_idx.size == 0:
        return train_cand.astype(np.int64, copy=False)

    te_t0 = t0[test_idx]
    te_t1 = t1[test_idx]
    # Sort test intervals by start so we can merge into disjoint runs.
    order = np.argsort(te_t0, kind="stable")
    te_t0 = te_t0[order]
    te_t1 = te_t1[order]

    # Merge overlapping/adjacent test intervals into disjoint runs.
    # After this loop, ``runs`` holds the disjoint closed intervals covered by test.
    run_starts: list[int] = [int(te_t0[0])]
    run_ends: list[int] = [int(te_t1[0])]
    for s, e in zip(te_t0[1:], te_t1[1:], strict=True):
        s_i = int(s)
        e_i = int(e)
        if s_i <= run_ends[-1]:
            if e_i > run_ends[-1]:
                run_ends[-1] = e_i
        else:
            run_starts.append(s_i)
            run_ends.append(e_i)

    runs_start = np.asarray(run_starts, dtype=np.int64)
    runs_end = np.asarray(run_ends, dtype=np.int64)

    tr_t0 = t0[train_cand]
    tr_t1 = t1[train_cand]

    # For each training row, find the first run whose start > training row's t1 via
    # binary search. The previous run (if any) is the only candidate that could
    # overlap; check it directly. This is O(|train| log |runs|) which is effectively
    # linear when |runs| is small (the common single-block test case).
    # overlap iff some run has max(runs_start[k], tr_t0) <= min(runs_end[k], tr_t1).
    pos = np.searchsorted(runs_start, tr_t1, side="right") - 1
    overlap = np.zeros(train_cand.size, dtype=bool)
    valid = pos >= 0
    if valid.any():
        picked_end = runs_end[np.maximum(pos, 0)]
        picked_start = runs_start[np.maximum(pos, 0)]
        lo = np.maximum(picked_start, tr_t0)
        hi = np.minimum(picked_end, tr_t1)
        overlap = valid & (lo <= hi)

    keep_mask = ~overlap
    return train_cand[keep_mask].astype(np.int64, copy=False)


def _embargo_train(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    embargo: int,
    *,
    symmetric: bool = False,
) -> np.ndarray:
    """Drop training rows adjacent to the test block's time boundary.

    Default (asymmetric) behaviour drops training rows whose ``t0`` lies in
    ``(max_t1_test, max_t1_test + embargo]``. With ``symmetric=True`` we also drop
    rows whose ``t1`` lies in ``[min_t0_test - embargo, min_t0_test)`` — this
    matches AFML §7's symmetric embargo and covers feature-level serial
    correlation leaking forward *into* the test block as well as backward out of it.
    """
    if embargo == 0 or train_idx.size == 0:
        return train_idx
    max_t1_test = int(np.max(t1[test_idx]))
    t0_tr = t0[train_idx]
    after_mask = (t0_tr > max_t1_test) & (t0_tr <= max_t1_test + embargo)
    keep_mask = ~after_mask
    if symmetric:
        min_t0_test = int(np.min(t0[test_idx]))
        t1_tr = t1[train_idx]
        before_mask = (t1_tr >= min_t0_test - embargo) & (t1_tr < min_t0_test)
        keep_mask &= ~before_mask
    return train_idx[keep_mask].astype(np.int64, copy=False)


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
    mask_arr = y.notna().to_numpy() & np.isfinite(t1_full) & X_raw.notna().all(axis=1).to_numpy()
    X = X_raw.loc[mask_arr].reset_index(drop=True)
    y = y.loc[mask_arr].reset_index(drop=True)
    t0 = t0_full[mask_arr]
    t1 = t1_full[mask_arr].astype(np.int64)
    return X, y, t0, t1
