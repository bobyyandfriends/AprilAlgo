"""Out-of-fold (OOF) predictions from purged k-fold (v0.5 Sprint 3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from aprilalgo.ml.cv import PurgedKFold
from aprilalgo.ml.trainer import Task

__all__ = ["compute_primary_oof"]


def compute_primary_oof(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    *,
    factory: Callable[[], Any],
    n_splits: int,
    embargo: int,
    task: Task,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Stack primary-model predictions from :class:`~aprilalgo.ml.cv.PurgedKFold` test folds.

    Each row appears in exactly one chronological test block; train rows are purged
    against test label intervals. Columns:

    * ``row_idx`` — integer position ``0..n-1`` aligned with the input *X* / *y* rows
    * ``y`` — ground-truth label used for training
    * ``oof_pred`` — ``est.predict`` on the held-out fold (may contain ``NaN`` if a
      fold was skipped due to empty train)
    * ``oof_proba_<c>`` — one column per class ``c`` in ``est.classes_`` order after
      the first successful fit
    """
    _ = task  # reserved for future task-specific handling
    n = len(X)
    y_arr = np.asarray(y)
    t0a = np.asarray(t0, dtype=np.int64)
    t1a = np.asarray(t1, dtype=np.int64)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if sw is not None and sw.shape[0] != n:
        raise ValueError("sample_weight must have length len(X)")

    pkf = PurgedKFold(n_splits=n_splits, embargo=embargo)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    oof_proba: np.ndarray | None = None
    classes_: np.ndarray | None = None

    for train_idx, test_idx in pkf.split(X, y=y_arr, sample_t0=t0a, sample_t1=t1a):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y_arr[train_idx]
        est = factory()
        if sw is not None:
            est.fit(X_tr, y_tr, sample_weight=sw[train_idx])
        else:
            est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        proba = est.predict_proba(X_te)
        if oof_proba is None:
            classes_ = np.asarray(est.classes_, dtype=np.float64).ravel()
            oof_proba = np.full((n, proba.shape[1]), np.nan, dtype=np.float64)
        oof_pred[test_idx] = pred.astype(np.float64, copy=False)
        oof_proba[test_idx] = proba.astype(np.float64, copy=False)

    if oof_proba is None or classes_ is None:
        raise ValueError(
            "No OOF predictions were produced (empty folds or PurgedKFold yielded no splits)"
        )

    out = pd.DataFrame(
        {
            "row_idx": np.arange(n, dtype=np.int64),
            "y": y_arr,
            "oof_pred": oof_pred,
        }
    )
    for j, c in enumerate(classes_):
        out[f"oof_proba_{c}"] = oof_proba[:, j]
    return out
