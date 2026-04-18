"""Purged cross-validation evaluation metrics (v0.3).

Uses :class:`aprilalgo.ml.cv.PurgedKFold` with per-row ``t0`` / ``t1``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from aprilalgo.ml.cv import PurgedKFold


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def purged_cv_evaluate(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    sample_t0: np.ndarray | pd.Series,
    sample_t1: np.ndarray | pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
) -> dict[str, Any]:
    """Run purged k-fold; fit a fresh estimator per train fold.

    Returns
    -------
    dict
        ``folds`` — list of per-fold metric dicts;
        ``mean`` — column means for numeric metrics;
        ``n_splits`` — requested splits.
    """
    y_arr = np.asarray(y)
    t0 = np.asarray(sample_t0, dtype=np.int64)
    t1 = np.asarray(sample_t1, dtype=np.int64)
    pkf = PurgedKFold(n_splits=n_splits, embargo=embargo)
    folds: list[dict[str, Any]] = []

    for train_idx, test_idx in pkf.split(
        X, y=y_arr, sample_t0=t0, sample_t1=t1
    ):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y_arr[train_idx]
        y_te = y_arr[test_idx]

        est = estimator_factory()
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))

        fold_metrics: dict[str, Any] = {
            "train_size": int(train_idx.size),
            "test_size": int(test_idx.size),
            "accuracy": float(accuracy_score(y_te, pred)),
        }
        if n_classes > 2:
            fold_metrics["f1_macro"] = float(
                f1_score(y_te, pred, average="macro", zero_division=0)
            )
            fold_metrics["f1_weighted"] = float(
                f1_score(y_te, pred, average="weighted", zero_division=0)
            )
            try:
                proba = est.predict_proba(X_te)
                fold_metrics["log_loss"] = float(
                    log_loss(y_te, proba, labels=sorted(np.unique(y_arr)))
                )
            except Exception:
                fold_metrics["log_loss"] = None
        else:
            fold_metrics["f1"] = float(
                f1_score(y_te, pred, average="binary", zero_division=0)
            )
            try:
                proba = est.predict_proba(X_te)[:, 1]
                fold_metrics["roc_auc"] = _safe_roc_auc(y_te, proba)
                fold_metrics["log_loss"] = float(log_loss(y_te, proba))
            except Exception:
                fold_metrics["roc_auc"] = None
                fold_metrics["log_loss"] = None

        fold_metrics["confusion_matrix"] = confusion_matrix(y_te, pred).tolist()
        folds.append(fold_metrics)

    if not folds:
        return {
            "folds": [],
            "folds_df": pd.DataFrame(),
            "mean": {},
            "n_splits": n_splits,
        }

    # aggregate means for numeric scalar keys
    keys = [
        k
        for k in folds[0]
        if k != "confusion_matrix" and isinstance(folds[0][k], (int, float))
    ]
    mean: dict[str, float] = {}
    for k in keys:
        vals = [float(f[k]) for f in folds if f.get(k) is not None]
        if vals:
            mean[k] = float(np.mean(vals))
    folds_df = pd.DataFrame(
        [{k: v for k, v in f.items() if k != "confusion_matrix"} for f in folds]
    )
    return {
        "folds": folds,
        "folds_df": folds_df,
        "mean": mean,
        "n_splits": n_splits,
    }


def fold_train_test_interval_disjoint(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
) -> bool:
    """Return True iff no training row's ``[t0,t1]`` overlaps any test row's interval."""
    for j in test_idx:
        for i in train_idx:
            if max(int(t0[i]), int(t0[j])) <= min(int(t1[i]), int(t1[j])):
                return False
    return True
