"""Purged cross-validation evaluation metrics (v0.3).

Uses :class:`aprilalgo.ml.cv.PurgedKFold` with per-row ``t0`` / ``t1``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    sample_weight: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Run purged k-fold; fit a fresh estimator per train fold.

    Parameters
    ----------
    sample_weight
        Optional per-row training weights (uniqueness or sequential-bootstrap
        weights). When provided, each fold's training call receives the
        corresponding slice via ``est.fit(..., sample_weight=sw_tr)`` — this
        keeps the tuner's hyperparameter selection aligned with the production
        weighting scheme.

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
    sw_arr: np.ndarray | None
    if sample_weight is None:
        sw_arr = None
    else:
        sw_arr = np.asarray(sample_weight, dtype=np.float64)
        if sw_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"sample_weight length {sw_arr.shape[0]} does not match y length {y_arr.shape[0]}")
    uniq_all = np.unique(y_arr)
    # sklearn warns if ``labels`` has length 1; for degenerate binary data use a 0/1 axis.
    if len(uniq_all) == 1 and float(uniq_all[0]) in (0.0, 1.0):
        label_universe = list(np.array([0, 1], dtype=y_arr.dtype))
    else:
        label_universe = sorted(uniq_all.tolist())
    pkf = PurgedKFold(n_splits=n_splits, embargo=embargo)
    folds: list[dict[str, Any]] = []

    for train_idx, test_idx in pkf.split(X, y=y_arr, sample_t0=t0, sample_t1=t1):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y_arr[train_idx]
        y_te = y_arr[test_idx]

        est = estimator_factory()
        if sw_arr is not None:
            est.fit(X_tr, y_tr, sample_weight=sw_arr[train_idx])
        else:
            est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
        y_te_uniq = np.unique(y_te)

        fold_metrics: dict[str, Any] = {
            "train_size": int(train_idx.size),
            "test_size": int(test_idx.size),
            "accuracy": float(accuracy_score(y_te, pred)),
        }
        if n_classes > 2:
            fold_metrics["f1_macro"] = float(f1_score(y_te, pred, average="macro", zero_division=0))
            fold_metrics["f1_weighted"] = float(f1_score(y_te, pred, average="weighted", zero_division=0))
            try:
                proba = est.predict_proba(X_te)
                if len(y_te_uniq) >= 2:
                    fold_metrics["log_loss"] = float(log_loss(y_te, proba, labels=label_universe))
                else:
                    fold_metrics["log_loss"] = None
            except Exception:
                fold_metrics["log_loss"] = None
        else:
            fold_metrics["f1"] = float(f1_score(y_te, pred, average="binary", zero_division=0))
            # If the training fold saw only one class, ``est.classes_`` has length 1
            # and ``predict_proba(...)[:, 1]`` would raise IndexError. Skip proba-based
            # metrics explicitly so we don't swallow unrelated errors via bare except.
            train_classes = np.unique(y_tr)
            if len(train_classes) < 2 or not hasattr(est, "predict_proba") or len(y_te_uniq) < 2:
                fold_metrics["roc_auc"] = None
                fold_metrics["log_loss"] = None
            else:
                try:
                    proba_full = est.predict_proba(X_te)
                    # Binary classifiers always emit a 2-column proba matrix when
                    # trained on 2+ classes, so indexing [:, 1] is safe here.
                    proba = proba_full[:, 1]
                    fold_metrics["roc_auc"] = _safe_roc_auc(y_te, proba)
                    fold_metrics["log_loss"] = float(log_loss(y_te, proba, labels=label_universe))
                except (ValueError, IndexError):
                    fold_metrics["roc_auc"] = None
                    fold_metrics["log_loss"] = None

        fold_metrics["confusion_matrix"] = confusion_matrix(y_te, pred, labels=label_universe).tolist()
        folds.append(fold_metrics)

    if not folds:
        return {
            "folds": [],
            "folds_df": pd.DataFrame(),
            "mean": {},
            "n_splits": n_splits,
        }

    # aggregate means for numeric scalar keys
    keys = [k for k in folds[0] if k != "confusion_matrix" and isinstance(folds[0][k], (int, float))]
    mean: dict[str, float] = {}
    for k in keys:
        vals = [float(f[k]) for f in folds if f.get(k) is not None]
        if vals:
            mean[k] = float(np.mean(vals))
    folds_df = pd.DataFrame([{k: v for k, v in f.items() if k != "confusion_matrix"} for f in folds])
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
    """Return True iff no training row's ``[t0,t1]`` overlaps any test row's interval.

    Vectorised via disjoint-run merge on the test side; the previous implementation
    was O(|train| · |test|) pure-Python, so a 10k-sample fold ran ~16M iterations.
    """
    if train_idx.size == 0 or test_idx.size == 0:
        return True
    te_t0 = t0[test_idx]
    te_t1 = t1[test_idx]
    order = np.argsort(te_t0, kind="stable")
    te_t0 = te_t0[order]
    te_t1 = te_t1[order]
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
    tr_t0 = t0[train_idx]
    tr_t1 = t1[train_idx]
    pos = np.searchsorted(runs_start, tr_t1, side="right") - 1
    valid = pos >= 0
    if not valid.any():
        return True
    picked_end = runs_end[np.maximum(pos, 0)]
    picked_start = runs_start[np.maximum(pos, 0)]
    lo = np.maximum(picked_start, tr_t0)
    hi = np.minimum(picked_end, tr_t1)
    return not bool((valid & (lo <= hi)).any())
