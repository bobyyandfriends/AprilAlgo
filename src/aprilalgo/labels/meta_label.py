"""Meta-labels: whether the primary model's prediction matches *y* (v0.4).

Secondary models consume out-of-fold primary predictions plus features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aprilalgo.ml.cv import PurgedKFold


def build_meta_labels(y_true: np.ndarray, oof_primary_pred: np.ndarray) -> np.ndarray:
    """``1`` if primary prediction equals *y_true*, else ``0``."""
    return (np.asarray(oof_primary_pred) == np.asarray(y_true)).astype(np.int64)


def fit_meta_logit_purged(
    X: pd.DataFrame,
    y_true: np.ndarray,
    oof_primary_pred: np.ndarray,
    *,
    sample_t0: np.ndarray,
    sample_t1: np.ndarray,
    n_splits: int = 3,
    embargo: int = 0,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Train meta logistic on ``[X | oof_primary_pred]`` to predict correctness.

    Returns ``(meta_fitted_on_all, meta_oof_proba_correct, z_all)`` where
    ``meta_oof_proba_correct`` is OOF probability of label ``1`` from purged CV,
    and ``z_all = build_meta_labels(y_true, oof_primary_pred)``.
    """
    y_true = np.asarray(y_true)
    oof_primary_pred = np.asarray(oof_primary_pred)
    z = build_meta_labels(y_true, oof_primary_pred)

    stack = np.column_stack([X.to_numpy(dtype=np.float64), oof_primary_pred.reshape(-1, 1)])
    names = list(X.columns) + ["primary_pred"]
    X_meta = pd.DataFrame(stack, columns=names)

    t0 = np.asarray(sample_t0, dtype=np.int64)
    t1 = np.asarray(sample_t1, dtype=np.int64)
    pkf = PurgedKFold(n_splits=n_splits, embargo=embargo)

    meta_oof = np.full(len(y_true), np.nan, dtype=np.float64)
    for tr, te in pkf.split(X_meta, y=z, sample_t0=t0, sample_t1=t1):
        m = LogisticRegression(max_iter=200, random_state=0)
        m.fit(X_meta.iloc[tr], z[tr])
        pr = m.predict_proba(X_meta.iloc[te])[:, 1]
        meta_oof[te] = pr

    meta_full = LogisticRegression(max_iter=200, random_state=0)
    meta_full.fit(X_meta, z)
    return meta_full, meta_oof, z
