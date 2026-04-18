"""Meta-labels: whether the primary model's prediction matches *y* (v0.4).

Secondary models consume out-of-fold primary predictions plus features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aprilalgo.ml.cv import PurgedKFold


def build_meta_labels(y_true: np.ndarray, oof_primary_pred: np.ndarray) -> np.ndarray:
    """``1.0`` if primary prediction equals *y_true*, ``0.0`` otherwise, ``NaN``
    when the primary prediction is NaN (e.g. a PurgedKFold fold was skipped after
    purging emptied its training block).

    Callers must mask the NaN rows before fitting; otherwise NaN predictions
    would silently be treated as "primary incorrect" and bias the meta dataset
    toward the positive class on every undefined bar.
    """
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(oof_primary_pred)
    out = np.full(p_arr.shape[0], np.nan, dtype=np.float64)
    valid = ~pd.isna(p_arr)
    out[valid] = (p_arr[valid] == y_arr[valid]).astype(np.float64)
    return out


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
    z_full = build_meta_labels(y_true, oof_primary_pred)
    # NaN marks rows where the primary OOF pred is undefined; those rows
    # cannot feed the meta fit. Restrict to the valid mask.
    valid_mask = ~np.isnan(z_full)
    z = z_full[valid_mask].astype(np.int64)
    if len(np.unique(z)) < 2:
        raise ValueError(
            "Meta labels z = (oof_pred == y) have a single class; need both correct and "
            "incorrect primary OOF rows. Regenerate oof_primary.csv vs current y, or "
            "check that the primary model is not trivially identical to y on all rows."
        )

    X_valid = X.iloc[valid_mask].reset_index(drop=True)
    pred_valid = oof_primary_pred[valid_mask].reshape(-1, 1)
    stack = np.column_stack([X_valid.to_numpy(dtype=np.float64), pred_valid])
    names = list(X.columns) + ["primary_pred"]
    X_meta = pd.DataFrame(stack, columns=names)

    t0_full = np.asarray(sample_t0, dtype=np.int64)
    t1_full = np.asarray(sample_t1, dtype=np.int64)
    t0 = t0_full[valid_mask]
    t1 = t1_full[valid_mask]
    pkf = PurgedKFold(n_splits=n_splits, embargo=embargo)

    meta_oof_valid = np.full(z.shape[0], np.nan, dtype=np.float64)
    for tr, te in pkf.split(X_meta, y=z, sample_t0=t0, sample_t1=t1):
        if len(np.unique(z[tr])) < 2:
            meta_oof_valid[te] = 0.5
            continue
        m = LogisticRegression(max_iter=200, random_state=0)
        m.fit(X_meta.iloc[tr], z[tr])
        pr = m.predict_proba(X_meta.iloc[te])[:, 1]
        meta_oof_valid[te] = pr

    meta_full = LogisticRegression(max_iter=200, random_state=0)
    meta_full.fit(X_meta, z)

    # Re-expand meta_oof / z to the original length using NaN / -1 for
    # rows that were filtered out of the fit.
    meta_oof = np.full(len(y_true), np.nan, dtype=np.float64)
    meta_oof[valid_mask] = meta_oof_valid
    return meta_full, meta_oof, z_full
