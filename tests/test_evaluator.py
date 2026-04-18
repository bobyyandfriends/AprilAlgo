"""Tests for purged CV evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from aprilalgo.ml.cv import PurgedKFold
from aprilalgo.ml.evaluator import (
    fold_train_test_interval_disjoint,
    purged_cv_evaluate,
)


def test_purged_cv_runs_and_returns_folds():
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2

    def factory() -> XGBClassifier:
        return XGBClassifier(
            n_estimators=10,
            max_depth=2,
            random_state=0,
        )

    out = purged_cv_evaluate(
        factory,
        X,
        y,
        sample_t0=t0,
        sample_t1=t1,
        n_splits=3,
        embargo=1,
    )
    assert "folds" in out and "mean" in out
    assert len(out["folds"]) == 3
    assert "accuracy" in out["mean"]
    assert "folds_df" in out
    assert len(out["folds_df"]) == 3


def test_purged_folds_train_test_label_intervals_disjoint():
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame({"f1": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3
    pkf = PurgedKFold(n_splits=4, embargo=2)
    for tr, te in pkf.split(X, y=y, sample_t0=t0, sample_t1=t1):
        assert fold_train_test_interval_disjoint(tr, te, t0, t1)
