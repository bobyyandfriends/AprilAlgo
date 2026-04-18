"""Sample-weight behavior for :func:`~aprilalgo.ml.trainer.train_xgb_classifier`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aprilalgo.ml.trainer import train_xgb_classifier


def test_uniform_sample_weight_matches_no_weight() -> None:
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((X["a"] + 0.1 * X["b"]) > 0).astype(np.int64)
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}
    clf0 = train_xgb_classifier(
        X, y, task="binary", random_state=42, xgb_params=params
    )
    w = np.ones(n, dtype=np.float64)
    clf1 = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=42,
        xgb_params=params,
        sample_weight=w,
    )
    np.testing.assert_allclose(
        clf0.feature_importances_, clf1.feature_importances_, rtol=1e-5, atol=1e-5
    )
    p0 = clf0.predict_proba(X)
    p1 = clf1.predict_proba(X)
    np.testing.assert_allclose(p0, p1, rtol=1e-4, atol=1e-4)


def test_skewed_sample_weight_changes_feature_importance() -> None:
    rng = np.random.default_rng(2)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = ((X["f1"] + 0.3 * X["f2"]) > 0).astype(np.int64)
    params = {"n_estimators": 40, "max_depth": 4, "learning_rate": 0.1}
    cu = train_xgb_classifier(
        X, y, task="binary", random_state=0, xgb_params=params
    )
    w = np.ones(n, dtype=np.float64)
    w[n // 2 :] = 0.01
    cw = train_xgb_classifier(
        X, y, task="binary", random_state=0, xgb_params=params, sample_weight=w
    )
    assert not np.allclose(
        cu.feature_importances_, cw.feature_importances_, atol=1e-3
    )
