"""Feature importance table shape and rank monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from aprilalgo.ml.importance import (
    permutation_importance_table,
    xgb_importance_table,
)


def test_xgb_importance_ranks_follow_scores():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] > X["b"]).astype(np.int64)
    clf = XGBClassifier(n_estimators=20, max_depth=2, random_state=0)
    clf.fit(X, y)
    tab = xgb_importance_table(clf, feature_names=list(X.columns))
    assert list(tab.columns)[:4] == ["feature", "method", "score", "rank"]
    scores = tab["score"].to_numpy()
    ranks = tab["rank"].to_numpy()
    assert (np.diff(scores) <= 0).all() or len(tab) <= 1
    assert list(ranks) == list(range(1, len(tab) + 1))


def test_permutation_importance_ranks_follow_scores():
    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = (X["f1"] + X["f2"] > 0).astype(np.int64)
    clf = XGBClassifier(n_estimators=15, max_depth=2, random_state=1)
    clf.fit(X, y)
    tab = permutation_importance_table(clf, X, y, n_repeats=5, random_state=0)
    assert "method" in tab.columns and tab["method"].iloc[0] == "permutation"
    scores = tab["score"].to_numpy()
    assert (np.diff(scores) <= 0).all() or len(tab) <= 1
    assert list(tab["rank"]) == list(range(1, len(tab) + 1))
