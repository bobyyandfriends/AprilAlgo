"""Walk-forward split tests."""

from __future__ import annotations

import numpy as np

from aprilalgo.tuner.walk_forward import walk_forward_splits, walk_forward_summary


def test_walk_forward_covers_tail():
    splits = list(walk_forward_splits(100, n_folds=4, min_train=40, test_size=10))
    assert len(splits) >= 1
    _last_train, last_test = splits[-1]
    assert last_test[-1] == 99 or last_test[-1] == 100 - 1


def test_walk_forward_summary_keys():
    splits = list(walk_forward_splits(120, n_folds=4, min_train=40, test_size=20))
    s = walk_forward_summary(120, splits)
    assert {"n_splits", "coverage_pct", "mean_train_size", "mean_test_size"} <= set(
        s.keys()
    )
    assert s["n_splits"] == len(splits)
    assert 0.0 < s["coverage_pct"] <= 100.0
