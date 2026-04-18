"""Tests for purged k-fold CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.ml.cv import PurgedKFold, learning_matrix


def _overlap_inclusive(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


class TestIntervalsOverlap:
    def test_touching_intervals_overlap(self):
        assert _overlap_inclusive(0, 2, 2, 4) is True

    def test_disjoint(self):
        assert _overlap_inclusive(0, 1, 2, 3) is False


class TestPurgedKFold:
    def test_requires_sample_t1(self):
        cv = PurgedKFold(n_splits=2)
        with pytest.raises(ValueError, match="sample_t1"):
            next(cv.split(np.zeros(10)))

    def test_rejects_nan_t1(self):
        cv = PurgedKFold(n_splits=2)
        t1 = np.array([2.0] * 8 + [np.nan, np.nan])
        with pytest.raises(ValueError, match="finite"):
            next(cv.split(np.zeros(10), sample_t1=t1))

    def test_purge_removes_overlapping_train(self):
        """Train j must not overlap any test k's [t0,t1]."""
        n = 12
        t0 = np.arange(n, dtype=np.int64)
        t1 = t0 + 2  # each sample uses 3 bars
        X = np.zeros(n)
        cv = PurgedKFold(n_splits=3, embargo=0)
        splits = list(cv.split(X, sample_t1=t1))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            for j in train_idx:
                for k in test_idx:
                    assert not _overlap_inclusive(
                        int(t0[j]), int(t1[j]), int(t0[k]), int(t1[k])
                    ), f"train {j} overlaps test {k}"

    def test_embargo_drops_train_after_test_block(self):
        n = 20
        t0 = np.arange(n, dtype=np.int64)
        t1 = t0.copy()
        X = np.zeros(n)
        cv = PurgedKFold(n_splits=2, embargo=3)
        train_idx, test_idx = next(cv.split(X, sample_t1=t1))
        max_t1_test = int(np.max(t1[test_idx]))
        for j in train_idx:
            t0j = int(t0[j])
            assert not (max_t1_test < t0j <= max_t1_test + 3)

    def test_n_splits_and_fold_count(self):
        X = np.zeros(100)
        t1 = np.arange(100, dtype=np.int64)
        cv = PurgedKFold(n_splits=5)
        assert cv.get_n_splits() == 5
        assert len(list(cv.split(X, sample_t1=t1))) == 5


class TestLearningMatrix:
    def test_produces_finite_t0_t1(self):
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 120, 80),
                "high": np.linspace(101, 121, 80),
                "low": np.linspace(99, 119, 80),
                "close": np.linspace(100, 120, 80),
                "volume": np.ones(80) * 1e6,
            }
        )
        X, y, t0, t1 = learning_matrix(
            df,
            indicator_config=[{"name": "rsi", "period": 14}],
            upper_pct=0.02,
            lower_pct=0.02,
            vertical_bars=5,
        )
        assert len(X) == len(y) == len(t0) == len(t1)
        assert np.isfinite(t0).all() and np.isfinite(t1).all()
        assert (t1 >= t0).all()

    def test_purged_kfold_runs_on_learning_matrix(self):
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 120, 80),
                "high": np.linspace(101, 121, 80),
                "low": np.linspace(99, 119, 80),
                "close": np.linspace(100, 120, 80),
                "volume": np.ones(80) * 1e6,
            }
        )
        X, y, t0, t1 = learning_matrix(
            df,
            indicator_config=[{"name": "rsi", "period": 14}],
            upper_pct=0.02,
            lower_pct=0.02,
            vertical_bars=5,
        )
        cv = PurgedKFold(n_splits=3, embargo=2)
        folds = list(cv.split(X, y, sample_t0=t0, sample_t1=t1))
        assert len(folds) == 3
        for tr, te in folds:
            assert len(np.intersect1d(tr, te)) == 0
