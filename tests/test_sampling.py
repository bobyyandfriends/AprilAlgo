"""Tests for overlap / uniqueness weights."""

from __future__ import annotations

import numpy as np

from aprilalgo.ml.sampling import (
    overlap_count_matrix,
    sequential_bootstrap_sample,
    uniqueness_weights,
)


def test_non_overlapping_intervals_diagonal_ones():
    t0 = np.arange(5, dtype=np.int64)
    t1 = t0.copy()
    m = overlap_count_matrix(t0, t1)
    assert (np.diag(m) == 1).all()
    assert m.sum() == 5


def test_sequential_bootstrap_nearly_uniform_when_disjoint():
    t0 = np.arange(12, dtype=np.int64)
    t1 = t0.copy()
    idx = sequential_bootstrap_sample(t0, t1, n_draw=6000, random_state=2)
    cnt = np.bincount(idx, minlength=12)
    assert cnt.min() > 350
    assert cnt.max() < 650


def test_uniqueness_weights_sum_to_n():
    t0 = np.array([0, 1, 1], dtype=np.int64)
    t1 = np.array([2, 3, 4], dtype=np.int64)
    w = uniqueness_weights(t0, t1)
    assert abs(w.sum() - len(t0)) < 1e-6
    assert (w > 0).all()


def test_uniqueness_weights_sum_to_n_on_overlap():
    """Partially overlapping [t0,t1] intervals still normalize to sum ``n``."""
    t0 = np.array([0, 2, 4], dtype=np.int64)
    t1 = np.array([5, 6, 8], dtype=np.int64)
    w = uniqueness_weights(t0, t1)
    assert abs(w.sum() - len(t0)) < 1e-6
    assert (w > 0).all()


def test_bootstrap_draw_reproducible_with_seed():
    t0 = np.array([0, 1, 2, 3], dtype=np.int64)
    t1 = np.array([2, 3, 4, 5], dtype=np.int64)
    a = sequential_bootstrap_sample(t0, t1, n_draw=200, random_state=7)
    b = sequential_bootstrap_sample(t0, t1, n_draw=200, random_state=7)
    np.testing.assert_array_equal(a, b)
