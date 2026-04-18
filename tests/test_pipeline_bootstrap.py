"""Tests for ml.pipeline weights_for_training bootstrap branches."""

from __future__ import annotations

import numpy as np

from aprilalgo.ml.pipeline import weights_for_training


def test_weights_bootstrap_n_draw_zero_returns_ones() -> None:
    t0 = np.arange(5, dtype=np.int64)
    t1 = t0 + 2
    cfg = {"sampling": {"strategy": "bootstrap", "n_draw": 0, "random_state": 0}}
    w = weights_for_training(cfg, t0, t1)
    assert w is not None
    assert w.shape == (5,)
    assert np.allclose(w, 1.0)


def test_weights_bootstrap_renormalizes_to_length() -> None:
    t0 = np.array([0, 10, 20], dtype=np.int64)
    t1 = np.array([5, 15, 25], dtype=np.int64)
    cfg = {"sampling": {"strategy": "bootstrap", "n_draw": 30, "random_state": 42}, "random_state": 42}
    w = weights_for_training(cfg, t0, t1)
    assert w is not None
    assert w.shape == (3,)
    assert abs(float(w.sum()) - 3.0) < 1e-6
    assert (w >= 0).all()
