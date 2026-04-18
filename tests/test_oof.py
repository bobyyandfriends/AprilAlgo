"""Tests for purged out-of-fold primary predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.ml.oof import compute_primary_oof


def _binary_xyt(n: int = 120) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64).to_numpy()
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 4
    return X, y, t0, t1


def _factory_binary(seed: int = 0):
    def factory():
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            n_estimators=12,
            max_depth=2,
            learning_rate=0.2,
        )

    return factory


def test_oof_shape() -> None:
    X, y, t0, t1 = _binary_xyt(120)
    out = compute_primary_oof(
        X,
        y,
        t0,
        t1,
        factory=_factory_binary(0),
        n_splits=3,
        embargo=1,
        task="binary",
    )
    assert list(out.columns[:3]) == ["row_idx", "y", "oof_pred"]
    proba_cols = [c for c in out.columns if c.startswith("oof_proba_")]
    assert len(proba_cols) == 2
    assert len(out) == len(X)


def test_oof_no_all_nan_on_sufficient_data() -> None:
    X, y, t0, t1 = _binary_xyt(150)
    out = compute_primary_oof(
        X,
        y,
        t0,
        t1,
        factory=_factory_binary(1),
        n_splits=3,
        embargo=0,
        task="binary",
    )
    assert np.isfinite(out["oof_pred"]).any()
    proba_cols = [c for c in out.columns if c.startswith("oof_proba_")]
    assert all(np.isfinite(out[c]).any() for c in proba_cols)


def test_oof_deterministic_with_seed() -> None:
    X, y, t0, t1 = _binary_xyt(130)
    a = compute_primary_oof(
        X,
        y,
        t0,
        t1,
        factory=_factory_binary(42),
        n_splits=3,
        embargo=1,
        task="binary",
    )
    b = compute_primary_oof(
        X,
        y,
        t0,
        t1,
        factory=_factory_binary(42),
        n_splits=3,
        embargo=1,
        task="binary",
    )
    pd.testing.assert_frame_equal(a, b)


def test_oof_sample_weight_mismatch_raises() -> None:
    X, y, t0, t1 = _binary_xyt(60)
    bad_w = np.ones(len(X) + 3)
    with pytest.raises(ValueError, match="sample_weight"):
        compute_primary_oof(
            X,
            y,
            t0,
            t1,
            factory=_factory_binary(0),
            n_splits=2,
            embargo=0,
            task="binary",
            sample_weight=bad_w,
        )
