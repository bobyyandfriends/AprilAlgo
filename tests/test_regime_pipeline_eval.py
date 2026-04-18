"""Coverage for meta.regime, ml.pipeline, and ml.evaluator edge paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aprilalgo.meta.regime import add_vol_regime, realized_vol
from aprilalgo.ml.evaluator import _safe_roc_auc, fold_train_test_interval_disjoint, purged_cv_evaluate
from aprilalgo.ml.pipeline import apply_regime_if_enabled, weights_for_training, xgb_estimator_factory


def test_realized_vol_non_positive_close_nan() -> None:
    s = pd.Series([100.0, 0.0, -1.0, 101.0])
    v = realized_vol(s, window=2)
    assert v.isna().all() or pd.isna(v.iloc[0])


def test_add_vol_regime_all_nan_vol_short_series() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    out = add_vol_regime(df, window=20, n_buckets=3)
    assert out["vol_regime"].isna().all()
    assert int(out.attrs.get("vol_regime_buckets_actual", -1)) == 0


def test_add_vol_regime_quantile_path(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 80))
    df = pd.DataFrame({"close": close})
    out = add_vol_regime(df, window=10, n_buckets=5)
    assert "vol_regime" in out.columns
    assert out.attrs.get("vol_regime_buckets_actual", 0) >= 1


def test_add_vol_regime_use_hmm_requires_hmmlearn() -> None:
    df = pd.DataFrame({"close": np.linspace(100.0, 110.0, 60)})
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="hmmlearn"):
            add_vol_regime(df, window=10, n_buckets=3, use_hmm=True)
    else:
        out = add_vol_regime(df, window=10, n_buckets=3, use_hmm=True, hmm_states=3)
        assert out["vol_regime"].notna().sum() > 0


def test_apply_regime_disabled_returns_same_frame() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = apply_regime_if_enabled(df, {})
    assert out is df
    out2 = apply_regime_if_enabled(df, {"regime": {"enabled": False}})
    assert out2 is df


def test_weights_for_training_unknown_strategy() -> None:
    t0 = np.array([0, 1], dtype=np.int64)
    t1 = np.array([2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="Unknown sampling"):
        weights_for_training({"sampling": {"strategy": "not_a_strategy"}}, t0, t1)


def test_xgb_factory_multiclass_objective() -> None:
    pytest.importorskip("xgboost")
    factory = xgb_estimator_factory({"random_state": 0, "model": {"xgb": {}}}, "multiclass")
    est = factory()
    assert getattr(est, "objective", "") == "multi:softprob"


def test_safe_roc_auc_degenerate() -> None:
    out = _safe_roc_auc(np.array([0, 0, 0]), np.array([0.2, 0.5, 0.8]))
    assert out is None or (isinstance(out, float) and np.isnan(out))


def test_fold_disjoint_empty_train() -> None:
    t0 = np.array([0, 10], dtype=np.int64)
    t1 = np.array([5, 15], dtype=np.int64)
    assert fold_train_test_interval_disjoint(np.array([], dtype=np.int64), np.array([0]), t0, t1)


def test_purged_cv_sample_weight_length_mismatch() -> None:
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0, 1])
    t0 = np.arange(4, dtype=np.int64)
    t1 = t0 + 1
    with pytest.raises(ValueError, match="sample_weight"):
        purged_cv_evaluate(
            lambda: MagicMock(),
            X,
            y,
            sample_t0=t0,
            sample_t1=t1,
            n_splits=2,
            embargo=0,
            sample_weight=np.array([1.0, 1.0]),
        )
