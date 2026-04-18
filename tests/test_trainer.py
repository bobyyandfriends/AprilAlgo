"""Tests for XGBoost model bundle save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.ml.trainer import (
    load_model_bundle,
    proba_positive_takeprofit,
    save_model_bundle,
    train_xgb_classifier,
)


def test_train_save_load_roundtrip_binary(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n)},
    )
    y = ((X["a"] + X["b"]) > 0).astype(np.int64)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 20, "max_depth": 2},
    )
    save_model_bundle(
        tmp_path,
        clf,
        feature_names=list(X.columns),
        task="binary",
        indicator_config=[{"name": "rsi", "period": 14}],
    )
    bundle = load_model_bundle(tmp_path)
    assert bundle.task == "binary"
    assert bundle.feature_names == ["a", "b"]
    proba = bundle.predict_proba(X.iloc[:3])
    p0 = proba_positive_takeprofit(bundle, proba[0])
    assert 0.0 <= p0 <= 1.0


def test_train_multiclass_bundle_takeprofit_index(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.integers(-1, 2, size=n)
    y[0], y[1], y[2] = -1, 0, 1
    clf = train_xgb_classifier(
        X,
        y,
        task="multiclass",
        random_state=1,
        xgb_params={"n_estimators": 15, "max_depth": 2},
    )
    save_model_bundle(tmp_path, clf, feature_names=list(X.columns), task="multiclass")
    bundle = load_model_bundle(tmp_path)
    proba = bundle.predict_proba_row(X.iloc[:1])
    p_tp = proba_positive_takeprofit(bundle, proba)
    assert np.isfinite(p_tp)
