"""SHAP explainability tests."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import shap  # noqa: F401

from aprilalgo.ml.explain import (
    _tree_explainer,
    shap_importance_table,
    shap_values_per_regime,
    shap_values_table,
)
from aprilalgo.ml.trainer import load_model_bundle, save_model_bundle, train_xgb_classifier


def test_shap_tables_shape(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 20, "max_depth": 2},
    )
    save_model_bundle(tmp_path, clf, feature_names=["a", "b"], task="binary")
    bundle = load_model_bundle(tmp_path)
    vals = shap_values_table(bundle, X.iloc[:25], max_samples=25)
    assert {"feature", "sample_idx", "shap_value"} <= set(vals.columns)
    imp = shap_importance_table(bundle, X.iloc[:25], max_samples=25)
    assert {"feature", "mean_abs_shap", "rank"} <= set(imp.columns)
    assert len(imp) == 2


def test_per_regime_shape(tmp_path):
    rng = np.random.default_rng(1)
    bundles = {}
    X_by = {}
    for k in ("0", "1"):
        n = 45
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = ((X["a"] + X["b"]) > 0).astype(np.int64)
        clf = train_xgb_classifier(
            X,
            y,
            task="binary",
            random_state=0,
            xgb_params={"n_estimators": 16, "max_depth": 2},
        )
        d = tmp_path / f"regime_{k}"
        save_model_bundle(d, clf, feature_names=["a", "b"], task="binary")
        bundles[k] = load_model_bundle(d)
        X_by[k] = X.iloc[:15]
    out = shap_values_per_regime(bundles, X_by, max_samples=15)
    assert set(out) == {"0", "1"}
    for k in out:
        assert {"feature", "sample_idx", "shap_value"} <= set(out[k]["values"].columns)
        assert len(out[k]["importance"]) == 2


def test_shap_matrix_handles_list_api(tmp_path):
    rng = np.random.default_rng(2)
    n = 30
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 12, "max_depth": 2},
    )
    save_model_bundle(tmp_path, clf, feature_names=["a", "b"], task="binary")
    bundle = load_model_bundle(tmp_path)

    class _Expl:
        def shap_values(self, xx: pd.DataFrame):
            m = len(xx)
            return [np.zeros((m, 2)), np.ones((m, 2)) * 0.5]

    with patch("aprilalgo.ml.explain._tree_explainer", return_value=_Expl()):
        vals = shap_values_table(bundle, X.iloc[:10], max_samples=10)
        imp = shap_importance_table(bundle, X.iloc[:10], max_samples=10)
    assert len(vals) == 10 * 2
    assert (imp["mean_abs_shap"] >= 0).all()


def test_shap_matrix_handles_3d_array(tmp_path):
    rng = np.random.default_rng(3)
    n = 25
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 10, "max_depth": 2},
    )
    save_model_bundle(tmp_path, clf, feature_names=["a", "b"], task="binary")
    bundle = load_model_bundle(tmp_path)

    class _Expl3:
        def shap_values(self, xx: pd.DataFrame):
            m = len(xx)
            return np.zeros((3, m, 2))

    with patch("aprilalgo.ml.explain._tree_explainer", return_value=_Expl3()):
        vals = shap_values_table(bundle, X.iloc[:8], max_samples=8)
        imp = shap_importance_table(bundle, X.iloc[:8], max_samples=8)
    assert len(vals) == 8 * 2
    assert (imp["mean_abs_shap"] >= 0).all()


def test_tree_explainer_import_error(tmp_path):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 8, "max_depth": 2},
    )
    save_model_bundle(tmp_path, clf, feature_names=["a", "b"], task="binary")
    bundle = load_model_bundle(tmp_path)

    real_import = builtins.__import__

    def _guard(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "shap":
            raise ImportError("simulated missing shap")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=_guard):
        with pytest.raises(ImportError, match="SHAP is not installed"):
            _tree_explainer(bundle)
