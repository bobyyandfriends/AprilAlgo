"""Tests for serialized meta-label logistic bundle."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aprilalgo.ml.meta_bundle import load_meta_logit_bundle, save_meta_logit_bundle


def test_predict_proba_matches_sklearn(tmp_path) -> None:
    rng = np.random.default_rng(7)
    n = 80
    Xf = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    primary = rng.integers(0, 2, size=n).astype(np.float64)
    X_meta = pd.concat(
        [Xf, pd.Series(primary, name="primary_pred")],
        axis=1,
    )
    z = ((Xf["a"] + primary) > 0).astype(int).to_numpy()
    clf = LogisticRegression(max_iter=300, random_state=0).fit(X_meta, z)
    save_meta_logit_bundle(
        tmp_path, clf, feature_names=list(X_meta.columns)
    )
    bundle = load_meta_logit_bundle(tmp_path)
    pr_sk = clf.predict_proba(X_meta)
    pr_b = bundle.predict_proba(X_meta)
    np.testing.assert_allclose(pr_b, pr_sk, rtol=1e-5, atol=1e-5)


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="meta_logit"):
        load_meta_logit_bundle(tmp_path)


def test_load_meta_logit_custom_rel_path(tmp_path) -> None:
    rng = np.random.default_rng(3)
    n = 40
    Xf = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    primary = rng.integers(0, 2, size=n).astype(np.float64)
    X_meta = pd.concat([Xf, pd.Series(primary, name="primary_pred")], axis=1)
    z = (Xf["a"] > 0).astype(int).to_numpy()
    clf = LogisticRegression(max_iter=300, random_state=0).fit(X_meta, z)
    alt = tmp_path / "nested" / "m.json"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_text(
        json.dumps(
            {
                "feature_names": list(X_meta.columns),
                "coef": np.asarray(clf.coef_).tolist(),
                "intercept": np.asarray(clf.intercept_).ravel().tolist(),
                "classes_": [float(c) for c in np.asarray(clf.classes_).ravel()],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    b = load_meta_logit_bundle(tmp_path, rel_path="nested/m.json")
    pr = b.predict_proba(X_meta)
    np.testing.assert_allclose(pr, clf.predict_proba(X_meta), rtol=1e-5, atol=1e-5)


def test_predict_proba_missing_column_raises(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = 30
    X_meta = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "primary_pred": rng.integers(0, 2, size=n).astype(float),
        }
    )
    z = (X_meta["a"] > 0).astype(int).to_numpy()
    clf = LogisticRegression(max_iter=200, random_state=0).fit(X_meta, z)
    save_meta_logit_bundle(tmp_path, clf, feature_names=list(X_meta.columns))
    bundle = load_meta_logit_bundle(tmp_path)
    bad = X_meta.drop(columns=["primary_pred"])
    with pytest.raises(ValueError, match="missing columns"):
        bundle.predict_proba(bad)
