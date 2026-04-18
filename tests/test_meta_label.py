"""Tests for meta-label logistic helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aprilalgo.labels.meta_label import build_meta_labels, fit_meta_logit_purged


def test_build_meta_labels():
    y = np.array([1, 0, 1])
    p = np.array([1, 1, 0])
    z = build_meta_labels(y, p)
    assert list(z) == [1, 0, 0]


def test_meta_gated_beats_primary_only_on_holdout():
    """Meta model learns when a biased primary is wrong; gating beats raw primary."""
    rng = np.random.default_rng(99)
    n = 2_000
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    X = pd.DataFrame({"f1": f1, "f2": f2})
    # XOR-style label: both features matter; primary uses only f1>0 (systematically wrong).
    y = (((f1 > 0) ^ (f2 > 0))).astype(int)
    primary = (f1 > 0).astype(int)
    split = 1_400
    tr, te = np.arange(split), np.arange(split, n)
    z_tr = (primary[tr] == y[tr]).astype(int)
    Xm_tr = pd.DataFrame(
        {
            "f1": X.iloc[tr]["f1"].to_numpy(),
            "f2": X.iloc[tr]["f2"].to_numpy(),
            "pp": primary[tr],
        }
    )
    meta = LogisticRegression(max_iter=300, random_state=0).fit(Xm_tr, z_tr)
    Xm_te = pd.DataFrame(
        {
            "f1": X.iloc[te]["f1"].to_numpy(),
            "f2": X.iloc[te]["f2"].to_numpy(),
            "pp": primary[te],
        }
    )
    meta_p = meta.predict_proba(Xm_te)[:, 1]
    gated = np.where(meta_p >= 0.5, primary[te], 1 - primary[te])
    acc_primary = float((primary[te] == y[te]).mean())
    acc_gated = float((gated == y[te]).mean())
    assert acc_gated > acc_primary


def test_fit_meta_logit_raises_on_single_class_z():
    rng = np.random.default_rng(3)
    n = 40
    X = pd.DataFrame({"f1": rng.normal(size=n)})
    y = np.ones(n, dtype=int)
    oof = np.ones(n, dtype=int)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    with pytest.raises(ValueError, match="single class"):
        fit_meta_logit_purged(X, y, oof, sample_t0=t0, sample_t1=t1, n_splits=2, embargo=0)


def test_fit_meta_logit_produces_oof():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({"f1": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    oof_pred = rng.integers(0, 2, size=n)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 2
    _, meta_oof, z = fit_meta_logit_purged(
        X, y, oof_pred, sample_t0=t0, sample_t1=t1, n_splits=3, embargo=0
    )
    assert len(z) == n
    assert np.isfinite(meta_oof).sum() > n // 2
