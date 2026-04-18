"""Streamlit runner: model metrics with purged CV + XY prep mocked."""

from __future__ import annotations

import numpy as np
import pandas as pd

import aprilalgo.ui.pages.model_metrics as mm
from aprilalgo.ui.pages.model_metrics import render


def _fake_prepare_xy(cfg):  # noqa: ANN001
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f0": rng.normal(size=48), "f1": rng.normal(size=48)})
    y = pd.Series(rng.integers(0, 2, size=48))
    t0 = np.arange(48, dtype=np.int64)
    t1 = t0 + 3
    return X, y, t0, t1, "binary"


def _fake_purged_cv_evaluate(*args, **kwargs):  # noqa: ANN002
    return {
        "mean": {"f1": 0.71, "accuracy": 0.70},
        "folds_df": pd.DataFrame({"fold": [0, 1], "accuracy": [0.69, 0.72]}),
        "folds": [],
    }


mm._prepare_xy = _fake_prepare_xy  # type: ignore[method-assign]
mm.purged_cv_evaluate = _fake_purged_cv_evaluate  # type: ignore[method-assign]

render()
