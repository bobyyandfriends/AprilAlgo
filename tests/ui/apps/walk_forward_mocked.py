"""Streamlit runner: walk-forward lab with subprocess.run patched."""

import json
from unittest.mock import MagicMock

import aprilalgo.ui.pages.walk_forward_lab as wfl
from aprilalgo.ui.pages.walk_forward_lab import render


def _fake_ok(*args, **kwargs):
    payload = {
        "n_bars": 100,
        "summary": {"n_splits": 2, "coverage_pct": 50.0, "mean_train_size": 40.0, "mean_test_size": 10.0},
        "splits": [
            {"fold": 0, "train_size": 40, "test_size": 10, "test_return": 0.01},
            {"fold": 1, "train_size": 50, "test_size": 10, "test_return": -0.02},
        ],
    }
    return MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")


wfl.subprocess.run = _fake_ok  # type: ignore[method-assign]

render()
