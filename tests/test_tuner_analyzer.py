"""Tests for aprilalgo.tuner.analyzer."""

from __future__ import annotations

import pandas as pd
import pytest

from aprilalgo.tuner.analyzer import _check_robustness, analyze_results


def test_empty_df_returns_empty_shape() -> None:
    r = analyze_results(pd.DataFrame(), metric="sharpe_ratio")
    assert r["best"] == {}
    assert r["top_n"].empty
    assert r["robustness"] == {}


def test_missing_metric_column() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    r = analyze_results(df, metric="sharpe_ratio")
    assert r["best"] == {}


def test_all_nan_metric_returns_empty() -> None:
    df = pd.DataFrame({"sharpe_ratio": [float("nan"), float("nan")], "max_depth": [2, 3]})
    r = analyze_results(df, metric="sharpe_ratio")
    assert r["best"] == {}


def test_best_selection_and_top_n() -> None:
    df = pd.DataFrame({"max_depth": range(12), "sharpe_ratio": [float(i) * 0.1 for i in range(12)]})
    r = analyze_results(df, metric="sharpe_ratio", top_n=5)
    assert r["best"]["sharpe_ratio"] == pytest.approx(1.1, rel=1e-6)
    assert len(r["top_n"]) == 5


def test_robustness_no_params() -> None:
    df = pd.DataFrame({"sharpe_ratio": [0.5, 0.4], "total_pnl": [1.0, 2.0]})
    r = analyze_results(df, metric="sharpe_ratio")
    assert r["robustness"]["robust"] is True
    assert "No parameters" in r["robustness"]["reason"]


def test_robustness_zero_best() -> None:
    df = pd.DataFrame({"max_depth": [2, 3], "sharpe_ratio": [0.0, -0.5]})
    r = analyze_results(df, metric="sharpe_ratio")
    assert r["robustness"]["robust"] is False
    assert r["robustness"]["reason"] == "Best metric is zero"


def test_robustness_negative_best_uses_abs_denominator() -> None:
    df = pd.DataFrame({"max_depth": [2, 2, 2], "sharpe_ratio": [-1.5, -1.0, -1.2]})
    best = {"max_depth": 2, "sharpe_ratio": -2.0}
    out = _check_robustness(df, best, "sharpe_ratio")
    assert out["degradation_pct"] < 0


def test_robustness_too_few_neighbors() -> None:
    df = pd.DataFrame({"max_depth": [3], "sharpe_ratio": [1.0]})
    best = {"max_depth": 3, "sharpe_ratio": 1.0}
    out = _check_robustness(df, best, "sharpe_ratio")
    assert out["robust"] is False
    assert "Too few nearby" in out["reason"]
    assert "neighbor_count" in out
