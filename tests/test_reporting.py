"""HTML report smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aprilalgo.reporting.report import (
    render_backtest_html,
    render_full_ml_report_html,
    render_meta_section,
    render_regime_section,
    render_sampling_section,
    render_wf_tuner_section,
)


def test_render_backtest_html(tmp_path: Path):
    p = tmp_path / "r.html"
    render_backtest_html({"sharpe_ratio": 1.2, "num_trades": 5}, p, title="T")
    text = p.read_text(encoding="utf-8")
    assert "sharpe_ratio" in text
    assert "1.2" in text
    assert 'id="section-metrics"' in text


def test_render_includes_optional_sections(tmp_path: Path):
    eq = pd.DataFrame({"datetime": [1, 2], "equity": [1.0, 1.01]})
    imp = pd.DataFrame({"feature": ["a"], "score": [1.0]})
    shap = pd.DataFrame({"feature": ["a"], "mean_abs_shap": [0.3], "rank": [1]})
    wf = pd.DataFrame({"fold": [0], "test_return": [0.01]})
    p = tmp_path / "full.html"
    render_backtest_html(
        {"x": 1},
        p,
        equity=eq,
        importance=imp,
        regime=eq,
        shap=shap,
        walk_forward=wf,
    )
    t = p.read_text(encoding="utf-8")
    for sid in (
        "section-equity",
        "section-importance",
        "section-regime-timeline",
        "section-shap",
        "section-walk-forward",
    ):
        assert sid in t


def test_sampling_section_id() -> None:
    w = pd.Series(np.linspace(0.1, 2.0, 50))
    html = render_sampling_section(w)
    assert 'id="section-sampling"' in html


def test_meta_section_id() -> None:
    coef = pd.DataFrame({"feature": ["primary_pred"], "coefficient": [0.42]})
    html = render_meta_section(coef, oof_coverage=0.91, oof_rows=100, oof_nonnull=88)
    assert 'id="section-meta"' in html


def test_regime_section_id() -> None:
    bc = pd.DataFrame({"bucket": [0, 1], "n_rows": [40, 60]})
    acc = pd.DataFrame({"bucket": [0, 1], "accuracy": [0.55, 0.60]})
    html = render_regime_section(bc, acc)
    assert 'id="section-regime"' in html


def test_wf_tuner_section_id() -> None:
    wf = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "wf_fold": [0, 1, 0, 1],
            "score": [0.5, 0.7, 0.4, 0.6],
        }
    )
    html = render_wf_tuner_section(wf, top_n=2)
    assert 'id="section-wf-tuner"' in html


def test_render_full_html_sections_when_all_artifacts_present(tmp_path: Path) -> None:
    w = pd.Series(np.random.default_rng(0).random(30) * 3.0 + 0.01)
    meta = pd.DataFrame({"feature": ["f"], "coef": [0.1]})
    bc = pd.DataFrame({"bucket": [0], "n_rows": [30]})
    acc = pd.DataFrame({"bucket": [0], "accuracy": [0.66]})
    wf = pd.DataFrame(
        {
            "grid_id": ["x", "x", "y", "y"],
            "wf_fold": [0, 1, 0, 1],
            "score": [0.8, 0.82, 0.5, 0.52],
        }
    )
    p = tmp_path / "ml_full.html"
    render_full_ml_report_html(
        {"auc": 0.7},
        p,
        title="ML smoke",
        sampling_weights=w,
        meta_coef=meta,
        oof_coverage=0.9,
        oof_rows=100,
        oof_nonnull=90,
        regime_bucket_counts=bc,
        regime_accuracy=acc,
        wf_tuner_results=wf,
    )
    t = p.read_text(encoding="utf-8")
    for sid in (
        "section-metrics",
        "section-sampling",
        "section-meta",
        "section-regime",
        "section-wf-tuner",
    ):
        assert sid in t


def test_render_sampling_section_empty_inputs() -> None:
    assert render_sampling_section(None) == ""
    assert render_sampling_section(pd.Series([], dtype=float)) == ""
    assert render_sampling_section(pd.Series([float("nan"), float("nan")])) == ""


def test_render_meta_section_empty_returns_empty_string() -> None:
    assert render_meta_section(None) == ""
    assert render_meta_section(pd.DataFrame()) == ""


def test_render_meta_section_stats_only_no_coef() -> None:
    html = render_meta_section(None, oof_coverage=0.5, oof_rows=10, oof_nonnull=8)
    assert 'id="section-meta"' in html
    assert "0.5000" in html
    assert "10" in html


def test_render_regime_section_empty_both() -> None:
    assert render_regime_section(None, None) == ""
    assert render_regime_section(pd.DataFrame(), pd.DataFrame()) == ""


def test_render_wf_tuner_section_empty() -> None:
    assert render_wf_tuner_section(None) == ""
    assert render_wf_tuner_section(pd.DataFrame()) == ""


def test_render_backtest_equity_truncation_note_over_500_rows(tmp_path: Path) -> None:
    n = 600
    eq = pd.DataFrame({"datetime": range(n), "equity": np.linspace(1.0, 1.2, n)})
    p = tmp_path / "big.html"
    render_backtest_html({"x": 1}, p, equity=eq)
    t = p.read_text(encoding="utf-8")
    assert "first 500" in t
    assert "600" in t


def test_render_full_ml_report_html_escapes_notes(tmp_path: Path) -> None:
    p = tmp_path / "notes.html"
    render_full_ml_report_html({"a": 1}, p, notes='<evil>&"')
    text = p.read_text(encoding="utf-8")
    assert "<evil>" not in text
    assert "&lt;evil&gt;" in text or "&amp;" in text
