"""Unit tests for charts ml_artifacts helpers (no full Streamlit app)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aprilalgo.ui.pages.charts.data.ml_artifacts as ma


def test_project_relative_and_resolve_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(ma, "_PROJECT_ROOT", root)
    assert ma.resolve_model_dir("outputs/ml") == root / "outputs" / "ml"
    abs_p = (tmp_path / "absmodel").resolve()
    abs_p.mkdir(parents=True)
    assert ma.resolve_model_dir(str(abs_p)) == abs_p
    assert ma.project_relative(root / "outputs" / "x") == "outputs/x"
    outside = tmp_path / "sibling" / "nope.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    assert ma.project_relative(outside) == str(outside)


def test_discover_model_dirs_finds_bundles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = tmp_path / "models"
    bundle = models / "run1"
    bundle.mkdir(parents=True)
    (bundle / "meta.json").write_text("{}", encoding="utf-8")
    (bundle / "xgboost.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ma, "_MODELS_DIR", models)
    found = ma.discover_model_dirs()
    assert bundle in found


def test_bundle_meta_empty_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_p: Path | str) -> None:
        raise FileNotFoundError("no")

    monkeypatch.setattr(ma, "load_bundle", _boom)
    assert ma.bundle_meta("any/path") == {}


def test_shap_values_to_wide_empty_and_nonempty() -> None:
    assert ma.shap_values_to_wide(None).empty
    assert ma.shap_values_to_wide(pd.DataFrame()).empty
    long_df = pd.DataFrame(
        {"sample_idx": [0, 0, 1, 1], "feature": ["a", "b", "a", "b"], "shap_value": [0.1, -0.2, 0.3, 0.0]}
    )
    wide = ma.shap_values_to_wide(long_df)
    assert wide.shape == (2, 2)
    assert "a" in wide.columns


def test_build_feature_frame_for_chart_none_config() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    X, mask = ma.build_feature_frame_for_chart(df, None)
    assert X.empty
    assert mask.shape == (2,) and not mask.any()


def test_align_shap_to_price_rows_padding() -> None:
    mask = np.array([False, True, False, True], dtype=bool)
    shap_wide = pd.DataFrame({"f1": [1.0, 2.0]})
    out = ma.align_shap_to_price_rows(shap_wide, mask)
    assert len(out) == 4
    assert pd.isna(out.iloc[0]["f1"])
    assert float(out.iloc[1]["f1"]) == 1.0


def test_align_shap_to_price_rows_empty_wide() -> None:
    mask = np.ones(3, dtype=bool)
    out = ma.align_shap_to_price_rows(pd.DataFrame(), mask)
    assert len(out) == 3

