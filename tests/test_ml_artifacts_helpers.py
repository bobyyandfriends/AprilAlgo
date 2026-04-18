"""Tests for charts/data/ml_artifacts helpers (Streamlit cache patched where needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import aprilalgo.ui.pages.charts.data.ml_artifacts as ma


def test_project_relative_and_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ma, "_PROJECT_ROOT", tmp_path)
    sub = tmp_path / "models" / "a"
    sub.mkdir(parents=True)
    rel = ma.project_relative(sub)
    assert rel.replace("\\", "/") == "models/a"
    assert ma.resolve_model_dir(rel).resolve() == sub.resolve()


def test_discover_model_dirs_empty_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ma, "_MODELS_DIR", tmp_path / "nomodels")
    assert ma.discover_model_dirs() == []


def test_discover_model_dirs_finds_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    monkeypatch.setattr(ma, "_PROJECT_ROOT", root)
    mdir = root / "models" / "b1"
    mdir.mkdir(parents=True)
    (mdir / "meta.json").write_text("{}", encoding="utf-8")
    (mdir / "xgboost.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ma, "_MODELS_DIR", root / "models")
    found = ma.discover_model_dirs()
    assert any("b1" in str(p).replace("\\", "/") for p in found)


def test_build_proba_frame_pads_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ma,
        "load_oof_proba",
        lambda md: pd.DataFrame({"proba_1": [0.1, 0.2]}),
    )
    wide = ma.build_proba_frame_for_df("m", df_len=5, df_signature=("S", "d", 1, 2))
    assert wide is not None
    assert len(wide) == 5
    assert wide["proba_1"].isna().sum() == 3


def test_build_proba_frame_tail_when_long_oof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ma,
        "load_oof_proba",
        lambda md: pd.DataFrame({"proba_1": [0.1, 0.2, 0.3, 0.4]}),
    )
    exact = ma.build_proba_frame_for_df("m", df_len=3, df_signature=("S", "d", 2, 3))
    assert exact is not None and len(exact) == 3


def test_build_proba_frame_no_proba_cols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ma, "load_oof_proba", lambda md: pd.DataFrame({"row_idx": [0]}))
    assert ma.build_proba_frame_for_df("m", df_len=2, df_signature=()) is None


def test_compute_shap_tables_missing_features_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    b = MagicMock()
    b.feature_names = ["a", "b"]
    monkeypatch.setattr(ma, "load_bundle", lambda p: b)
    ff = pd.DataFrame({"a": [1.0, 2.0]})
    assert ma.compute_shap_tables("md", ("a", "b"), ff) is None


def test_compute_shap_tables_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(p):
        raise FileNotFoundError

    monkeypatch.setattr(ma, "load_bundle", boom)
    assert ma.compute_shap_tables("md", ("a",), pd.DataFrame({"a": [1.0]})) is None


def test_compute_shap_tables_shap_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    b = MagicMock()
    b.feature_names = ["a"]
    monkeypatch.setattr(ma, "load_bundle", lambda p: b)

    def boom(*_a, **_k):
        raise ImportError("shap not installed")

    monkeypatch.setattr(ma, "shap_values_table", boom)
    monkeypatch.setattr(ma, "shap_importance_table", lambda *a, **k: pd.DataFrame())
    assert ma.compute_shap_tables("md", ("a",), pd.DataFrame({"a": [1.0]})) is None


def test_shap_values_to_wide_empty() -> None:
    assert ma.shap_values_to_wide(pd.DataFrame()).empty


def test_align_shap_to_price_rows_padding() -> None:
    mask = np.array([False, True, True, False])
    wide = pd.DataFrame({"f": [0.1, 0.2]})
    out = ma.align_shap_to_price_rows(wide, mask)
    assert len(out) == 4
    assert pd.isna(out.iloc[0]["f"])


def test_build_feature_frame_none_config() -> None:
    df = pd.DataFrame({"a": [1.0]})
    X, m = ma.build_feature_frame_for_chart(df, None)
    assert X.empty and m.sum() == 0
