"""ML strategy backtest and logging."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression

from aprilalgo.backtest.engine import run_backtest
from aprilalgo.data import load_price_data
from aprilalgo.ml.features import build_feature_matrix
from aprilalgo.ml.meta_bundle import save_meta_logit_bundle
from aprilalgo.ml.trainer import save_model_bundle, train_xgb_classifier
from aprilalgo.strategies.configurable import ConfigurableStrategy
from aprilalgo.strategies.ml_strategy import MLStrategy

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ROOT_REPO = Path(__file__).resolve().parents[1]
_CFG_ML = _ROOT_REPO / "configs" / "ml" / "default.yaml"
_INDICATORS = [
    {"name": "rsi", "period": 14},
    {"name": "sma", "period": 20},
]


def _train_xgb_and_meta_bundle(tmp_path: Path, df: pd.DataFrame) -> Path:
    """Write ``model_dir`` with XGBoost + ``meta_logit.json`` and ``meta_logit`` enabled in ``meta.json``."""
    X = build_feature_matrix(df, indicator_config=_INDICATORS).reset_index(drop=True)
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    c0 = X.iloc[:, 0].to_numpy(dtype=np.float64)
    y = pd.Series((c0 > np.nanmedian(c0)).astype(np.int64))
    if y.nunique() < 2:
        y = pd.Series(np.arange(len(X), dtype=np.int64) % 2)

    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 25, "max_depth": 3},
    )
    model_dir = tmp_path / "m"
    save_model_bundle(
        model_dir,
        clf,
        feature_names=list(X.columns),
        task="binary",
        indicator_config=_INDICATORS,
        extra_meta={
            "symbol": "TEST",
            "meta_logit": {
                "enabled": True,
                "path": "meta_logit.json",
                "threshold": 0.5,
            },
        },
    )
    primary_hat = clf.predict(X).astype(np.float64)
    X_meta = pd.concat(
        [X, pd.Series(primary_hat, name="primary_pred")],
        axis=1,
    )
    z_meta = (X_meta.iloc[:, 0] > float(np.nanmedian(X_meta.iloc[:, 0]))).astype(int).to_numpy()
    if np.unique(z_meta).size < 2:
        z_meta = np.arange(len(X_meta), dtype=np.int64) % 2
    meta_clf = LogisticRegression(max_iter=500, random_state=0).fit(X_meta, z_meta)
    save_meta_logit_bundle(model_dir, meta_clf, feature_names=list(X_meta.columns))
    return model_dir


def test_ml_strategy_ctor_has_meta_proba_threshold() -> None:
    assert "meta_proba_threshold" in inspect.signature(MLStrategy.__init__).parameters


def test_meta_bundle_loaded_when_enabled(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    model_dir = _train_xgb_and_meta_bundle(tmp_path, df)
    strat = MLStrategy(model_dir, symbol="TEST", timeframe="daily")
    strat.init(df)
    assert strat._meta_gate_enabled
    assert strat._meta_bundle is not None


def test_meta_proba_shape(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    model_dir = _train_xgb_and_meta_bundle(tmp_path, df)
    strat = MLStrategy(model_dir, symbol="TEST", timeframe="daily")
    strat.init(df)
    mb = strat._meta_bundle
    assert mb is not None
    for i in range(len(strat._X)):
        xr = strat._X.iloc[i : i + 1]
        if xr.isna().any(axis=None):
            continue
        pred = float(strat._bundle.predict(xr)[0])  # type: ignore[union-attr]
        p = strat._p_meta(xr, pred)
        assert p is not None
        x_meta = xr.copy()
        x_meta["primary_pred"] = pred
        pr = mb.predict_proba(x_meta[list(mb.feature_names)])
        assert pr.shape == (1, 2)
        break
    else:
        pytest.fail("no valid feature row for meta proba")


def test_meta_gate_blocks_entry_below_threshold(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    model_dir = _train_xgb_and_meta_bundle(tmp_path, df)

    def _num_trades(meta_th: float) -> int:
        s = MLStrategy(
            model_dir,
            symbol="TEST",
            timeframe="daily",
            entry_proba_threshold=0.0,
            meta_proba_threshold=meta_th,
            position_pct=0.5,
        )
        r = run_backtest(s, df, initial_capital=50_000.0)
        return int(r["metrics"]["num_trades"])

    open_trades = _num_trades(0.0)
    blocked = _num_trades(1.01)  # above any [0, 1] probability → gate always blocks
    assert open_trades > 0, "fixture should produce trades when meta gate is wide open"
    assert blocked == 0, "meta threshold above 1.0 should block every entry"


def test_meta_proba_logged_in_event(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    model_dir = _train_xgb_and_meta_bundle(tmp_path, df)
    log_path = tmp_path / "sig.jsonl"
    strat = MLStrategy(
        model_dir,
        symbol="TEST",
        timeframe="daily",
        entry_proba_threshold=0.0,
        meta_proba_threshold=0.0,
        position_pct=0.5,
        signal_log_path=log_path,
    )
    run_backtest(strat, df, initial_capital=50_000.0)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    for raw in lines:
        row = json.loads(raw)
        if row.get("event") == "entry":
            assert row.get("pred_proba_meta") is not None
            assert isinstance(row["pred_proba_meta"], (int, float))
            return
    pytest.fail("expected at least one entry event in log")


def test_legacy_bundle_no_meta_gate(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    X = build_feature_matrix(df, indicator_config=_INDICATORS).reset_index(drop=True)
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    c0 = X.iloc[:, 0].to_numpy(dtype=np.float64)
    y = pd.Series((c0 > np.nanmedian(c0)).astype(np.int64))
    if y.nunique() < 2:
        y = pd.Series(np.arange(len(X), dtype=np.int64) % 2)
    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 25, "max_depth": 3},
    )
    model_dir = tmp_path / "legacy"
    save_model_bundle(
        model_dir,
        clf,
        feature_names=list(X.columns),
        task="binary",
        indicator_config=_INDICATORS,
        extra_meta={"symbol": "TEST"},
    )
    strat = MLStrategy(model_dir, symbol="TEST", timeframe="daily")
    strat.init(df)
    assert not strat._meta_gate_enabled
    assert strat._meta_bundle is None


def test_backtest_runs_with_meta_bundle(tmp_path: Path) -> None:
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    model_dir = _train_xgb_and_meta_bundle(tmp_path, df)
    strat = MLStrategy(
        model_dir,
        symbol="TEST",
        timeframe="daily",
        entry_proba_threshold=0.0,
        meta_proba_threshold=0.0,
        position_pct=0.5,
    )
    res = run_backtest(strat, df, initial_capital=50_000.0)
    assert res["metrics"] is not None
    assert "num_trades" in res["metrics"]


def test_ml_strategy_logs_full_schema_events(tmp_path: Path) -> None:

    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    X = build_feature_matrix(df, indicator_config=_INDICATORS).reset_index(drop=True)
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    # Synthetic binary target from first feature (guarantees 2 classes for XGBoost).
    c0 = X.iloc[:, 0].to_numpy(dtype=np.float64)
    y = pd.Series((c0 > np.nanmedian(c0)).astype(np.int64))
    if y.nunique() < 2:
        y = pd.Series(np.arange(len(X), dtype=np.int64) % 2)

    clf = train_xgb_classifier(
        X,
        y,
        task="binary",
        random_state=0,
        xgb_params={"n_estimators": 25, "max_depth": 3},
    )
    model_dir = tmp_path / "m"
    save_model_bundle(
        model_dir,
        clf,
        feature_names=list(X.columns),
        task="binary",
        indicator_config=_INDICATORS,
        extra_meta={"symbol": "TEST"},
    )

    log_path = tmp_path / "sig.jsonl"
    strat = MLStrategy(
        model_dir,
        symbol="TEST",
        timeframe="daily",
        entry_proba_threshold=0.0,
        position_pct=0.5,
        signal_log_path=log_path,
        position_sizer="fixed_fraction",
        sizer_fraction=0.05,
    )
    res = run_backtest(strat, df, initial_capital=50_000.0)
    assert res["metrics"] is not None
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    if text.strip():
        row = json.loads(text.splitlines()[0])
        assert "features_hash" in row and "pred_proba" in row
        assert "pred_proba_meta" in row
        assert row["pred_proba_meta"] is None


@pytest.mark.skipif(not _CFG_ML.is_file(), reason="default ML config missing")
def test_regime_applied_in_strategy(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG_ML.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_strat_regime"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 8,
        "n_buckets": 3,
        "use_hmm": False,
    }
    cfg_path = tmp_path / "ml_strat_regime.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    strat = MLStrategy(
        out_bundle,
        symbol="TEST",
        timeframe="daily",
        entry_proba_threshold=0.99,
    )
    strat.init(df)
    assert "vol_regime" in strat._X.columns


@pytest.mark.skipif(not _CFG_ML.is_file(), reason="default ML config missing")
def test_strategy_routes_per_regime(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG_ML.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_strat_gb"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_strat_gb.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    strat = MLStrategy(
        out_bundle,
        symbol="TEST",
        timeframe="daily",
        entry_proba_threshold=0.99,
    )
    strat.init(df)
    assert strat._regime_bundles is not None
    assert len(strat._regime_bundles) >= 1
    res = run_backtest(strat, df, initial_capital=50_000.0)
    assert res["metrics"] is not None


def test_configurable_strategy_runs_same_fixture():
    df = load_price_data("TEST", "daily", data_dir=_FIXTURES)
    strat = ConfigurableStrategy(
        indicators=_INDICATORS,
        entry_threshold=0.5,
        exit_threshold=-0.2,
        position_pct=0.5,
    )
    res = run_backtest(strat, df, initial_capital=50_000.0)
    assert "equity" in res
