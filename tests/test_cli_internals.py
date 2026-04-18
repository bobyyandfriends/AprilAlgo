"""Unit tests for aprilalgo.cli internals (high branch coverage; mocks for heavy ML)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

import aprilalgo.cli as cli


def test_eval_result_for_json_empty_and_nonempty_folds_df() -> None:
    res = {"mean": {"a": 1}, "folds_df": pd.DataFrame()}
    out = cli._eval_result_for_json(res)
    assert out["folds_df"] == []
    res2 = {"mean": {"a": 1}, "folds_df": pd.DataFrame({"f": [0], "x": [1.0]})}
    out2 = cli._eval_result_for_json(res2)
    assert len(out2["folds_df"]) == 1


def test_read_oof_primary_aligned_errors(tmp_path: Path) -> None:
    p = tmp_path / "oof_primary.csv"
    with pytest.raises(FileNotFoundError, match="Missing"):
        cli._read_oof_primary_aligned(p, n_rows=3)
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_idx"):
        cli._read_oof_primary_aligned(p, n_rows=1)
    p.write_text("row_idx,oof_pred\n0,0.5\n1,0.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared feature matrix has 3"):
        cli._read_oof_primary_aligned(p, n_rows=3)
    p.write_text("row_idx,oof_pred\n0,0.5\n2,0.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_idx must equal"):
        cli._read_oof_primary_aligned(p, n_rows=2)


def test_read_oof_primary_aligned_success(tmp_path: Path) -> None:
    p = tmp_path / "oof_primary.csv"
    p.write_text("row_idx,oof_pred\n1,0.1\n0,0.2\n", encoding="utf-8")
    arr, df = cli._read_oof_primary_aligned(p, n_rows=2)
    assert np.allclose(arr, [0.2, 0.1])
    assert len(df) == 2


def test_cmd_meta_train_requires_symbol(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"out_dir": str(tmp_path)}}), encoding="utf-8")
    with pytest.raises(ValueError, match="symbol"):
        cli.cmd_meta_train(SimpleNamespace(config=str(cfg_path)))


def test_cmd_meta_train_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "mout"
    out.mkdir()
    (out / "meta.json").write_text('{"existing": true}', encoding="utf-8")
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 1.0, 1.0]})
    y = pd.Series([0.0, 1.0, 0.0])
    t0 = np.arange(3, dtype=np.int64)
    t1 = t0 + 1

    def _pxy(_cfg, symbol=None):
        return X, y, t0, t1, "binary"

    meta_mock = MagicMock()

    def _fit(*_a, **_k):
        return meta_mock, np.array([0.5, 0.5, 0.5]), np.array([1, 0, 1], dtype=np.int64)

    monkeypatch.setattr(cli, "_prepare_xy", _pxy)
    monkeypatch.setattr(cli, "fit_meta_logit_purged", _fit)
    monkeypatch.setattr(cli, "save_meta_logit_bundle", lambda *a, **k: None)
    oof = out / "oof_primary.csv"
    oof.write_text("row_idx,oof_pred\n0,0.0\n1,1.0\n2,0.0\n", encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symbol": "Z", "model": {"out_dir": str(out)}, "meta_label": {"threshold": 0.6}}),
        encoding="utf-8",
    )
    cli.cmd_meta_train(SimpleNamespace(config=str(cfg_path)))
    assert (out / "meta_oof.csv").is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("meta_logit", {}).get("threshold") == 0.6


def test_cmd_oof_updates_meta_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    (out_dir / "meta.json").write_text("{}", encoding="utf-8")
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y = pd.Series([0, 1] * 10)
    t0 = np.arange(20, dtype=np.int64)
    t1 = t0 + 2

    def _pxy(_cfg, symbol=None):
        return X, y, t0, t1, "binary"

    def _factory():
        return MagicMock()

    oof_df = pd.DataFrame({"row_idx": np.arange(20), "oof_pred": np.linspace(0, 1, 20)})

    monkeypatch.setattr(cli, "_prepare_xy", _pxy)
    monkeypatch.setattr(cli, "_xgb_estimator_factory", lambda c, t: _factory)
    monkeypatch.setattr(cli, "_weights_for_training", lambda *a, **k: None)
    monkeypatch.setattr(cli, "compute_primary_oof", lambda *a, **kw: oof_df)
    cfg_path = tmp_path / "oof.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(out_dir)}, "cv": {"n_splits": 2, "embargo": 0}}),
        encoding="utf-8",
    )
    cli.cmd_oof(SimpleNamespace(config=str(cfg_path)))
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("oof", {}).get("path") == "oof_primary.csv"


def test_train_regime_groupby_missing_vol_regime() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([0.0, 1.0])
    with pytest.raises(ValueError, match="vol_regime"):
        cli._train_regime_groupby_bundles(
            {"indicators": []},
            X=X,
            y=y,
            t0=np.zeros(2, dtype=np.int64),
            t1=np.ones(2, dtype=np.int64),
            task="binary",
            out_dir=Path("."),
            symbol="S",
            seed=0,
            xgb_params={},
        )


def test_train_regime_groupby_all_nan_regime() -> None:
    X = pd.DataFrame({"vol_regime": [np.nan, np.nan], "a": [1.0, 2.0]})
    y = pd.Series([0.0, 1.0])
    with pytest.raises(ValueError, match="non-NaN vol_regime"):
        cli._train_regime_groupby_bundles(
            {"indicators": []},
            X=X,
            y=y,
            t0=np.zeros(2, dtype=np.int64),
            t1=np.ones(2, dtype=np.int64),
            task="binary",
            out_dir=Path("."),
            symbol="S",
            seed=0,
            xgb_params={},
        )


def test_train_regime_groupby_class_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "rg"
    X = pd.DataFrame({"vol_regime": [0.0, 0.0, 1.0, 1.0], "f": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    t0 = np.arange(4, dtype=np.int64)
    t1 = t0 + 1
    calls = {"n": 0}

    def fake_train(*_a, **_k):
        calls["n"] += 1
        m = MagicMock()
        if calls["n"] == 1:
            m.classes_ = np.array([0.0, 1.0])
        else:
            m.classes_ = np.array([0.0])
        return m

    monkeypatch.setattr(cli, "train_xgb_classifier", fake_train)
    monkeypatch.setattr(cli, "save_model_bundle", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_weights_for_training", lambda *a, **k: None)
    with pytest.raises(ValueError, match="classes_"):
        cli._train_regime_groupby_bundles(
            {"indicators": [], "regime": {"enabled": True, "groupby": True}},
            X=X,
            y=y,
            t0=t0,
            t1=t1,
            task="binary",
            out_dir=out,
            symbol="S",
            seed=0,
            xgb_params={},
        )


def test_train_regime_groupby_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "rg2"
    X = pd.DataFrame({"vol_regime": [0.0, 0.0, 1.0, 1.0], "f": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    t0 = np.arange(4, dtype=np.int64)
    t1 = t0 + 1

    def fake_train(*_a, **_k):
        m = MagicMock()
        m.classes_ = np.array([0.0, 1.0])
        return m

    monkeypatch.setattr(cli, "train_xgb_classifier", fake_train)
    monkeypatch.setattr(cli, "save_model_bundle", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_weights_for_training", lambda *a, **k: None)
    cli._train_regime_groupby_bundles(
        {"indicators": [], "regime": {"enabled": True, "groupby": True}},
        X=X,
        y=y,
        t0=t0,
        t1=t1,
        task="binary",
        out_dir=out,
        symbol="S",
        seed=0,
        xgb_params={},
    )
    assert (out / "regime_index.json").is_file()


def test_cmd_train_groupby_symbol_calls_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_train_save(cfg, *, symbol, out_dir):
        called.append(symbol)

    monkeypatch.setattr(cli, "_train_and_save", fake_train_save)
    cfg_path = tmp_path / "t.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "symbol": "A",
                "symbols": ["X", "Y"],
                "groupby_symbol": True,
                "model": {"out_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )
    cli.cmd_train(SimpleNamespace(config=str(cfg_path)))
    assert called == ["X", "Y"]


def test_cmd_wf_tune_value_errors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "w.yaml"
    cfg_path.write_text(yaml.safe_dump({"wf_tuner": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="wf_tuner"):
        cli.cmd_wf_tune(SimpleNamespace(config=str(cfg_path)))
    cfg_path.write_text(yaml.safe_dump({"wf_tuner": {"grid": [{"a": 1}]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="metric"):
        cli.cmd_wf_tune(SimpleNamespace(config=str(cfg_path)))


def test_cmd_wf_tune_writes_csv_and_prints_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    rows = pd.DataFrame(
        {
            "grid_id": ["g1", "g1"],
            "wf_fold": [0, 1],
            "score": [0.5, 0.6],
            "grid_params_json": ['{"a": 1}', '{"a": 1}'],
        }
    )

    def fake_tune(*_a, **_k):
        return rows

    monkeypatch.setattr("aprilalgo.tuner.ml_walk_forward.ml_walk_forward_tune", fake_tune)
    out = tmp_path / "wfout"
    out.mkdir()
    cfg_path = tmp_path / "wf.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {"out_dir": str(out)},
                "wf_tuner": {"metric": "accuracy", "grid": [{"max_depth": 2}], "n_folds": 2},
                "walk_forward": {"n_folds": 2},
            }
        ),
        encoding="utf-8",
    )
    cli.cmd_wf_tune(SimpleNamespace(config=str(cfg_path)))
    assert (out / "wf_tune_results.csv").is_file()
    out_cap = capsys.readouterr().out
    assert "Top grid" in out_cap
    assert "g1" in out_cap


def test_cmd_shap_per_regime_empty_result_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "md"
    model_dir.mkdir()
    (model_dir / "regime_index.json").write_text(
        json.dumps({"buckets": {"0": "regime_0"}, "default": "regime_0"}), encoding="utf-8"
    )
    cfg_path = tmp_path / "s.yaml"
    cfg_path.write_text(yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(model_dir)}}), encoding="utf-8")

    bundle = MagicMock()
    bundle.meta = {"symbol": "S"}
    X = pd.DataFrame({"vol_regime": [0.0], "f": [1.0]})

    monkeypatch.setattr("aprilalgo.ml.explain.load_regime_bundles_shap", lambda p: {"0": bundle})

    def _pxy_shap(_cfg, symbol=None):
        return X, pd.Series([0]), np.array([0]), np.array([1]), "binary"

    monkeypatch.setattr(cli, "_prepare_xy", _pxy_shap)
    monkeypatch.setattr("aprilalgo.ml.explain.shap_values_per_regime", lambda *a, **k: {})

    ns = SimpleNamespace(
        config=str(cfg_path),
        model_dir=None,
        max_samples=5,
        per_regime=True,
        output=str(tmp_path / "v.csv"),
        importance_output=str(tmp_path / "i.csv"),
    )
    with pytest.raises(ValueError, match="no rows matched"):
        cli.cmd_shap(ns)


def test_predict_regime_routed_feature_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    md = tmp_path / "m"
    md.mkdir()
    (md / "regime_index.json").write_text(
        json.dumps({"buckets": {"0": "regime_0", "1": "regime_1"}, "default": "regime_0"}), encoding="utf-8"
    )

    b0 = MagicMock()
    b0.feature_names = ["a", "vol_regime"]
    b0.classes_ = np.array([0.0, 1.0])
    b0.task = "binary"
    b1 = MagicMock()
    b1.feature_names = ["a"]
    b1.classes_ = np.array([0.0, 1.0])
    b1.task = "binary"

    def fake_load(p):
        if str(p).endswith("regime_0") or p.name == "regime_0":
            return b0
        return b1

    monkeypatch.setattr(cli, "load_model_bundle", fake_load)
    X = pd.DataFrame({"vol_regime": [0.0], "a": [1.0]})

    def _pxy_feat(_cfg, symbol=None):
        return X, pd.Series([0.0]), np.array([0]), np.array([1]), "binary"

    monkeypatch.setattr(cli, "_prepare_xy", _pxy_feat)
    with pytest.raises(ValueError, match="feature_names"):
        cli._predict_regime_routed(md, {"symbol": "S"}, sym="S")


def test_predict_regime_routed_missing_vol_regime_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    md = tmp_path / "m2"
    md.mkdir()
    (md / "regime_index.json").write_text(
        json.dumps({"buckets": {"0": "regime_0"}, "default": "regime_0"}), encoding="utf-8"
    )
    b0 = MagicMock()
    b0.feature_names = ["a"]
    b0.classes_ = np.array([0.0, 1.0])
    b0.task = "binary"
    monkeypatch.setattr(cli, "load_model_bundle", lambda p: b0)
    X = pd.DataFrame({"a": [1.0]})

    def _pxy_vol(_cfg, symbol=None):
        return X, pd.Series([0.0]), np.array([0]), np.array([1]), "binary"

    monkeypatch.setattr(cli, "_prepare_xy", _pxy_vol)
    with pytest.raises(ValueError, match="vol_regime"):
        cli._predict_regime_routed(md, {}, sym="S")


def test_main_bars_subcommand(tmp_path: Path) -> None:
    inp = tmp_path / "i.csv"
    pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=15, freq="min"),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
        }
    ).to_csv(inp, index=False)
    out = tmp_path / "o.csv"
    cli.main(
        [
            "bars",
            "--input",
            str(inp),
            "--bar-type",
            "volume",
            "--threshold",
            "500",
            "--output",
            str(out),
        ]
    )
    assert out.is_file()


def test_main_missing_subcommand_exits() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_cmd_evaluate_prints_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    X = pd.DataFrame({"a": [0.0, 1.0]})
    y = pd.Series([0, 1])
    t0 = np.array([0, 1], dtype=np.int64)
    t1 = np.array([2, 3], dtype=np.int64)

    monkeypatch.setattr(cli, "_prepare_xy", lambda c, symbol=None: (X, y, t0, t1, "binary"))
    monkeypatch.setattr(cli, "_xgb_estimator_factory", lambda c, t: (lambda: MagicMock()))
    monkeypatch.setattr(cli, "_weights_for_training", lambda *a, **k: None)

    def fake_eval(*_a, **_k):
        return {"mean": {"accuracy": 0.5}, "folds_df": pd.DataFrame({"accuracy": [0.5, 0.5]})}

    monkeypatch.setattr(cli, "purged_cv_evaluate", fake_eval)
    cfg_path = tmp_path / "ev.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(tmp_path)}}, default_flow_style=False),
        encoding="utf-8",
    )
    cli.cmd_evaluate(SimpleNamespace(config=str(cfg_path)))
    out = capsys.readouterr().out
    assert "accuracy" in out
    data = json.loads(out)
    assert data["mean"]["accuracy"] == 0.5
    assert len(data["folds_df"]) == 2


def test_cmd_importance_writes_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    out_dir = tmp_path / "imp"
    out_dir.mkdir()
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    y = pd.Series([0, 1, 0])
    t0 = np.arange(3, dtype=np.int64)
    t1 = t0 + 1
    clf = MagicMock()

    monkeypatch.setattr(cli, "_prepare_xy", lambda c, symbol=None: (X, y, t0, t1, "binary"))
    monkeypatch.setattr(cli, "train_xgb_classifier", lambda *a, **k: clf)
    monkeypatch.setattr(
        cli,
        "xgb_importance_table",
        lambda c, feature_names: pd.DataFrame({"feature": ["a"], "gain": [1.0]}),
    )
    monkeypatch.setattr(
        cli,
        "permutation_importance_table",
        lambda *a, **k: pd.DataFrame({"feature": ["a"], "importance": [0.5]}),
    )
    cfg_path = tmp_path / "imp.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(out_dir)}, "importance_repeats": 2}),
        encoding="utf-8",
    )
    cli.cmd_importance(SimpleNamespace(config=str(cfg_path)))
    assert (out_dir / "importance_gain.csv").is_file()
    assert (out_dir / "importance_permutation.csv").is_file()
    cap = capsys.readouterr().out
    assert "a" in cap


def test_cmd_predict_single_bundle_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mdir = tmp_path / "single_model"
    mdir.mkdir()
    bundle = MagicMock()
    bundle.meta = {"symbol": "S"}
    bundle.classes_ = np.array([0.0, 1.0])
    bundle.predict_proba = MagicMock(return_value=np.array([[0.7, 0.3], [0.2, 0.8]]))
    bundle.predict = MagicMock(return_value=np.array([0.0, 1.0]))
    X = pd.DataFrame({"f": [1.0, 2.0]})
    y = pd.Series([0.0, 1.0])

    monkeypatch.setattr(cli, "load_model_bundle", lambda p: bundle)
    monkeypatch.setattr(cli, "_prepare_xy", lambda cfg, symbol: (X, y, np.array([0, 1]), np.array([2, 3]), "binary"))
    cfg_path = tmp_path / "pred.yaml"
    cfg_path.write_text(yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(mdir)}}), encoding="utf-8")
    outp = tmp_path / "pred_out.csv"
    cli.cmd_predict(SimpleNamespace(config=str(cfg_path), model_dir=None, output=str(outp)))
    df = pd.read_csv(outp)
    assert len(df) == 2
    assert any(str(c).startswith("proba_") for c in df.columns)
    bundle.predict_proba.assert_called_once()


def test_cmd_shap_default_writes_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    mdir = tmp_path / "shap_m"
    mdir.mkdir()
    bundle = MagicMock()
    bundle.meta = {"symbol": "S"}
    X = pd.DataFrame({"a": np.arange(5, dtype=float)})

    monkeypatch.setattr(cli, "_load_reference_bundle", lambda p: bundle)

    def _pxy_shap_def(_cfg, symbol=None):
        return X, pd.Series([0] * 5), np.arange(5), np.arange(5) + 1, "binary"

    monkeypatch.setattr(cli, "_prepare_xy", _pxy_shap_def)
    monkeypatch.setattr(
        cli,
        "shap_values_table",
        lambda b, X, max_samples=300: pd.DataFrame({"sample_idx": [0], "feature": ["a"], "shap_value": [0.1]}),
    )
    monkeypatch.setattr(
        cli,
        "shap_importance_table",
        lambda b, X, max_samples=300: pd.DataFrame({"feature": ["a"], "mean_abs": [0.1]}),
    )
    cfg_path = tmp_path / "sh.yaml"
    cfg_path.write_text(yaml.safe_dump({"symbol": "S", "model": {"out_dir": str(mdir)}}), encoding="utf-8")
    vout = tmp_path / "sv.csv"
    iout = tmp_path / "si.csv"
    ns = SimpleNamespace(
        config=str(cfg_path),
        model_dir=None,
        max_samples=50,
        per_regime=False,
        output=str(vout),
        importance_output=str(iout),
    )
    cli.cmd_shap(ns)
    assert vout.is_file() and iout.is_file()
    assert "SHAP" in capsys.readouterr().out


def test_train_and_save_single_bundle_with_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "bundle1"
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0, 1])
    t0 = np.array([0, 1], dtype=np.int64)
    t1 = np.array([2, 3], dtype=np.int64)
    clf = MagicMock()
    clf.classes_ = np.array([0, 1])
    captured: dict[str, Any] = {}

    def fake_save(od, c, *, feature_names, task, indicator_config, extra_meta=None):
        captured["extra_meta"] = dict(extra_meta or {})
        captured["feature_names"] = list(feature_names)

    monkeypatch.setattr(cli, "_regime_groupby_training", lambda c: False)
    monkeypatch.setattr(cli, "_prepare_xy", lambda cfg, symbol=None: (X, y, t0, t1, "binary"))
    monkeypatch.setattr(cli, "train_xgb_classifier", lambda *a, **k: clf)
    monkeypatch.setattr(cli, "save_model_bundle", fake_save)
    monkeypatch.setattr(cli, "_weights_for_training", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "information_bars_meta_from_cfg",
        lambda c: {"enabled": True, "bar_type": "volume", "threshold": 100.0, "source_timeframe": "daily"},
    )
    cfg = {
        "symbol": "Z",
        "timeframe": "daily",
        "indicators": [{"type": "sma", "params": {"period": 2}}],
        "random_state": 7,
        "model": {"xgb": {"max_depth": 2}},
        "regime": {"enabled": False},
    }
    cli._train_and_save(cfg, symbol="Z", out_dir=out)
    assert captured["feature_names"] == ["x"]
    assert captured["extra_meta"]["symbol"] == "Z"
    assert captured["extra_meta"]["information_bars"]["enabled"] is True
    assert captured["extra_meta"]["regime"]["enabled"] is False


def test_load_reference_bundle_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[Path] = []

    def rec(p):
        paths.append(Path(p))
        b = MagicMock()
        b.meta = {}
        return b

    monkeypatch.setattr(cli, "load_model_bundle", rec)
    root = tmp_path / "r1"
    root.mkdir()
    assert cli._load_reference_bundle(root) is not None
    assert paths == [root]

    paths.clear()
    sub = root / "regime_0"
    sub.mkdir()
    (root / "regime_index.json").write_text(
        json.dumps({"buckets": {"0": "regime_0"}, "default": "regime_0"}), encoding="utf-8"
    )
    cli._load_reference_bundle(root)
    assert any(p.name == "regime_0" for p in paths)


def test_predict_regime_routed_full_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    md = tmp_path / "mr"
    md.mkdir()
    (md / "regime_index.json").write_text(
        json.dumps({"buckets": {"0": "regime_0", "1": "regime_1"}, "default": "regime_0"}), encoding="utf-8"
    )
    b0 = MagicMock()
    b0.meta = {}
    b0.feature_names = ["a", "vol_regime"]
    b0.classes_ = np.array([0.0, 1.0])
    b0.task = "binary"
    b0.predict_proba = lambda Xi: np.array([[0.8, 0.2]] * len(Xi))
    b0.predict = lambda Xi: np.zeros(len(Xi), dtype=float)
    b1 = MagicMock()
    b1.meta = {}
    b1.feature_names = ["a", "vol_regime"]
    b1.classes_ = np.array([0.0, 1.0])
    b1.task = "binary"
    b1.predict_proba = lambda Xi: np.array([[0.1, 0.9]] * len(Xi))
    b1.predict = lambda Xi: np.ones(len(Xi), dtype=float)

    def fake_load(p: Path):
        p = Path(p)
        return b1 if p.name == "regime_1" else b0

    monkeypatch.setattr(cli, "load_model_bundle", fake_load)
    X = pd.DataFrame({"vol_regime": [0.0, 1.0, np.nan], "a": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 0.0])

    def _pxy_loop(_cfg, symbol=None):
        return X, y, np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64), "binary"

    monkeypatch.setattr(cli, "_prepare_xy", _pxy_loop)
    Xo, yo, pred, proba, classes = cli._predict_regime_routed(md, {"symbol": "S"}, sym="S")
    assert len(Xo) == 3
    assert proba.shape == (3, 2)
    assert list(classes) == [0.0, 1.0]
    assert pred[0] == 0.0 and pred[1] == 1.0


def test_wf_tune_grid_from_yaml_branches() -> None:
    assert cli._wf_tune_grid_from_yaml({"grid": [{"max_depth": 2}]}) == [{"max_depth": 2}]
    out = cli._wf_tune_grid_from_yaml({"grid": {"max_depth": [2, 3], "learning_rate": [0.1]}})
    assert len(out) == 2
    with pytest.raises(ValueError, match="wf_tuner.grid is required"):
        cli._wf_tune_grid_from_yaml({})
    with pytest.raises(ValueError, match="non-empty"):
        cli._wf_tune_grid_from_yaml({"grid": []})
    with pytest.raises(ValueError, match="list or dict"):
        cli._wf_tune_grid_from_yaml({"grid": "nope"})


def test_cmd_walk_forward_prints_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=80, freq="D"),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": np.linspace(100, 110, 80),
            "volume": 1e6,
        }
    )
    monkeypatch.setattr(cli, "load_ohlcv_for_ml", lambda cfg, sym: df)
    cfg_path = tmp_path / "wf.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symbol": "S", "walk_forward": {"n_folds": 3, "min_train": 20}}), encoding="utf-8"
    )
    cli.cmd_walk_forward(SimpleNamespace(config=str(cfg_path)))
    data = json.loads(capsys.readouterr().out)
    assert data["n_bars"] == 80
    assert "splits" in data and len(data["splits"]) >= 1
