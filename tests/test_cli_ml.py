"""Smoke-test ML CLI subcommands (fixture config)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "ml" / "default.yaml"

assert _CFG.is_file(), (
    f"Required ML fixture config missing: {_CFG}. "
    "If intentionally removed, migrate these tests to an in-repo fixture config."
)


def test_cli_train_predict_roundtrip(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pred_csv = tmp_path / "p.csv"

    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "predict",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--output",
            str(pred_csv),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert pred_csv.is_file() and pred_csv.stat().st_size > 0


def test_cli_train_predict_roundtrip_information_bars(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_ib"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["information_bars"] = {
        "enabled": True,
        "bar_type": "tick",
        "threshold": 2,
    }
    cfg_path = tmp_path / "ml_ib.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pred_csv = tmp_path / "p_ib.csv"

    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("information_bars", {}).get("enabled") is True
    assert meta["information_bars"]["bar_type"] == "tick"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "predict",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--output",
            str(pred_csv),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert pred_csv.is_file() and pred_csv.stat().st_size > 0


def test_cli_train_persists_sampling_meta(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_sampling"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml_sampling.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("sampling", {}).get("strategy") == "none"


def test_cli_oof_writes_csv(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_oof"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml_oof.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "oof", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    oof_csv = out_bundle / "oof_primary.csv"
    assert oof_csv.is_file() and oof_csv.stat().st_size > 0
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("oof", {}).get("path") == "oof_primary.csv"


def test_cli_meta_train_writes_artifacts(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_meta"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml_meta.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for cmd in ("train", "oof"):
        subprocess.run(
            [sys.executable, "-m", "aprilalgo.cli", cmd, "--config", str(cfg_path)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    # Fixture can yield degenerate z (primary always matches y); break symmetry so meta fits.
    oof_csv = out_bundle / "oof_primary.csv"
    oof_df = pd.read_csv(oof_csv)
    npert = max(20, len(oof_df) // 3)
    sl = oof_df.index[:npert]
    oof_df.loc[sl, "oof_pred"] = 1.0 - oof_df.loc[sl, "y"].astype(float)
    oof_df.to_csv(oof_csv, index=False)
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "meta-train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out_bundle / "meta_logit.json").is_file()
    assert (out_bundle / "meta_oof.csv").is_file()
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    ml = meta.get("meta_logit", {})
    assert ml.get("enabled") is True
    assert ml.get("path") == "meta_logit.json"
    assert float(ml.get("threshold", 0)) == 0.5


def test_cli_meta_train_requires_oof(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_no_oof"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml_no_oof.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "meta-train",
            "--config",
            str(cfg_path),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "Missing" in err or "oof_primary" in err


def test_cli_train_uniqueness_persists_meta(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_uq"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["sampling"] = {"strategy": "uniqueness"}
    cfg_path = tmp_path / "ml_uq.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("sampling", {}).get("strategy") == "uniqueness"


def test_cli_train_bootstrap_persists_meta(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_bs"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["sampling"] = {"strategy": "bootstrap", "random_state": 123, "n_draw": 80}
    cfg_path = tmp_path / "ml_bs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    sam = meta.get("sampling", {})
    assert sam.get("strategy") == "bootstrap"
    assert sam.get("random_state") == 123
    assert sam.get("n_draw") == 80


def test_cli_evaluate_prints_json_metrics() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "evaluate", "--config", str(_CFG)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    assert "mean" in data
    assert "folds" in data


def test_cli_importance_writes_csvs(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_dir = tmp_path / "imp_out"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_dir)}
    cfg_path = tmp_path / "ml_importance.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "importance", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out_dir / "importance_gain.csv").is_file()
    assert (out_dir / "importance_permutation.csv").is_file()


def test_cli_walk_forward_json_has_summary() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "walk-forward",
            "--config",
            str(_CFG),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    assert "splits" in data and isinstance(data["splits"], list)
    assert "summary" in data
    assert {"n_splits", "coverage_pct", "mean_train_size", "mean_test_size"} <= set(
        data["summary"].keys()
    )


def test_cli_train_groupby_symbol_writes_per_symbol_dirs(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_root = tmp_path / "models"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_root)}
    cfg["groupby_symbol"] = True
    cfg["symbols"] = ["TEST"]
    cfg_path = tmp_path / "grouped.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out_root / "TEST" / "meta.json").is_file()
    assert (out_root / "TEST" / "xgboost.json").is_file()


def test_cli_bars_builds_output(tmp_path: Path) -> None:
    inp = _ROOT / "tests" / "fixtures" / "daily_data" / "TEST_daily.csv"
    outp = tmp_path / "bars.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "bars",
            "--input",
            str(inp),
            "--bar-type",
            "volume",
            "--threshold",
            "5000000",
            "--output",
            str(outp),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert outp.is_file() and outp.stat().st_size > 0


def test_cli_shap_writes_outputs(tmp_path: Path) -> None:
    import shap  # noqa: F401
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg_path = tmp_path / "ml.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    shap_csv = tmp_path / "shap_values.csv"
    imp_csv = tmp_path / "shap_importance.csv"
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "shap",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--output",
            str(shap_csv),
            "--importance-output",
            str(imp_csv),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert shap_csv.is_file() and imp_csv.is_file()


def test_cli_train_with_regime_enabled(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_regime"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 12,
        "n_buckets": 3,
        "use_hmm": False,
    }
    cfg_path = tmp_path / "ml_regime.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("regime", {}).get("enabled") is True
    assert int(meta["regime"]["window"]) == 12
    assert "vol_regime" in meta["feature_names"]
    assert "realized_vol" not in meta["feature_names"]


def test_cli_predict_regime_roundtrip(tmp_path: Path) -> None:
    from aprilalgo.cli import _cfg_for_inference
    from aprilalgo.ml.trainer import load_model_bundle

    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_regime_rt"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 9,
        "n_buckets": 4,
        "use_hmm": False,
    }
    cfg_path = tmp_path / "ml_regime_rt.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pred_csv = tmp_path / "pred_rt.csv"
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    bundle = load_model_bundle(out_bundle)
    cfg_bad = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg_bad["regime"] = {
        "enabled": True,
        "window": 1,
        "n_buckets": 2,
        "use_hmm": False,
    }
    merged = _cfg_for_inference(cfg_bad, bundle)
    assert int(merged["regime"]["window"]) == 9
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "predict",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--output",
            str(pred_csv),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert pred_csv.is_file() and pred_csv.stat().st_size > 0


def test_regime_off_matches_legacy(tmp_path: Path) -> None:
    from aprilalgo.cli import _prepare_xy

    base = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    base["model"] = {**base.get("model", {}), "out_dir": str(tmp_path / "unused")}
    base.pop("regime", None)
    cfg_no = dict(base)
    cfg_off = dict(base)
    cfg_off["regime"] = {
        "enabled": False,
        "window": 99,
        "n_buckets": 7,
        "use_hmm": False,
    }
    xa, ya, ta0, ta1, tska = _prepare_xy(cfg_no, symbol="TEST")
    xb, yb, tb0, tb1, tskb = _prepare_xy(cfg_off, symbol="TEST")
    pd.testing.assert_frame_equal(xa, xb)
    pd.testing.assert_series_equal(ya, yb)
    assert tska == tskb
    np.testing.assert_array_equal(ta0, tb0)
    np.testing.assert_array_equal(ta1, tb1)


def test_cli_train_groupby_regime_writes_per_bucket(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_gb"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_gb.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    idx = json.loads((out_bundle / "regime_index.json").read_text(encoding="utf-8"))
    assert "buckets" in idx and "default" in idx
    assert len(idx["buckets"]) >= 1
    for sub in idx["buckets"].values():
        assert (out_bundle / sub / "meta.json").is_file()


def test_per_bucket_meta_has_bucket(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_meta_bucket"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_pm.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    idx = json.loads((out_bundle / "regime_index.json").read_text(encoding="utf-8"))
    first_sub = next(iter(idx["buckets"].values()))
    meta = json.loads((out_bundle / first_sub / "meta.json").read_text(encoding="utf-8"))
    assert "bucket" in meta.get("regime", {})


def test_cli_predict_regime_index_routing(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_rt_idx"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_rt.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pred_csv = tmp_path / "p_rt.csv"
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "predict",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--output",
            str(pred_csv),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    df = pd.read_csv(pred_csv)
    assert len(df) > 0
    assert "pred" in df.columns


def test_per_regime_predict_deterministic(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_det"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_det.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for outp in (p1, p2):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "aprilalgo.cli",
                "predict",
                "--config",
                str(cfg_path),
                "--model-dir",
                str(out_bundle),
                "--output",
                str(outp),
            ],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    assert p1.read_bytes() == p2.read_bytes()


def test_cli_wf_tune_writes_csv(tmp_path: Path) -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_dir = tmp_path / "wf_tune_out"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_dir)}
    cfg["walk_forward"] = {"n_folds": 2, "min_train": 28, "test_size": 22}
    cfg["cv"] = {"n_splits": 2, "embargo": 0}
    cfg["model"]["xgb"] = {
        **cfg.get("model", {}).get("xgb", {}),
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.2,
    }
    cfg["wf_tuner"] = {
        "metric": "accuracy",
        "n_folds": 2,
        "grid": {"max_depth": [2, 3]},
    }
    cfg_path = tmp_path / "wf_tune_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "wf-tune",
            "--config",
            str(cfg_path),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    csv_p = out_dir / "wf_tune_results.csv"
    assert csv_p.is_file()
    df = pd.read_csv(csv_p)
    assert {"grid_id", "score", "wf_fold"} <= set(df.columns)
    assert len(df) >= 1


def test_cli_wf_tune_prints_top5(tmp_path: Path) -> None:
    from aprilalgo.tuner.ml_walk_forward import aggregate_grid

    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_dir = tmp_path / "wf_tune_top5"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_dir)}
    cfg["walk_forward"] = {"n_folds": 2, "min_train": 28, "test_size": 22}
    cfg["cv"] = {"n_splits": 2, "embargo": 0}
    cfg["model"]["xgb"] = {
        **cfg.get("model", {}).get("xgb", {}),
        "n_estimators": 8,
        "max_depth": 2,
        "learning_rate": 0.2,
    }
    cfg["wf_tuner"] = {
        "metric": "accuracy",
        "n_folds": 2,
        "grid": {"max_depth": [2, 3, 4], "learning_rate": [0.1, 0.2]},
    }
    cfg_path = tmp_path / "wf_tune_top5.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "wf-tune",
            "--config",
            str(cfg_path),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Top grid points by mean score" in proc.stdout
    r = pd.read_csv(out_dir / "wf_tune_results.csv")
    top = aggregate_grid(r, "score").sort_values("mean", ascending=False).head(5)
    assert len(top) == 5
    for gid in top["grid_id"]:
        assert gid in proc.stdout


def test_cli_shap_per_regime(tmp_path: Path) -> None:
    import shap  # noqa: F401
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    out_bundle = tmp_path / "bundle_shap_pr"
    cfg["model"] = {**cfg.get("model", {}), "out_dir": str(out_bundle)}
    cfg["regime"] = {
        "enabled": True,
        "window": 5,
        "n_buckets": 2,
        "use_hmm": False,
        "groupby": True,
    }
    cfg_path = tmp_path / "ml_shap_pr.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "aprilalgo.cli", "train", "--config", str(cfg_path)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aprilalgo.cli",
            "shap",
            "--config",
            str(cfg_path),
            "--model-dir",
            str(out_bundle),
            "--per-regime",
            "--max-samples",
            "80",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    vals_files = list(out_bundle.glob("regime_*_shap_values.csv"))
    imp_files = list(out_bundle.glob("regime_*_shap_importance.csv"))
    assert len(vals_files) >= 1
    assert len(imp_files) >= 1
    df0 = pd.read_csv(vals_files[0])
    assert {"feature", "sample_idx", "shap_value"} <= set(df0.columns)
