"""CLI for ML train / evaluate / oof / predict / importance / shap / walk-forward / bars.

Usage::

    uv run python -m aprilalgo.cli train --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli evaluate --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli oof --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli predict --config configs/ml/default.yaml --output predictions.csv
    uv run python -m aprilalgo.cli importance --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli walk-forward --config configs/ml/default.yaml
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aprilalgo.data import (
    build_dollar_bars,
    build_tick_bars,
    build_volume_bars,
    information_bars_meta_from_cfg,
    load_ohlcv_for_ml,
    load_price_data,
)
from aprilalgo.labels.targets import build_triple_barrier_targets
from aprilalgo.ml.evaluator import purged_cv_evaluate
from aprilalgo.ml.oof import compute_primary_oof
from aprilalgo.ml.explain import shap_importance_table, shap_values_table
from aprilalgo.ml.features import build_feature_matrix
from aprilalgo.ml.importance import (
    permutation_importance_table,
    xgb_importance_table,
)
from aprilalgo.ml.sampling import sequential_bootstrap_sample, uniqueness_weights
from aprilalgo.ml.trainer import Task, load_model_bundle, save_model_bundle, train_xgb_classifier
from aprilalgo.tuner.walk_forward import walk_forward_splits, walk_forward_summary


def _load_cfg(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _symbols_for_cfg(cfg: dict[str, Any]) -> list[str]:
    syms = cfg.get("symbols")
    if syms:
        return [str(s) for s in syms]
    return [str(cfg["symbol"])]


def _weights_for_training(
    cfg: dict[str, Any], t0: np.ndarray, t1: np.ndarray
) -> np.ndarray | None:
    """Return per-row ``sample_weight`` for :func:`~aprilalgo.ml.trainer.train_xgb_classifier`.

    *t0* / *t1* are label interval indices aligned with the filtered training matrix.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    sam = cfg.get("sampling")
    if sam is None:
        return None
    strategy = str(sam.get("strategy", "none")).lower()
    if strategy in ("", "none"):
        return None
    if strategy == "uniqueness":
        return uniqueness_weights(t0, t1)
    if strategy == "bootstrap":
        n = len(t0)
        raw_nd = sam.get("n_draw")
        n_draw = n if raw_nd is None else int(raw_nd)
        rs = int(sam.get("random_state", cfg.get("random_state", 42)))
        idx = sequential_bootstrap_sample(t0, t1, n_draw=n_draw, random_state=rs)
        counts = np.bincount(idx, minlength=n).astype(np.float64)
        tot = float(counts.sum())
        if tot <= 0.0:
            return np.ones(n, dtype=np.float64)
        return counts * (n / tot)
    raise ValueError(
        f"Unknown sampling.strategy: {strategy!r} (expected none, uniqueness, or bootstrap)"
    )


def _sampling_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    """Persisted under ``meta.json`` key ``sampling`` (defaults match absent YAML)."""
    sam = cfg.get("sampling") or {}
    strat = str(sam.get("strategy", "none")).lower()
    out: dict[str, Any] = {"strategy": strat}
    if strat == "bootstrap":
        rs = int(sam.get("random_state", cfg.get("random_state", 42)))
        out["random_state"] = rs
        nd = sam.get("n_draw")
        out["n_draw"] = None if nd is None else int(nd)
    elif "random_state" in sam:
        out["random_state"] = int(sam["random_state"])
    return out


def _cfg_for_inference(cfg: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """Prefer bar recipe from saved ``meta.json`` so predict/SHAP match training."""
    ib = bundle.meta.get("information_bars")
    if not ib:
        return cfg
    merged = {**cfg, "information_bars": ib}
    return merged


def _prepare_xy(
    cfg: dict[str, Any],
    *,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, Task]:
    sym = symbol if symbol is not None else cfg["symbol"]
    df = load_ohlcv_for_ml(cfg, str(sym))

    b = cfg["triple_barrier"]
    targets = build_triple_barrier_targets(
        df,
        upper_pct=float(b["upper_pct"]),
        lower_pct=float(b["lower_pct"]),
        vertical_bars=int(b["vertical_bars"]),
    )

    task: Task = cfg.get("task", "binary")
    if task == "binary":
        y = targets["label_binary"]
    else:
        y = targets["label_multiclass"]

    X = build_feature_matrix(df, indicator_config=cfg["indicators"])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    t0 = targets.loc[mask, "label_t0"].to_numpy(dtype=np.int64)
    t1 = targets.loc[mask, "label_t1"].to_numpy(dtype=np.int64)
    return X, y, t0, t1, task


def _train_and_save(cfg: dict[str, Any], *, symbol: str, out_dir: Path) -> None:
    X, y, t0, t1, task = _prepare_xy(cfg, symbol=symbol)
    seed = int(cfg.get("random_state", 42))
    xgb_params = cfg.get("model", {}).get("xgb", {})
    sw = _weights_for_training(cfg, t0, t1)
    clf = train_xgb_classifier(
        X,
        y,
        task=task,
        random_state=seed,
        xgb_params=xgb_params,
        sample_weight=sw,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": cfg.get("timeframe"),
        "groupby_symbol": cfg.get("groupby_symbol", False),
        "sampling": _sampling_meta(cfg),
    }
    ib_meta = information_bars_meta_from_cfg(cfg)
    if ib_meta is not None:
        extra["information_bars"] = ib_meta
    save_model_bundle(
        out_dir,
        clf,
        feature_names=list(X.columns),
        task=task,
        indicator_config=cfg["indicators"],
        extra_meta=extra,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    out_root = Path(cfg["model"]["out_dir"])
    symbols = _symbols_for_cfg(cfg)
    if cfg.get("groupby_symbol"):
        for sym in symbols:
            _train_and_save(cfg, symbol=sym, out_dir=out_root / sym)
            print(f"Saved model bundle for {sym} -> {(out_root / sym).resolve()}")
    else:
        sym = str(cfg["symbol"])
        _train_and_save(cfg, symbol=sym, out_dir=out_root)
        print(f"Saved model bundle to {out_root.resolve()}")


def _xgb_estimator_factory(cfg: dict[str, Any], task: Task) -> Callable[[], Any]:
    """Return a zero-arg factory usable by :func:`~aprilalgo.ml.evaluator.purged_cv_evaluate` / OOF."""
    seed = int(cfg.get("random_state", 42))
    xgb_params = cfg.get("model", {}).get("xgb", {})

    def factory() -> Any:
        from xgboost import XGBClassifier

        if task == "binary":
            return XGBClassifier(
                objective="binary:logistic",
                random_state=seed,
                n_estimators=int(xgb_params.get("n_estimators", 50)),
                max_depth=int(xgb_params.get("max_depth", 3)),
                learning_rate=float(xgb_params.get("learning_rate", 0.1)),
            )
        return XGBClassifier(
            objective="multi:softprob",
            random_state=seed,
            n_estimators=int(xgb_params.get("n_estimators", 50)),
            max_depth=int(xgb_params.get("max_depth", 3)),
            learning_rate=float(xgb_params.get("learning_rate", 0.1)),
        )

    return factory


def _eval_result_for_json(res: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in res.items() if k != "folds_df"}
    fd = res.get("folds_df")
    if isinstance(fd, pd.DataFrame) and not fd.empty:
        out["folds_df"] = fd.to_dict(orient="records")
    elif isinstance(fd, pd.DataFrame):
        out["folds_df"] = []
    return out


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    X, y, t0, t1, task = _prepare_xy(cfg)
    cv = cfg.get("cv", {})
    n_splits = int(cv.get("n_splits", 3))
    embargo = int(cv.get("embargo", 0))
    factory = _xgb_estimator_factory(cfg, task)

    res = purged_cv_evaluate(
        factory,
        X,
        y,
        sample_t0=t0,
        sample_t1=t1,
        n_splits=n_splits,
        embargo=embargo,
    )
    print(json.dumps(_eval_result_for_json(res), indent=2, default=str))


def cmd_oof(args: argparse.Namespace) -> None:
    """Write purged OOF predictions to ``oof_primary.csv`` under ``model.out_dir``."""
    cfg = _load_cfg(args.config)
    out_dir = Path(cfg["model"]["out_dir"])
    sym = str(cfg.get("symbol") or "")
    if not sym:
        raise ValueError("config.symbol is required for oof")
    X, y, t0, t1, task = _prepare_xy(cfg, symbol=sym)
    cv = cfg.get("cv", {})
    n_splits = int(cv.get("n_splits", 3))
    embargo = int(cv.get("embargo", 0))
    sw = _weights_for_training(cfg, t0, t1)
    factory = _xgb_estimator_factory(cfg, task)
    oof_df = compute_primary_oof(
        X,
        y,
        t0,
        t1,
        factory=factory,
        n_splits=n_splits,
        embargo=embargo,
        task=task,
        sample_weight=sw,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "oof_primary.csv"
    out_path = out_dir / fname
    oof_df.to_csv(out_path, index=False)
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["oof"] = {"path": fname}
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"Wrote {len(oof_df)} rows to {out_path.resolve()}")


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["model"]["out_dir"])
    bundle = load_model_bundle(model_dir)
    sym = bundle.meta.get("symbol") or cfg.get("symbol")
    if not sym:
        raise ValueError("Config needs symbol or model meta.json must include symbol")
    cfg_inf = _cfg_for_inference(cfg, bundle)
    X, y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
    proba = bundle.predict_proba(X)
    pred = bundle.predict(X)
    out = pd.DataFrame({"y_true": y.to_numpy(), "pred": pred})
    classes = [float(c) for c in np.asarray(bundle.classes_).ravel()]
    for j, c in enumerate(classes):
        out[f"proba_{c}"] = proba[:, j]
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"Wrote {len(out)} rows to {outp.resolve()}")


def cmd_importance(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    X, y, t0, t1, task = _prepare_xy(cfg)
    seed = int(cfg.get("random_state", 42))
    xgb_params = cfg.get("model", {}).get("xgb", {})
    sw = _weights_for_training(cfg, t0, t1)
    clf = train_xgb_classifier(
        X,
        y,
        task=task,
        random_state=seed,
        xgb_params=xgb_params,
        sample_weight=sw,
    )
    gain = xgb_importance_table(clf, feature_names=list(X.columns))
    perm = permutation_importance_table(
        clf, X, y, n_repeats=int(cfg.get("importance_repeats", 5)), random_state=seed
    )
    out_dir = Path(cfg["model"].get("out_dir", "outputs/ml"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gain.to_csv(out_dir / "importance_gain.csv", index=False)
    perm.to_csv(out_dir / "importance_permutation.csv", index=False)
    print(gain.head(15).to_string(index=False))
    print("--- permutation (top 15) ---")
    print(perm.head(15).to_string(index=False))
    print(f"Wrote CSVs under {out_dir.resolve()}")


def cmd_walk_forward(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    wf = cfg.get("walk_forward", {})
    sym = str(cfg["symbol"])
    df = load_ohlcv_for_ml(cfg, sym)
    n = len(df)
    splits = list(
        walk_forward_splits(
            n,
            n_folds=int(wf.get("n_folds", 4)),
            min_train=int(wf.get("min_train", 50)),
            test_size=wf.get("test_size"),
        )
    )
    report = [
        {
            "fold": i,
            "train_start": int(tr[0]),
            "train_end": int(tr[-1]),
            "test_start": int(te[0]),
            "test_end": int(te[-1]),
            "train_size": int(tr.size),
            "test_size": int(te.size),
            "test_return": float(
                (float(df.iloc[te[-1]]["close"]) / float(df.iloc[te[0]]["close"])) - 1.0
            ),
        }
        for i, (tr, te) in enumerate(splits)
    ]
    print(
        json.dumps(
            {
                "n_bars": n,
                "summary": walk_forward_summary(n, splits),
                "splits": report,
            },
            indent=2,
        )
    )


def cmd_shap(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["model"]["out_dir"])
    bundle = load_model_bundle(model_dir)
    sym = bundle.meta.get("symbol") or cfg.get("symbol")
    if not sym:
        raise ValueError("Config needs symbol or model meta.json must include symbol")
    cfg_inf = _cfg_for_inference(cfg, bundle)
    X, _y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
    vals = shap_values_table(bundle, X, max_samples=int(args.max_samples))
    imp = shap_importance_table(bundle, X, max_samples=int(args.max_samples))
    out_vals = Path(args.output)
    out_imp = Path(args.importance_output)
    out_vals.parent.mkdir(parents=True, exist_ok=True)
    out_imp.parent.mkdir(parents=True, exist_ok=True)
    vals.to_csv(out_vals, index=False)
    imp.to_csv(out_imp, index=False)
    print(f"Wrote SHAP values to {out_vals.resolve()}")
    print(f"Wrote SHAP importance to {out_imp.resolve()}")


def cmd_bars(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    df = pd.read_csv(inp)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    kind = str(args.bar_type)
    threshold = float(args.threshold)
    if kind == "tick":
        out = build_tick_bars(df, threshold=int(threshold))
    elif kind == "volume":
        out = build_volume_bars(df, threshold=threshold)
    elif kind == "dollar":
        out = build_dollar_bars(df, threshold=threshold)
    else:
        raise ValueError(f"Unknown bar type: {kind}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} {kind} bars to {out_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aprilalgo.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train XGBoost and save bundle")
    p_train.add_argument("--config", type=str, required=True)
    p_train.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Purged CV metrics (JSON to stdout)")
    p_ev.add_argument("--config", type=str, required=True)
    p_ev.set_defaults(func=cmd_evaluate)

    p_oof = sub.add_parser(
        "oof",
        help="Purged out-of-fold primary predictions -> oof_primary.csv under model.out_dir",
    )
    p_oof.add_argument("--config", type=str, required=True)
    p_oof.set_defaults(func=cmd_oof)

    p_pred = sub.add_parser("predict", help="Run saved model on config data; CSV out")
    p_pred.add_argument("--config", type=str, required=True)
    p_pred.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override model directory (default: model.out_dir from config)",
    )
    p_pred.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.csv",
    )
    p_pred.set_defaults(func=cmd_predict)

    p_imp = sub.add_parser("importance", help="Gain + permutation tables")
    p_imp.add_argument("--config", type=str, required=True)
    p_imp.set_defaults(func=cmd_importance)

    p_shap = sub.add_parser("shap", help="Compute SHAP values + feature importance CSVs")
    p_shap.add_argument("--config", type=str, required=True)
    p_shap.add_argument("--model-dir", type=str, default=None)
    p_shap.add_argument("--output", type=str, default="outputs/ml/shap_values.csv")
    p_shap.add_argument(
        "--importance-output",
        type=str,
        default="outputs/ml/shap_importance.csv",
    )
    p_shap.add_argument("--max-samples", type=int, default=300)
    p_shap.set_defaults(func=cmd_shap)

    p_wf = sub.add_parser(
        "walk-forward",
        help="Print walk-forward index ranges for config symbol (JSON)",
    )
    p_wf.add_argument("--config", type=str, required=True)
    p_wf.set_defaults(func=cmd_walk_forward)

    p_bars = sub.add_parser("bars", help="Build information-driven bars from OHLCV CSV")
    p_bars.add_argument("--input", type=str, required=True)
    p_bars.add_argument("--bar-type", type=str, choices=["tick", "volume", "dollar"], required=True)
    p_bars.add_argument("--threshold", type=float, required=True)
    p_bars.add_argument("--output", type=str, required=True)
    p_bars.set_defaults(func=cmd_bars)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
