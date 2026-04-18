"""CLI for ML train / evaluate / oof / predict / importance / shap / walk-forward / wf-tune / bars.

Usage::

    uv run python -m aprilalgo.cli train --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli evaluate --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli oof --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli meta-train --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli predict --config configs/ml/default.yaml --output predictions.csv
    uv run python -m aprilalgo.cli importance --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli shap --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli walk-forward --config configs/ml/default.yaml
    uv run python -m aprilalgo.cli wf-tune --config configs/ml/default.yaml
"""

from __future__ import annotations

import argparse
import json
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
)
from aprilalgo.labels.meta_label import fit_meta_logit_purged
from aprilalgo.ml.evaluator import purged_cv_evaluate
from aprilalgo.ml.explain import shap_importance_table, shap_values_table
from aprilalgo.ml.importance import (
    permutation_importance_table,
    xgb_importance_table,
)
from aprilalgo.ml.meta_bundle import save_meta_logit_bundle
from aprilalgo.ml.oof import compute_primary_oof
from aprilalgo.ml.pipeline import (
    prepare_xy as _prepare_xy,
)
from aprilalgo.ml.pipeline import (
    weights_for_training as _weights_for_training,
)
from aprilalgo.ml.pipeline import (
    xgb_estimator_factory as _xgb_estimator_factory,
)
from aprilalgo.ml.trainer import (
    ModelBundle,
    Task,
    load_model_bundle,
    save_model_bundle,
    train_xgb_classifier,
)
from aprilalgo.tuner.walk_forward import walk_forward_splits, walk_forward_summary


def _load_cfg(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _symbols_for_cfg(cfg: dict[str, Any]) -> list[str]:
    syms = cfg.get("symbols")
    if syms:
        return [str(s) for s in syms]
    return [str(cfg["symbol"])]


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


def _regime_meta_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Persisted under ``meta.json`` key ``regime`` (defaults when YAML omits the block)."""
    raw = cfg.get("regime")
    if isinstance(raw, dict):
        return {
            "enabled": bool(raw.get("enabled", False)),
            "window": int(raw.get("window", 20)),
            "n_buckets": int(raw.get("n_buckets", 3)),
            "use_hmm": bool(raw.get("use_hmm", False)),
            "groupby": bool(raw.get("groupby", False)),
        }
    return {
        "enabled": False,
        "window": 20,
        "n_buckets": 3,
        "use_hmm": False,
        "groupby": False,
    }


def _cfg_for_inference(cfg: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """Merge persisted ``meta.json`` fields over *cfg* so predict/SHAP match training."""
    merged = dict(cfg)
    ib = bundle.meta.get("information_bars")
    if ib:
        merged["information_bars"] = ib
    reg = bundle.meta.get("regime")
    if isinstance(reg, dict):
        merged["regime"] = reg
    return merged


def _regime_groupby_training(cfg: dict[str, Any]) -> bool:
    r = cfg.get("regime")
    return isinstance(r, dict) and bool(r.get("enabled")) and bool(r.get("groupby", False))


def _train_regime_groupby_bundles(
    cfg: dict[str, Any],
    *,
    X: pd.DataFrame,
    y: pd.Series,
    t0: np.ndarray,
    t1: np.ndarray,
    task: Task,
    out_dir: Path,
    symbol: str,
    seed: int,
    xgb_params: dict[str, Any],
) -> None:
    """Train one XGBoost bundle per ``vol_regime`` bucket; write ``regime_index.json``."""
    if "vol_regime" not in X.columns:
        raise ValueError("regime.groupby requires vol_regime in the feature matrix (enable regime in YAML).")
    reg_col = X["vol_regime"]
    keys = sorted({int(round(float(v))) for v in reg_col.dropna().unique()})
    if not keys:
        raise ValueError("regime.groupby training requires at least one non-NaN vol_regime value.")
    default_train_k = int(keys[0])

    base_extra: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": cfg.get("timeframe"),
        "groupby_symbol": cfg.get("groupby_symbol", False),
        "sampling": _sampling_meta(cfg),
    }
    ib_meta = information_bars_meta_from_cfg(cfg)
    if ib_meta is not None:
        base_extra["information_bars"] = ib_meta
    reg_meta_base = {**_regime_meta_from_cfg(cfg), "groupby": True}

    buckets_map: dict[str, str] = {}
    feats_ref: list[str] | None = None
    classes_ref: tuple[float, ...] | None = None

    for k in keys:
        mask = reg_col.isna() | reg_col.eq(float(k)) if k == default_train_k else reg_col.eq(float(k))
        mask_arr = mask.to_numpy()
        if not mask_arr.any():
            continue
        X_b = X.loc[mask].reset_index(drop=True)
        y_b = y.loc[mask].reset_index(drop=True)
        t0_b = t0[mask_arr]
        t1_b = t1[mask_arr]
        if len(y_b) < 2:
            continue
        sw = _weights_for_training(cfg, t0_b, t1_b)
        clf = train_xgb_classifier(
            X_b,
            y_b,
            task=task,
            random_state=seed,
            xgb_params=xgb_params,
            sample_weight=sw,
        )
        cl = tuple(float(c) for c in np.asarray(clf.classes_).ravel())
        if classes_ref is None:
            classes_ref = cl
        elif cl != classes_ref:
            raise ValueError(
                f"regime.groupby: all buckets must expose the same sklearn classes_ (got {classes_ref} vs {cl})"
            )
        subdir = out_dir / f"regime_{k}"
        extra = dict(base_extra)
        rmeta = dict(reg_meta_base)
        rmeta["bucket"] = int(k)
        extra["regime"] = rmeta
        save_model_bundle(
            subdir,
            clf,
            feature_names=list(X_b.columns),
            task=task,
            indicator_config=cfg["indicators"],
            extra_meta=extra,
        )
        buckets_map[str(k)] = f"regime_{k}"
        if feats_ref is None:
            feats_ref = list(X_b.columns)
        elif list(X_b.columns) != feats_ref:
            raise ValueError("per-regime bundles must share identical feature columns")

    if not buckets_map:
        raise ValueError("regime.groupby: no bucket produced a valid model")
    if str(default_train_k) not in buckets_map:
        raise ValueError(
            f"regime.groupby: the default vol_regime bucket could not be trained (expected bucket {default_train_k})."
        )
    default_key = str(min(int(bk) for bk in buckets_map))
    default_sub = buckets_map[default_key]
    index = {"buckets": buckets_map, "default": default_sub}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regime_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def _train_and_save(cfg: dict[str, Any], *, symbol: str, out_dir: Path) -> None:
    X, y, t0, t1, task = _prepare_xy(cfg, symbol=symbol)
    seed = int(cfg.get("random_state", 42))
    xgb_params = cfg.get("model", {}).get("xgb", {})
    if _regime_groupby_training(cfg):
        _train_regime_groupby_bundles(
            cfg,
            X=X,
            y=y,
            t0=t0,
            t1=t1,
            task=task,
            out_dir=out_dir,
            symbol=symbol,
            seed=seed,
            xgb_params=xgb_params,
        )
        return

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
    extra["regime"] = _regime_meta_from_cfg(cfg)
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
    sw = _weights_for_training(cfg, t0, t1)

    res = purged_cv_evaluate(
        factory,
        X,
        y,
        sample_t0=t0,
        sample_t1=t1,
        n_splits=n_splits,
        embargo=embargo,
        sample_weight=sw,
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
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(oof_df)} rows to {out_path.resolve()}")


def _read_oof_primary_aligned(oof_path: Path, *, n_rows: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Load ``oof_primary.csv`` and return ``oof_pred`` aligned to ``0..n_rows-1``."""
    if not oof_path.is_file():
        raise FileNotFoundError(
            f"Missing {oof_path}: run `python -m aprilalgo.cli oof --config <same.yaml>` "
            "after training so OOF rows match the current feature matrix."
        )
    oof = pd.read_csv(oof_path)
    if "row_idx" not in oof.columns or "oof_pred" not in oof.columns:
        raise ValueError(f"{oof_path} must contain columns 'row_idx' and 'oof_pred' (from `cli oof`).")
    oof = oof.sort_values("row_idx").reset_index(drop=True)
    if len(oof) != n_rows:
        raise ValueError(
            f"{oof_path} has {len(oof)} rows but the prepared feature matrix has {n_rows}; "
            "regenerate OOF with the same ML config and symbol."
        )
    ridx = oof["row_idx"].to_numpy(dtype=np.int64)
    if not (ridx == np.arange(n_rows, dtype=np.int64)).all():
        raise ValueError(f"{oof_path}: row_idx must equal 0..{n_rows - 1} in order (same mask as train/oof).")
    return oof["oof_pred"].to_numpy(dtype=np.float64), oof


def cmd_meta_train(args: argparse.Namespace) -> None:
    """Fit purged meta logit from ``oof_primary.csv`` + features; write bundle + ``meta_oof.csv``."""
    cfg = _load_cfg(args.config)
    out_dir = Path(cfg["model"]["out_dir"])
    sym = str(cfg.get("symbol") or "")
    if not sym:
        raise ValueError("config.symbol is required for meta-train")
    oof_path = out_dir / "oof_primary.csv"
    X, y, t0, t1, _task = _prepare_xy(cfg, symbol=sym)
    oof_pred, _oof_df = _read_oof_primary_aligned(oof_path, n_rows=len(X))

    cv = cfg.get("cv", {})
    n_splits = int(cv.get("n_splits", 3))
    embargo = int(cv.get("embargo", 0))
    meta_block = cfg.get("meta_label") or {}
    n_splits_meta = int(meta_block.get("n_splits", n_splits))
    embargo_meta = int(meta_block.get("embargo", embargo))

    meta_full, meta_oof, z = fit_meta_logit_purged(
        X,
        np.asarray(y),
        oof_pred,
        sample_t0=t0,
        sample_t1=t1,
        n_splits=n_splits_meta,
        embargo=embargo_meta,
    )
    feature_names = list(X.columns) + ["primary_pred"]
    save_meta_logit_bundle(out_dir, meta_full, feature_names=feature_names)

    meta_oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(X), dtype=np.int64),
            "y_true": np.asarray(y, dtype=np.float64),
            "z": z,
            "meta_oof_proba": meta_oof,
        }
    )
    meta_oof_df.to_csv(out_dir / "meta_oof.csv", index=False)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    threshold = float(meta_block.get("threshold", 0.5))
    meta["meta_logit"] = {
        "enabled": True,
        "path": "meta_logit.json",
        "threshold": threshold,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Wrote meta_logit.json + meta_oof.csv under {out_dir.resolve()}; "
        f"updated {meta_path.name} meta_logit.enabled=true"
    )


def _load_reference_bundle(model_dir: Path) -> ModelBundle:
    """Single bundle at *model_dir*, or the default sub-bundle when ``regime_index.json`` exists."""
    idx_path = model_dir / "regime_index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        return load_model_bundle(model_dir / idx["default"])
    return load_model_bundle(model_dir)


def _predict_regime_routed(
    model_dir: Path, cfg: dict[str, Any], *, sym: str
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, list[float]]:
    """Row-wise routing to per-regime bundles (``regime_index.json``)."""
    idx = json.loads((model_dir / "regime_index.json").read_text(encoding="utf-8"))
    buckets: dict[str, str] = idx["buckets"]
    default_subdir = idx["default"]
    default_bundle = load_model_bundle(model_dir / default_subdir)
    default_key = next(k for k, v in buckets.items() if v == default_subdir)
    bundles = {str(k): load_model_bundle(model_dir / rel) for k, rel in buckets.items()}
    cfg_inf = _cfg_for_inference(cfg, default_bundle)
    X, y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
    cols0 = list(default_bundle.feature_names)
    for b in bundles.values():
        if list(b.feature_names) != cols0:
            raise ValueError("per-regime bundles must share identical feature_names for routed predict")
    if "vol_regime" not in X.columns:
        raise ValueError("routed predict requires vol_regime in the prepared feature matrix")

    master_classes = sorted({float(c) for b in bundles.values() for c in np.asarray(b.classes_).ravel()})
    if not master_classes:
        raise ValueError("routed predict: no classes_ found on regime bundles")

    vr = X["vol_regime"]
    keys_avail = set(buckets.keys())
    row_bucket: list[str] = []
    for v in vr:
        if pd.isna(v):
            row_bucket.append(default_key)
            continue
        kk = str(int(round(float(v))))
        row_bucket.append(kk if kk in keys_avail else default_key)

    n = len(X)
    n_cls = len(master_classes)
    proba = np.zeros((n, n_cls), dtype=np.float64)
    pred = np.zeros(n, dtype=np.float64)

    for key in sorted(keys_avail, key=lambda x: int(x)):
        positions = [i for i, rb in enumerate(row_bucket) if rb == key]
        if not positions:
            continue
        b = bundles[key]
        Xi = X.iloc[positions]
        pr = b.predict_proba(Xi)
        pdv = b.predict(Xi)
        for j, pos in enumerate(positions):
            rowp = np.asarray(pr[j, :], dtype=np.float64).ravel()
            if b.task == "binary" and rowp.size == 2:
                for c_canon, val in zip((0.0, 1.0), rowp, strict=True):
                    if c_canon in master_classes:
                        proba[pos, master_classes.index(c_canon)] = val
            else:
                bc = [float(c) for c in np.asarray(b.classes_).ravel()]
                col_ix = [master_classes.index(c) for c in bc]
                for dst, val in zip(col_ix, rowp, strict=True):
                    proba[pos, dst] = val
            pred[pos] = float(pdv[j])

    return X, y, pred, proba, master_classes


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["model"]["out_dir"])
    idx_path = model_dir / "regime_index.json"
    if idx_path.is_file():
        ref = _load_reference_bundle(model_dir)
        sym = ref.meta.get("symbol") or cfg.get("symbol")
        if not sym:
            raise ValueError("Config needs symbol or model meta.json must include symbol")
        X, y, pred, proba, classes = _predict_regime_routed(model_dir, cfg, sym=str(sym))
    else:
        bundle = load_model_bundle(model_dir)
        sym = bundle.meta.get("symbol") or cfg.get("symbol")
        if not sym:
            raise ValueError("Config needs symbol or model meta.json must include symbol")
        cfg_inf = _cfg_for_inference(cfg, bundle)
        X, y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
        proba = bundle.predict_proba(X)
        pred = bundle.predict(X)
        classes = [float(c) for c in np.asarray(bundle.classes_).ravel()]
    out = pd.DataFrame({"y_true": y.to_numpy(), "pred": pred})
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
    perm = permutation_importance_table(clf, X, y, n_repeats=int(cfg.get("importance_repeats", 5)), random_state=seed)
    out_dir = Path(cfg["model"].get("out_dir", "outputs/ml"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gain.to_csv(out_dir / "importance_gain.csv", index=False)
    perm.to_csv(out_dir / "importance_permutation.csv", index=False)
    print(gain.head(15).to_string(index=False))
    print("--- permutation (top 15) ---")
    print(perm.head(15).to_string(index=False))
    print(f"Wrote CSVs under {out_dir.resolve()}")


def _wf_tune_grid_from_yaml(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """Build hyperparameter grid list from ``wf_tuner.grid`` in YAML.

    * If *grid* is a list of dicts, it is used as-is (explicit points).
    * If *grid* is a dict, it is passed to :func:`~aprilalgo.tuner.ml_walk_forward.expand_grid`
      (Cartesian product; scalars become single-value lists).
    """
    from aprilalgo.tuner.ml_walk_forward import expand_grid

    raw = wf.get("grid")
    if raw is None:
        raise ValueError("wf_tuner.grid is required (list of dicts or expand-grid spec dict)")
    if isinstance(raw, list):
        out = [dict(x) for x in raw]
        if not out:
            raise ValueError("wf_tuner.grid list must be non-empty")
        return out
    if isinstance(raw, dict):
        return expand_grid(raw)
    raise ValueError(f"wf_tuner.grid must be a list or dict, got {type(raw).__name__}")


def cmd_wf_tune(args: argparse.Namespace) -> None:
    """Purged CV inside each walk-forward train window; CSV + top-5 summary to stdout."""
    from aprilalgo.tuner.ml_walk_forward import aggregate_grid, ml_walk_forward_tune

    cfg = _load_cfg(args.config)
    wf = cfg.get("wf_tuner")
    if not isinstance(wf, dict):
        raise ValueError("config must define a wf_tuner: mapping (metric, grid, …)")
    metric = wf.get("metric")
    if not metric:
        raise ValueError("wf_tuner.metric is required (e.g. accuracy, f1_macro, neg_log_loss)")
    grid = _wf_tune_grid_from_yaml(wf)
    wf_block = cfg.get("walk_forward") or {}
    n_folds = int(wf.get("n_folds", wf_block.get("n_folds", 4)))

    results = ml_walk_forward_tune(cfg, grid, n_folds, str(metric))
    out_dir = Path(cfg["model"].get("out_dir", "outputs/ml"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wf_tune_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote {len(results)} rows to {csv_path.resolve()}")

    agg = aggregate_grid(results, "score").sort_values("mean", ascending=False)
    top = agg.head(5)
    print("\nTop grid points by mean score (aggregated across walk-forward folds):")
    print(top.to_string(index=False))

    # Human-readable params for the printed top rows (first matching raw row per grid_id).
    if not top.empty and "grid_params_json" in results.columns:
        print("\nParams (JSON) for top rows:")
        for gid in top["grid_id"].tolist():
            sub = results.loc[results["grid_id"] == gid, "grid_params_json"]
            if not sub.empty:
                print(f"  {gid}: {sub.iloc[0]}")


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
            "test_return": float((float(df.iloc[te[-1]]["close"]) / float(df.iloc[te[0]]["close"])) - 1.0),
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


def _regime_bucket_key_series(vr: pd.Series, idx: dict[str, Any]) -> pd.Series:
    """Map each ``vol_regime`` row to a bucket id string matching ``regime_index.json`` keys."""
    buckets: dict[str, str] = idx["buckets"]
    default_subdir = idx["default"]
    default_key = next(k for k, v in buckets.items() if v == default_subdir)
    keys_avail = {str(k) for k in buckets}
    out: list[str] = []
    for v in vr:
        if pd.isna(v):
            out.append(str(default_key))
            continue
        kk = str(int(round(float(v))))
        out.append(kk if kk in keys_avail else str(default_key))
    return pd.Series(out, index=vr.index, dtype=str)


def cmd_shap(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["model"]["out_dir"])
    max_s = int(args.max_samples)

    if bool(getattr(args, "per_regime", False)):
        from aprilalgo.ml.explain import load_regime_bundles_shap, shap_values_per_regime

        idx_path = model_dir / "regime_index.json"
        if not idx_path.is_file():
            raise ValueError("--per-regime requires regime_index.json (train with regime.groupby: true)")
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        bundles = load_regime_bundles_shap(model_dir)
        first_k = next(iter(idx["buckets"]))
        ref = bundles[str(first_k)]
        sym = ref.meta.get("symbol") or cfg.get("symbol")
        if not sym:
            raise ValueError("Config needs symbol or bundle meta.json must include symbol")
        cfg_inf = _cfg_for_inference(cfg, ref)
        X, _y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
        if "vol_regime" not in X.columns:
            raise ValueError("--per-regime needs vol_regime in the feature matrix (enable regime in YAML / meta)")
        keys = _regime_bucket_key_series(X["vol_regime"], idx)
        X_by: dict[str, pd.DataFrame] = {}
        for k in bundles:
            kk = str(k)
            sub = X.loc[keys == kk]
            if len(sub) > 0:
                X_by[kk] = sub
        res = shap_values_per_regime(bundles, X_by, max_samples=max_s)
        for k, d in res.items():
            v_path = model_dir / f"regime_{k}_shap_values.csv"
            i_path = model_dir / f"regime_{k}_shap_importance.csv"
            d["values"].to_csv(v_path, index=False)
            d["importance"].to_csv(i_path, index=False)
            print(f"Wrote {v_path.resolve()} and {i_path.resolve()}")
        if not res:
            raise ValueError("per-regime SHAP: no rows matched any trained regime bucket")
        return

    bundle = _load_reference_bundle(model_dir)
    sym = bundle.meta.get("symbol") or cfg.get("symbol")
    if not sym:
        raise ValueError("Config needs symbol or model meta.json must include symbol")
    cfg_inf = _cfg_for_inference(cfg, bundle)
    X, _y, _t0, _t1, _task = _prepare_xy(cfg_inf, symbol=str(sym))
    vals = shap_values_table(bundle, X, max_samples=max_s)
    imp = shap_importance_table(bundle, X, max_samples=max_s)
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

    p_meta = sub.add_parser(
        "meta-train",
        help="Fit meta logistic from oof_primary.csv + features; writes meta_logit.json",
    )
    p_meta.add_argument("--config", type=str, required=True)
    p_meta.set_defaults(func=cmd_meta_train)

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
    p_shap.add_argument(
        "--per-regime",
        action="store_true",
        help="With regime_index.json: one SHAP export per bucket under regime_<k>_shap_*.csv",
    )
    p_shap.set_defaults(func=cmd_shap)

    p_wf = sub.add_parser(
        "walk-forward",
        help="Print walk-forward index ranges for config symbol (JSON)",
    )
    p_wf.add_argument("--config", type=str, required=True)
    p_wf.set_defaults(func=cmd_walk_forward)

    p_wf_tune = sub.add_parser(
        "wf-tune",
        help="Walk-forward grid search with purged CV per train window; writes wf_tune_results.csv",
    )
    p_wf_tune.add_argument("--config", type=str, required=True)
    p_wf_tune.set_defaults(func=cmd_wf_tune)

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
