"""Neutral ML pipeline helpers decoupled from :mod:`aprilalgo.cli`.

Historically :func:`_prepare_xy` and :func:`_xgb_estimator_factory` lived in
``aprilalgo.cli`` and were lazy-imported from downstream modules (the walk-forward
tuner, Streamlit pages). That coupling meant any unrelated CLI import failure
(e.g. a new subcommand pulling in a heavy optional dependency) would break the
whole ML pipeline.

This module is the canonical location. ``aprilalgo.cli`` re-exports the same
symbols for backward compatibility; callers should prefer importing from here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from aprilalgo.data import load_ohlcv_for_ml
from aprilalgo.labels.targets import build_triple_barrier_targets
from aprilalgo.meta.regime import add_vol_regime
from aprilalgo.ml.features import build_feature_matrix
from aprilalgo.ml.sampling import sequential_bootstrap_sample, uniqueness_weights
from aprilalgo.ml.trainer import Task


def apply_regime_if_enabled(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a ``vol_regime`` column when ``cfg['regime']['enabled']`` is truthy.

    Parameters mirror the persisted ``meta.json.regime`` block so training and
    inference stay consistent (``window``, ``n_buckets``, ``use_hmm``).
    """
    r = cfg.get("regime")
    if not isinstance(r, dict) or not r.get("enabled"):
        return df
    return add_vol_regime(
        df,
        window=int(r.get("window", 20)),
        n_buckets=int(r.get("n_buckets", 3)),
        use_hmm=bool(r.get("use_hmm", False)),
    )


def prepare_xy(
    cfg: dict[str, Any],
    *,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, Task]:
    """Load OHLCV, apply regime, build features + targets, filter NaNs.

    Returns ``(X, y, t0, t1, task)`` where ``t0`` / ``t1`` are the triple-barrier
    label intervals aligned with the filtered feature matrix.
    """
    sym = symbol if symbol is not None else cfg["symbol"]
    df = load_ohlcv_for_ml(cfg, str(sym))
    df = apply_regime_if_enabled(df, cfg)

    b = cfg["triple_barrier"]
    targets = build_triple_barrier_targets(
        df,
        upper_pct=float(b["upper_pct"]),
        lower_pct=float(b["lower_pct"]),
        vertical_bars=int(b["vertical_bars"]),
    )

    task: Task = cfg.get("task", "binary")
    y = targets["label_binary"] if task == "binary" else targets["label_multiclass"]

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


def xgb_estimator_factory(cfg: dict[str, Any], task: Task) -> Callable[[], Any]:
    """Return a zero-arg factory usable by purged CV / OOF helpers."""
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


def weights_for_training(cfg: dict[str, Any], t0: np.ndarray, t1: np.ndarray) -> np.ndarray | None:
    """Resolve per-row ``sample_weight`` from ``cfg['sampling']``.

    * ``strategy: none`` (or omitted) → return ``None`` so estimators fall back
      to uniform weighting.
    * ``strategy: uniqueness`` → :func:`aprilalgo.ml.sampling.uniqueness_weights`.
    * ``strategy: bootstrap`` → sequential-bootstrap counts renormalised to sum
      to ``len(t0)`` (so downstream ``sample_weight`` semantics match uniform).
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
    raise ValueError(f"Unknown sampling.strategy: {strategy!r} (expected none, uniqueness, or bootstrap)")


__all__ = [
    "apply_regime_if_enabled",
    "prepare_xy",
    "weights_for_training",
    "xgb_estimator_factory",
]
