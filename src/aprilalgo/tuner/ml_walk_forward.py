"""Purged walk-forward hyperparameter search (v0.5 Sprint 8).

For each walk-forward train window, runs :func:`~aprilalgo.ml.evaluator.purged_cv_evaluate`
on that window only, then scores the outer walk-forward test block implicitly via the
mean of inner CV metrics (same pattern as evaluating stability of hyperparameters over time).

Imports neutral ML pipeline helpers from :mod:`aprilalgo.ml.pipeline` rather than
``aprilalgo.cli``, so unrelated CLI imports (new handlers, heavy optional deps)
cannot break the tuner.
"""

from __future__ import annotations

import copy
import hashlib
import json
from itertools import product
from typing import Any

import pandas as pd

from aprilalgo.ml.evaluator import purged_cv_evaluate
from aprilalgo.ml.pipeline import prepare_xy, weights_for_training, xgb_estimator_factory
from aprilalgo.ml.trainer import Task
from aprilalgo.tuner.walk_forward import walk_forward_splits

SUPPORTED_METRICS: frozenset[str] = frozenset({"accuracy", "f1_macro", "neg_log_loss"})


def supported_metrics() -> frozenset[str]:
    """Metric names accepted by :func:`ml_walk_forward_tune`."""
    return SUPPORTED_METRICS


def expand_grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of hyperparameter lists.

    Parameters
    ----------
    spec
        Each key maps to a list of values (or a scalar, treated as a one-element list).
        Example: ``{"max_depth": [2, 3], "learning_rate": [0.1]}`` → two dicts.

    Returns
    -------
    list[dict[str, Any]]
        One dict per grid point; keys are the same as *spec* keys.
    """
    if not spec:
        return [{}]
    keys = list(spec.keys())
    value_lists: list[list[Any]] = []
    for k in keys:
        v = spec[k]
        if isinstance(v, (list, tuple)):
            value_lists.append(list(v))
        else:
            value_lists.append([v])
    out: list[dict[str, Any]] = []
    for combo in product(*value_lists):
        out.append(dict(zip(keys, combo, strict=True)))
    return out


def aggregate_grid(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate per-fold scores by ``grid_id``.

    Parameters
    ----------
    results_df
        Must contain columns ``grid_id`` and *metric* (numeric scores).
    metric
        Column name to aggregate (typically ``"score"`` from :func:`ml_walk_forward_tune`).

    Returns
    -------
    pd.DataFrame
        Columns ``grid_id``, ``mean``, ``std``, ``n_folds`` (count of rows per grid).
    """
    if results_df.empty:
        return pd.DataFrame(columns=["grid_id", "mean", "std", "n_folds"])
    if "grid_id" not in results_df.columns or metric not in results_df.columns:
        raise ValueError(f"results_df must contain 'grid_id' and '{metric}' columns")
    g = results_df.groupby("grid_id", sort=False)[metric]
    agg = g.agg(mean="mean", std="std", n_folds="count").reset_index()
    return agg


def _grid_id(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _score_from_purged_mean(mean: dict[str, Any], metric: str, task: Task) -> float:
    if metric == "accuracy":
        v = mean.get("accuracy")
    elif metric == "f1_macro":
        v = mean.get("f1_macro")
        if v is None and task == "binary":
            v = mean.get("f1")
    elif metric == "neg_log_loss":
        ll = mean.get("log_loss")
        v = -float(ll) if ll is not None else None
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(SUPPORTED_METRICS)}")
    if v is None:
        return float("nan")
    return float(v)


def _merge_xgb_params(cfg: dict[str, Any], grid_point: dict[str, Any]) -> dict[str, Any]:
    base = dict(cfg.get("model", {}).get("xgb", {}))
    base.update(grid_point)
    rs = int(cfg.get("random_state", 42))
    base.setdefault("random_state", rs)
    return base


def ml_walk_forward_tune(
    cfg: dict[str, Any],
    grid: list[dict[str, Any]],
    n_folds: int,
    metric: str,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Walk-forward grid search with purged CV inside each train window.

    For each ``grid`` point and each walk-forward split, fits purged k-fold CV on the
    train window only and records the mean *metric* from inner folds (no re-fit on the
    outer WF test block — that is reserved for Sprint 10-style reporting if needed).

    Parameters
    ----------
    cfg
        ML YAML dict (``symbol``, ``triple_barrier``, ``indicators``, ``model.xgb``,
        ``walk_forward``, ``cv``, ``task``, …).
    grid
        List of XGBoost param dict fragments merged into ``cfg['model']['xgb']`` per run.
    n_folds
        Walk-forward fold count passed to :func:`~aprilalgo.tuner.walk_forward.walk_forward_splits`.
    metric
        One of ``accuracy``, ``f1_macro``, ``neg_log_loss`` (negative sklearn log-loss).
    symbol
        Optional symbol override for :func:`~aprilalgo.cli._prepare_xy`.

    Returns
    -------
    pd.DataFrame
        One row per ``(grid_id, wf_fold)`` with at least
        ``grid_id``, ``wf_fold``, ``metric``, ``score``, ``train_rows``, ``purged_cv_n_splits``.
    """
    if not grid:
        raise ValueError("grid must be a non-empty list of parameter dicts")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}, got {metric!r}")

    sym = symbol if symbol is not None else str(cfg["symbol"])
    X, y, t0, t1, task = prepare_xy(cfg, symbol=sym)
    # Thread full-dataset sample weights through the inner CV so tuner
    # hyperparameter selection matches the production training distribution
    # (§AUDIT B4).
    sample_weight_full = weights_for_training(cfg, t0, t1)
    wf = cfg.get("walk_forward") or {}
    min_train = int(wf.get("min_train", 50))
    test_size = wf.get("test_size")
    cv_block = cfg.get("cv") or {}
    inner_splits = int(cv_block.get("n_splits", 3))
    embargo = int(cv_block.get("embargo", 0))

    n = len(X)
    splits = list(walk_forward_splits(n, n_folds=n_folds, min_train=min_train, test_size=test_size))
    if not splits:
        raise ValueError("walk_forward_splits produced no splits; lower min_train or check data length")

    rows: list[dict[str, Any]] = []
    for gi, grid_point in enumerate(grid):
        gid = _grid_id(grid_point)
        # ``copy.deepcopy`` handles arbitrary value types (Path, tuple, numpy scalars,
        # datetime), which ``json.dumps`` does not — a yaml-loaded config that contains
        # any of those otherwise raises ``TypeError``.
        cfg_m = copy.deepcopy(cfg)
        cfg_m.setdefault("model", {})
        cfg_m["model"] = dict(cfg_m.get("model", {}))
        cfg_m["model"]["xgb"] = _merge_xgb_params(cfg_m, grid_point)

        for wf_fold, (train_idx, _test_idx) in enumerate(splits):
            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            t0_tr = t0[train_idx]
            t1_tr = t1[train_idx]
            sw_tr = None if sample_weight_full is None else sample_weight_full[train_idx]

            factory = xgb_estimator_factory(cfg_m, task)
            res = purged_cv_evaluate(
                factory,
                X_tr,
                y_tr,
                sample_t0=t0_tr,
                sample_t1=t1_tr,
                n_splits=inner_splits,
                embargo=embargo,
                sample_weight=sw_tr,
            )
            mean = res.get("mean") or {}
            score = _score_from_purged_mean(mean, metric, task)
            rows.append(
                {
                    "grid_id": gid,
                    "grid_index": gi,
                    "wf_fold": wf_fold,
                    "metric": metric,
                    "score": score,
                    "train_rows": int(train_idx.size),
                    "purged_cv_n_splits": int(res.get("n_splits", inner_splits)),
                    "grid_params_json": json.dumps(grid_point, sort_keys=True, default=str),
                }
            )

    return pd.DataFrame(rows)
