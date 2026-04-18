"""SHAP explainability helpers for XGBoost bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aprilalgo.ml.trainer import ModelBundle, load_model_bundle


def _tree_explainer(bundle: ModelBundle):
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Add dependency 'shap' to enable explainability.") from e
    return shap.TreeExplainer(bundle.booster)


def _shap_matrix(bundle: ModelBundle, X: pd.DataFrame, *, max_samples: int = 500) -> tuple[np.ndarray, pd.DataFrame]:
    xx = X[bundle.feature_names].copy().head(max_samples)
    explainer = _tree_explainer(bundle)
    shap_vals = explainer.shap_values(xx)
    arr = np.asarray(shap_vals)
    if isinstance(shap_vals, list):
        # Per-class list path (older SHAP API). Average absolute values so features
        # with large opposite-signed class contributions are not cancelled to zero
        # in the aggregate importance ranking.
        arr = np.abs(np.stack([np.asarray(v) for v in shap_vals], axis=0)).mean(axis=0)
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=0)
    return np.asarray(arr, dtype=np.float64), xx


def shap_values_table(
    bundle: ModelBundle,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Long-form SHAP values table with columns: sample_idx, feature, shap_value."""
    mat, xx = _shap_matrix(bundle, X, max_samples=max_samples)
    rows: list[dict[str, float | int | str]] = []
    for i in range(mat.shape[0]):
        for j, f in enumerate(xx.columns):
            rows.append({"sample_idx": int(i), "feature": f, "shap_value": float(mat[i, j])})
    return pd.DataFrame(rows)


def shap_importance_table(
    bundle: ModelBundle,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Feature importance from mean absolute SHAP values."""
    mat, xx = _shap_matrix(bundle, X, max_samples=max_samples)
    means = np.abs(mat).mean(axis=0)
    out = pd.DataFrame({"feature": list(xx.columns), "mean_abs_shap": means})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def shap_values_per_regime(
    bundles: dict[str, ModelBundle],
    X_by_regime: dict[str, pd.DataFrame],
    *,
    max_samples: int = 300,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run SHAP per trained regime bucket bundle.

    Parameters
    ----------
    bundles
        Maps regime bucket id (e.g. ``\"0\"``, ``\"1\"``) to the :class:`ModelBundle` for that bucket.
    X_by_regime
        Feature rows routed to each bucket (same keys as *bundles*; empty frames are skipped).

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        Each key has ``\"values\"`` and ``\"importance\"`` tables from
        :func:`shap_values_table` / :func:`shap_importance_table`.
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for k, X in X_by_regime.items():
        if k not in bundles or X is None or len(X) == 0:
            continue
        b = bundles[k]
        out[k] = {
            "values": shap_values_table(b, X, max_samples=max_samples),
            "importance": shap_importance_table(b, X, max_samples=max_samples),
        }
    return out


def load_regime_bundles_shap(model_dir: str | Path) -> dict[str, ModelBundle]:
    """Load every sub-bundle listed in ``regime_index.json`` (for per-regime SHAP)."""
    import json

    root = Path(model_dir)
    idx_path = root / "regime_index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing regime_index.json under {root}")
    idx: dict[str, Any] = json.loads(idx_path.read_text(encoding="utf-8"))
    buckets = idx.get("buckets") or {}
    return {str(k): load_model_bundle(root / rel) for k, rel in buckets.items()}
