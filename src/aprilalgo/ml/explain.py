"""SHAP explainability helpers for XGBoost bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aprilalgo.ml.trainer import ModelBundle


def _tree_explainer(bundle: ModelBundle):
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "SHAP is not installed. Add dependency 'shap' to enable explainability."
        ) from e
    return shap.TreeExplainer(bundle.booster)


def _shap_matrix(bundle: ModelBundle, X: pd.DataFrame, *, max_samples: int = 500) -> tuple[np.ndarray, pd.DataFrame]:
    xx = X[bundle.feature_names].copy().head(max_samples)
    explainer = _tree_explainer(bundle)
    shap_vals = explainer.shap_values(xx)
    arr = np.asarray(shap_vals)
    if isinstance(shap_vals, list):
        arr = np.stack([np.asarray(v) for v in shap_vals], axis=0).mean(axis=0)
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
