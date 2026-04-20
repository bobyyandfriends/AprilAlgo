"""Cached loaders for XGBoost model bundles, OOF probabilities, and SHAP tables.

All entry points are ``@st.cache_data``-wrapped where they return dataframes,
and keyed on ``(model_dir, symbol, timeframe, ...)`` so switching symbols does
not invalidate cached SHAP matrices for other symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from aprilalgo.ml.explain import shap_importance_table, shap_values_table
from aprilalgo.ml.features import build_feature_matrix
from aprilalgo.ml.trainer import ModelBundle, load_model_bundle

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_MODELS_DIR = _PROJECT_ROOT / "models"


def discover_model_dirs() -> list[Path]:
    """Return every directory under ``models/`` that looks like a bundle.

    A directory counts as a bundle if it (or any immediate child) contains
    both ``meta.json`` and ``xgboost.json``. Paths are returned relative to
    the project root so they render nicely in a ``st.selectbox``.
    """
    out: list[Path] = []
    if not _MODELS_DIR.exists():
        return out
    for p in sorted(_MODELS_DIR.rglob("meta.json")):
        if (p.parent / "xgboost.json").is_file():
            out.append(p.parent)
    return out


def project_relative(path: Path) -> str:
    """Format *path* as a project-relative string for UI display."""
    try:
        return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_model_dir(rel_or_abs: str) -> Path:
    """Resolve a project-relative string back to an absolute path."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Bundle + metadata (non-cached — the bundle holds an ``xgb.Booster`` which is
# not picklable by ``st.cache_data``).
# ---------------------------------------------------------------------------


def load_bundle(model_dir: Path | str) -> ModelBundle:
    return load_model_bundle(resolve_model_dir(str(model_dir)))


def bundle_meta(model_dir: Path | str) -> dict[str, Any]:
    try:
        b = load_bundle(model_dir)
    except Exception:
        return {}
    return dict(b.meta or {})


# ---------------------------------------------------------------------------
# OOF probabilities — read the CSV written by ``cli.py oof`` (oof_primary.csv).
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=600)
def load_oof_proba(model_dir: str) -> pd.DataFrame | None:
    """Return the OOF probability frame for *model_dir* or ``None`` if absent.

    Looks for ``oof_primary.csv`` first (written by ``cli oof``), then
    ``oof.parquet`` as a secondary convention.
    """
    root = resolve_model_dir(model_dir)
    csv_path = root / "oof_primary.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        return df
    pq_path = root / "oof.parquet"
    if pq_path.is_file():
        try:
            df = pd.read_parquet(pq_path)
            return df
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False, ttl=600)
def build_proba_frame_for_df(
    model_dir: str,
    df_len: int,
    df_signature: tuple[Any, ...],
) -> pd.DataFrame | None:
    """Align an OOF CSV to the length of the currently rendered price frame.

    OOF rows are produced from the feature matrix *after* NaN warm-up rows are
    dropped, so we left-pad the probability frame with NaNs up to *df_len*.

    *df_signature* is used purely to key the Streamlit cache; callers can
    pass ``(symbol, timeframe, first_ts, last_ts)`` so changing the window
    re-runs the alignment.
    """
    _ = df_signature  # only used as a cache key
    oof = load_oof_proba(model_dir)
    if oof is None or oof.empty:
        return None
    proba_cols = [c for c in oof.columns if c.lower().startswith(("oof_proba_", "proba_"))]
    if not proba_cols:
        return None
    # OOF is written with training-row ordering; pad head with NaNs to match
    # the displayed candle window.
    n_pad = max(0, df_len - len(oof))
    if n_pad:
        pad = pd.DataFrame(
            {c: np.full(n_pad, np.nan, dtype=np.float64) for c in proba_cols}
        )
        out = pd.concat([pad, oof[proba_cols]], ignore_index=True)
    else:
        out = oof[proba_cols].tail(df_len).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# SHAP matrix / importance — computed on demand, cached per model dir.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Computing SHAP values...", ttl=600)
def compute_shap_tables(
    model_dir: str,
    feature_signature: tuple[str, ...],
    feature_frame: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> dict[str, pd.DataFrame] | None:
    """Compute the long-form SHAP values and importance tables.

    *feature_signature* (e.g. ``tuple(X.columns)``) is part of the cache key
    so shape/column changes invalidate the cache.
    """
    _ = feature_signature
    try:
        bundle = load_bundle(model_dir)
    except FileNotFoundError:
        return None

    missing = [c for c in bundle.feature_names if c not in feature_frame.columns]
    if missing:
        return None
    X = feature_frame[bundle.feature_names].copy()

    try:
        values = shap_values_table(bundle, X, max_samples=max_samples)
        importance = shap_importance_table(bundle, X, max_samples=max_samples)
    except ImportError:
        return None
    return {"values": values, "importance": importance}


def shap_values_to_wide(values_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form ``shap_values_table`` into ``(n_samples, n_features)``."""
    if values_long is None or values_long.empty:
        return pd.DataFrame()
    wide = values_long.pivot(
        index="sample_idx", columns="feature", values="shap_value"
    ).sort_index()
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------------------
# Feature-matrix alignment helpers (needed to map SHAP rows back onto price
# rows after warm-up NaN drops).
# ---------------------------------------------------------------------------


def build_feature_frame_for_chart(
    df: pd.DataFrame,
    indicator_config: list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(feature_frame, valid_mask)`` aligned row-for-row with *df*.

    ``feature_frame`` has ``len(df)`` rows (NaN-filled where the feature
    pipeline produced a row with missing values). ``valid_mask`` is a boolean
    numpy array marking rows whose features are all finite — those are the
    rows SHAP can be computed on.
    """
    if indicator_config is None:
        return pd.DataFrame(index=df.index), np.zeros(len(df), dtype=bool)
    X = build_feature_matrix(df, indicator_config=indicator_config)
    X = X.reset_index(drop=True)
    mask = X.notna().all(axis=1).to_numpy()
    return X, mask


def align_shap_to_price_rows(
    shap_wide: pd.DataFrame,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    """Pad *shap_wide* (indexed 0..n_valid-1) out to ``len(valid_mask)``.

    Rows where ``valid_mask`` is False get NaN (no SHAP contribution). The
    output's row index matches the price frame index.
    """
    n = len(valid_mask)
    if shap_wide is None or shap_wide.empty:
        return pd.DataFrame(index=range(n))

    out = pd.DataFrame(
        index=range(n),
        columns=list(shap_wide.columns),
        dtype=np.float64,
    )
    valid_positions = np.flatnonzero(valid_mask)
    k = min(len(valid_positions), len(shap_wide))
    if k == 0:
        return out
    src = shap_wide.iloc[:k].to_numpy(dtype=np.float64, copy=False)
    out.iloc[valid_positions[:k]] = src
    return out
