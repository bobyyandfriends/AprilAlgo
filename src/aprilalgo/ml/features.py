"""Feature matrix construction for ML — indicator columns only, no raw OHLCV.

Per ARCHITECTURE.md §4.6, training features come from the indicator registry
(enriched columns), not open/high/low/close/volume, to reduce price-level leakage.

See docs/DATA_SCHEMA.md §2 (Enriched Features) for column naming.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aprilalgo.indicators.registry import IndicatorRegistry

# Canonical OHLCV + common time columns — never treated as ML features by default
DEFAULT_EXCLUDED_FROM_FEATURES: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "date",
        "datetime",
        # information-driven bar metadata (see data/bars.py)
        "bar_type",
        "threshold",
        "source_rows",
        "dollar_value",
    }
)


def feature_column_names(
    df: pd.DataFrame,
    *,
    excluded: frozenset[str] | None = None,
    extra_exclude: frozenset[str] | None = None,
) -> list[str]:
    """Return sorted names of columns kept as features (not excluded metadata/OHLCV)."""
    skip = DEFAULT_EXCLUDED_FROM_FEATURES if excluded is None else excluded
    if extra_exclude:
        skip = skip | extra_exclude
    return sorted(c for c in df.columns if c not in skip)


def extract_feature_matrix(
    enriched: pd.DataFrame,
    *,
    excluded: frozenset[str] | None = None,
    extra_exclude: frozenset[str] | None = None,
    convert_bool: bool = True,
) -> pd.DataFrame:
    """Return a copy of *enriched* with only feature columns (strip OHLCV and time fields)."""
    cols = feature_column_names(
        enriched, excluded=excluded, extra_exclude=extra_exclude
    )
    if not cols:
        return pd.DataFrame(index=enriched.index)
    X = enriched.loc[:, cols].copy()
    if convert_bool:
        X = _floatify_bool_columns(X)
    return X


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    indicator_config: list[dict[str, Any]] | None = None,
    registry: IndicatorRegistry | None = None,
    excluded: frozenset[str] | None = None,
    extra_exclude: frozenset[str] | None = None,
    convert_bool: bool = True,
) -> pd.DataFrame:
    """Build an indicator-only feature matrix.

    If ``indicator_config`` or ``registry`` is provided, indicators are applied to
    *df* first (expected to contain OHLCV). If both are omitted, *df* is assumed
    already enriched and only non-OHLCV columns are kept.

    Parameters
    ----------
    df
        OHLCV bars, or an already-enriched frame if no pipeline is passed.
    indicator_config
        List of ``{"name": "rsi", "period": 14, ...}`` dicts — see
        :meth:`IndicatorRegistry.from_config`.
    registry
        Pre-built registry; alternative to ``indicator_config``.
    excluded
        Override default excluded column names (defaults to
        :data:`DEFAULT_EXCLUDED_FROM_FEATURES`).
    extra_exclude
        Additional column names to drop (e.g. ``{"confluence_score"}``).
    convert_bool
        If True, cast boolean indicator columns to float64 for sklearn/XGBoost.
    """
    if indicator_config is not None and registry is not None:
        raise ValueError("Pass at most one of indicator_config and registry")

    work = df
    if indicator_config is not None:
        work = IndicatorRegistry.from_config(indicator_config).apply(df)
    elif registry is not None:
        work = registry.apply(df)

    return extract_feature_matrix(
        work,
        excluded=excluded,
        extra_exclude=extra_exclude,
        convert_bool=convert_bool,
    )


def align_features_and_labels(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    label_name: str = "y",
    dropna_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align *features* and *labels* on index; drop rows with missing labels.

    Rows with NaN labels (e.g. triple-barrier horizon too short) are removed.
    If ``dropna_features`` is True, rows with any NaN in *features* are also dropped.
    """
    if not features.index.equals(labels.index):
        common = features.index.intersection(labels.index)
        features = features.loc[common]
        labels = labels.loc[common]

    combined = features.copy()
    combined[label_name] = labels
    combined = combined.dropna(subset=[label_name])
    if dropna_features:
        combined = combined.dropna()
    y = combined[label_name]
    X = combined.drop(columns=[label_name])
    return X, y


def _floatify_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("float64")
    return out
