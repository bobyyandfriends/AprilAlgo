"""Indicator descriptors — self-describing metadata for the UI and tuner.

Each IndicatorSpec wraps a raw indicator function with:
- display name, category, description
- parameter specs (name, range, default) for auto-generating UI controls
- overlay flag (draw on price chart vs sub-panel)

The global catalog is the SINGLE source of truth for what indicators exist.
Adding a new indicator = create the function file + add one entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ParamSpec:
    """One tunable parameter of an indicator."""

    name: str
    display_name: str
    default: int | float
    min_val: int | float
    max_val: int | float
    step: int | float = 1


@dataclass
class IndicatorSpec:
    """Self-describing indicator wrapping a function with metadata."""

    name: str
    display_name: str
    fn: Callable[..., pd.DataFrame]
    params: list[ParamSpec]
    category: str = "momentum"
    overlay: bool = False
    description: str = ""
    # Primary output column names at default parameters (for catalog / QA docs).
    output_columns: tuple[str, ...] = ()
    _param_transform: Callable[[dict], dict] | None = field(default=None, repr=False)

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        merged = self.default_params()
        merged.update(kwargs)
        if self._param_transform:
            merged = self._param_transform(merged)
        return self.fn(df, **merged)

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


def _identity(df: pd.DataFrame, **_: Any) -> pd.DataFrame:
    """No-op pipeline step used by UI-only pseudo-indicators (ML / SHAP overlays)."""
    return df


def _build_catalog() -> dict[str, IndicatorSpec]:
    """Build the indicator catalog. Imports are inside the function to avoid cycles."""
    from aprilalgo.indicators.bollinger import bollinger_bands
    from aprilalgo.indicators.demark import demark
    from aprilalgo.indicators.ehlers import decycler, roofing_filter, super_smoother
    from aprilalgo.indicators.hurst import hurst
    from aprilalgo.indicators.pv_sequences import pv_sequences
    from aprilalgo.indicators.rsi import rsi
    from aprilalgo.indicators.sma import sma
    from aprilalgo.indicators.tmi import tmi
    from aprilalgo.indicators.volume_trend import volume_trend

    def _hurst_transform(p: dict) -> dict:
        w = p.pop("window", 100)
        p["windows"] = [w]
        return p

    return {
        "rsi": IndicatorSpec(
            name="rsi",
            display_name="RSI",
            fn=rsi,
            params=[ParamSpec("period", "Period", 14, 5, 30, 1)],
            category="momentum",
            overlay=False,
            description="Relative Strength Index",
            output_columns=("rsi_14", "rsi_14_bull", "rsi_14_bear"),
        ),
        "sma": IndicatorSpec(
            name="sma",
            display_name="SMA",
            fn=sma,
            params=[ParamSpec("period", "Period", 20, 5, 200, 5)],
            category="trend",
            overlay=True,
            description="Simple Moving Average",
            output_columns=("sma_20", "sma_20_bull", "sma_20_bear"),
        ),
        "bollinger_bands": IndicatorSpec(
            name="bollinger_bands",
            display_name="Bollinger Bands",
            fn=bollinger_bands,
            params=[
                ParamSpec("period", "Period", 20, 5, 50, 5),
                ParamSpec("std_dev", "Std Dev", 2.0, 1.0, 3.0, 0.5),
            ],
            category="volatility",
            overlay=True,
            description="Bollinger Bands (mid, upper, lower)",
            output_columns=("bb_20_mid", "bb_20_upper", "bb_20_lower", "bb_20_pct", "bb_20_bull", "bb_20_bear"),
        ),
        "volume_trend": IndicatorSpec(
            name="volume_trend",
            display_name="Volume Trend",
            fn=volume_trend,
            params=[
                ParamSpec("vol_period", "Period", 20, 5, 50, 5),
                ParamSpec("threshold", "Threshold", 1.5, 1.0, 3.0, 0.25),
            ],
            category="volume",
            overlay=False,
            description="Volume-confirmed price trend",
            output_columns=("vol_20_sma", "vol_20_bull", "vol_20_bear"),
        ),
        "demark": IndicatorSpec(
            name="demark",
            display_name="DeMark",
            fn=demark,
            params=[ParamSpec("lookback", "Lookback", 4, 3, 6, 1)],
            category="exhaustion",
            overlay=True,
            description="TD Sequential Setup & Countdown",
            output_columns=("td_bull", "td_bear", "td_buy_setup"),
        ),
        "hurst": IndicatorSpec(
            name="hurst",
            display_name="Hurst Exponent",
            fn=hurst,
            params=[ParamSpec("window", "Window", 100, 50, 300, 25)],
            category="regime",
            overlay=False,
            description="Trend persistence vs mean-reversion detection",
            output_columns=("hurst_100", "hurst_bull", "hurst_bear"),
            _param_transform=_hurst_transform,
        ),
        "super_smoother": IndicatorSpec(
            name="super_smoother",
            display_name="Super Smoother",
            fn=super_smoother,
            params=[ParamSpec("period", "Period", 10, 5, 50, 5)],
            category="cycle",
            overlay=True,
            description="Ehlers Super Smoother filter",
            output_columns=("ss_10", "ss_10_bull", "ss_10_bear"),
        ),
        "roofing_filter": IndicatorSpec(
            name="roofing_filter",
            display_name="Roofing Filter",
            fn=roofing_filter,
            params=[
                ParamSpec("hp_period", "HP Period", 48, 20, 100, 5),
                ParamSpec("lp_period", "LP Period", 10, 5, 30, 5),
            ],
            category="cycle",
            overlay=False,
            description="Ehlers Roofing Filter (bandpass)",
            output_columns=("roof_48_10", "roof_48_10_bull", "roof_48_10_bear"),
        ),
        "decycler": IndicatorSpec(
            name="decycler",
            display_name="Decycler",
            fn=decycler,
            params=[ParamSpec("period", "Period", 125, 50, 250, 25)],
            category="trend",
            overlay=True,
            description="Ehlers Decycler (trend extraction)",
            output_columns=("decycler_125", "decycler_125_bull", "decycler_125_bear"),
        ),
        "tmi": IndicatorSpec(
            name="tmi",
            display_name="TMI",
            fn=tmi,
            params=[
                ParamSpec("period", "Period", 14, 5, 30, 1),
                ParamSpec("smooth", "Smooth", 5, 2, 10, 1),
            ],
            category="momentum",
            overlay=False,
            description="Turn Measurement Index",
            output_columns=("tmi_14", "tmi_14_bull", "tmi_14_bear"),
        ),
        "pv_sequences": IndicatorSpec(
            name="pv_sequences",
            display_name="PV Sequences",
            fn=pv_sequences,
            params=[ParamSpec("streak_threshold", "Streak", 3, 2, 7, 1)],
            category="pattern",
            overlay=False,
            description="Price-Volume state transitions",
            output_columns=("pv_bull", "pv_bear", "pv_state"),
        ),
        # --- UI-only pseudo-indicators (consumed by the charts page) ---
        "demark_counts": IndicatorSpec(
            name="demark_counts",
            display_name="DeMark Counts",
            fn=_identity,
            params=[ParamSpec("min_count", "Min count", 4, 1, 13, 1)],
            category="exhaustion",
            overlay=True,
            description="TD Sequential Setup / Countdown integer labels",
            output_columns=("_identity_passthrough",),
        ),
        "ml_proba": IndicatorSpec(
            name="ml_proba",
            display_name="ML Probability",
            fn=_identity,
            params=[ParamSpec("threshold", "Threshold", 0.55, 0.5, 0.95, 0.05)],
            category="ml",
            overlay=False,
            description="XGBoost out-of-fold class-1 probability",
            output_columns=("_identity_passthrough",),
        ),
        "shap_local": IndicatorSpec(
            name="shap_local",
            display_name="SHAP Contributions",
            fn=_identity,
            params=[ParamSpec("top_k", "Top-K features", 5, 3, 15, 1)],
            category="ml",
            overlay=False,
            description="Signed stacked SHAP contributions per bar (top-K globally)",
            output_columns=("_identity_passthrough",),
        ),
    }


_CATALOG: dict[str, IndicatorSpec] | None = None


def get_catalog() -> dict[str, IndicatorSpec]:
    """Return the global indicator catalog (lazy-loaded on first call)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG
