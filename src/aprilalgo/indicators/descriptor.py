"""Indicator descriptors — self-describing metadata for the UI and tuner.

Each IndicatorSpec wraps a raw indicator function with:
- display name, category, description
- parameter specs (name, range, default) for auto-generating UI controls
- overlay flag (draw on price chart vs sub-panel)

The global catalog is the SINGLE source of truth for what indicators exist.
Adding a new indicator = create the function file + add one entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
    _param_transform: Callable[[dict], dict] | None = field(
        default=None, repr=False
    )

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        merged = self.default_params()
        merged.update(kwargs)
        if self._param_transform:
            merged = self._param_transform(merged)
        return self.fn(df, **merged)

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


def _build_catalog() -> dict[str, IndicatorSpec]:
    """Build the indicator catalog. Imports are inside the function to avoid cycles."""
    from aprilalgo.indicators.rsi import rsi
    from aprilalgo.indicators.sma import sma
    from aprilalgo.indicators.bollinger import bollinger_bands
    from aprilalgo.indicators.volume_trend import volume_trend
    from aprilalgo.indicators.demark import demark
    from aprilalgo.indicators.hurst import hurst
    from aprilalgo.indicators.ehlers import super_smoother, roofing_filter, decycler
    from aprilalgo.indicators.tmi import tmi
    from aprilalgo.indicators.pv_sequences import pv_sequences

    def _hurst_transform(p: dict) -> dict:
        w = p.pop("window", 100)
        p["windows"] = [w]
        return p

    return {
        "rsi": IndicatorSpec(
            name="rsi", display_name="RSI", fn=rsi,
            params=[ParamSpec("period", "Period", 14, 5, 30, 1)],
            category="momentum", overlay=False,
            description="Relative Strength Index",
        ),
        "sma": IndicatorSpec(
            name="sma", display_name="SMA", fn=sma,
            params=[ParamSpec("period", "Period", 20, 5, 200, 5)],
            category="trend", overlay=True,
            description="Simple Moving Average",
        ),
        "bollinger_bands": IndicatorSpec(
            name="bollinger_bands", display_name="Bollinger Bands", fn=bollinger_bands,
            params=[
                ParamSpec("period", "Period", 20, 5, 50, 5),
                ParamSpec("std_dev", "Std Dev", 2.0, 1.0, 3.0, 0.5),
            ],
            category="volatility", overlay=True,
            description="Bollinger Bands (mid, upper, lower)",
        ),
        "volume_trend": IndicatorSpec(
            name="volume_trend", display_name="Volume Trend", fn=volume_trend,
            params=[
                ParamSpec("vol_period", "Period", 20, 5, 50, 5),
                ParamSpec("threshold", "Threshold", 1.5, 1.0, 3.0, 0.25),
            ],
            category="volume", overlay=False,
            description="Volume-confirmed price trend",
        ),
        "demark": IndicatorSpec(
            name="demark", display_name="DeMark", fn=demark,
            params=[ParamSpec("lookback", "Lookback", 4, 3, 6, 1)],
            category="exhaustion", overlay=True,
            description="TD Sequential Setup & Countdown",
        ),
        "hurst": IndicatorSpec(
            name="hurst", display_name="Hurst Exponent", fn=hurst,
            params=[ParamSpec("window", "Window", 100, 50, 300, 25)],
            category="regime", overlay=False,
            description="Trend persistence vs mean-reversion detection",
            _param_transform=_hurst_transform,
        ),
        "super_smoother": IndicatorSpec(
            name="super_smoother", display_name="Super Smoother", fn=super_smoother,
            params=[ParamSpec("period", "Period", 10, 5, 50, 5)],
            category="cycle", overlay=True,
            description="Ehlers Super Smoother filter",
        ),
        "roofing_filter": IndicatorSpec(
            name="roofing_filter", display_name="Roofing Filter", fn=roofing_filter,
            params=[
                ParamSpec("hp_period", "HP Period", 48, 20, 100, 5),
                ParamSpec("lp_period", "LP Period", 10, 5, 30, 5),
            ],
            category="cycle", overlay=False,
            description="Ehlers Roofing Filter (bandpass)",
        ),
        "decycler": IndicatorSpec(
            name="decycler", display_name="Decycler", fn=decycler,
            params=[ParamSpec("period", "Period", 125, 50, 250, 25)],
            category="trend", overlay=True,
            description="Ehlers Decycler (trend extraction)",
        ),
        "tmi": IndicatorSpec(
            name="tmi", display_name="TMI", fn=tmi,
            params=[
                ParamSpec("period", "Period", 14, 5, 30, 1),
                ParamSpec("smooth", "Smooth", 5, 2, 10, 1),
            ],
            category="momentum", overlay=False,
            description="Turn Measurement Index",
        ),
        "pv_sequences": IndicatorSpec(
            name="pv_sequences", display_name="PV Sequences", fn=pv_sequences,
            params=[ParamSpec("streak_threshold", "Streak", 3, 2, 7, 1)],
            category="pattern", overlay=False,
            description="Price-Volume state transitions",
        ),
    }


_CATALOG: dict[str, IndicatorSpec] | None = None


def get_catalog() -> dict[str, IndicatorSpec]:
    """Return the global indicator catalog (lazy-loaded on first call)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG
