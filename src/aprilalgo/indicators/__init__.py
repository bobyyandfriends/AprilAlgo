"""Technical indicators, pluggable registry, and self-describing catalog."""

from aprilalgo.indicators.bollinger import bollinger_bands
from aprilalgo.indicators.demark import demark
from aprilalgo.indicators.descriptor import (
    IndicatorSpec,
    ParamSpec,
    get_catalog,
)
from aprilalgo.indicators.ehlers import decycler, roofing_filter, super_smoother
from aprilalgo.indicators.hurst import hurst
from aprilalgo.indicators.pv_sequences import pv_sequences
from aprilalgo.indicators.registry import IndicatorRegistry, apply_indicators
from aprilalgo.indicators.rsi import rsi
from aprilalgo.indicators.sma import sma
from aprilalgo.indicators.tmi import tmi
from aprilalgo.indicators.volume_trend import volume_trend

__all__ = [
    "IndicatorRegistry",
    "apply_indicators",
    "get_catalog",
    "IndicatorSpec",
    "ParamSpec",
    "rsi",
    "sma",
    "bollinger_bands",
    "volume_trend",
    "demark",
    "hurst",
    "super_smoother",
    "roofing_filter",
    "decycler",
    "tmi",
    "pv_sequences",
]
