"""Tests for ML OHLCV loading with optional information-driven bars."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aprilalgo.data.loader import (
    information_bars_meta_from_cfg,
    load_ohlcv_for_ml,
    resolved_source_timeframe_for_ml,
)

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "ml" / "default.yaml"


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_load_ohlcv_for_ml_tick_bars_shorter_than_raw() -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    sym = str(cfg["symbol"])
    raw_len = len(load_ohlcv_for_ml({**cfg, "information_bars": {"enabled": False}}, sym))
    cfg["information_bars"] = {
        "enabled": True,
        "bar_type": "tick",
        "threshold": 5,
    }
    bar_len = len(load_ohlcv_for_ml(cfg, sym))
    assert bar_len < raw_len
    assert bar_len >= 1


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_information_bars_meta_from_cfg_roundtrip_fields() -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg["information_bars"] = {
        "enabled": True,
        "bar_type": "volume",
        "threshold": 1e6,
        "source_timeframe": "daily",
    }
    meta = information_bars_meta_from_cfg(cfg)
    assert meta is not None
    assert meta["enabled"] is True
    assert meta["bar_type"] == "volume"
    assert meta["threshold"] == 1e6
    assert meta["source_timeframe"] == "daily"


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_resolved_source_timeframe_uses_nested_default() -> None:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg["timeframe"] = "daily"
    cfg["information_bars"] = {"enabled": True, "bar_type": "tick", "threshold": 3}
    assert resolved_source_timeframe_for_ml(cfg) == "daily"
