"""Load YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config file and return it as a dict.

    Falls back to ``configs/default.yaml`` when *path* is ``None``.
    """
    fpath = Path(path) if path else _DEFAULT_CONFIG
    if not fpath.exists():
        return _defaults()
    with open(fpath) as f:
        cfg = yaml.safe_load(f) or {}
    merged = _defaults()
    merged.update(cfg)
    return merged


def _defaults() -> dict:
    return {
        "initial_capital": 100_000.0,
        "commission": 0.0,
        "slippage": 0.0005,
        "strategy": "rsi_sma",
        "strategy_params": {},
        "symbol": "AAPL",
        "timeframe": "daily",
    }
