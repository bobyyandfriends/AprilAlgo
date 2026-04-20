"""Load OHLCV price data from CSV files by symbol and timeframe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aprilalgo.data.bars import apply_information_bars_from_config

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"


def get_price_path(symbol: str, timeframe: str = "daily", data_dir: Path | None = None) -> Path:
    """Return the path to a symbol's CSV for a given timeframe.

    Convention: ``data/{timeframe}_data/{SYMBOL}_{timeframe}.csv``
    """
    base = Path(data_dir) if data_dir else _DATA_DIR
    return base / f"{timeframe}_data" / f"{symbol.upper()}_{timeframe}.csv"


def load_price_data(
    symbol: str,
    timeframe: str = "daily",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load OHLCV data for *symbol* at *timeframe*, returning a clean DataFrame.

    Columns: ``datetime | open | high | low | close | volume``
    """
    fpath = get_price_path(symbol, timeframe, data_dir)
    if not fpath.exists():
        raise FileNotFoundError(f"Price file not found: {fpath}")

    df = pd.read_csv(fpath)

    if "datetime" not in df.columns:
        for alt in ("timestamp", "date", "Date", "Datetime"):
            if alt in df.columns:
                df.rename(columns={alt: "datetime"}, inplace=True)
                break
        else:
            raise ValueError(f"No datetime column found in {fpath}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def information_bars_enabled(cfg: dict[str, Any]) -> bool:
    ib = cfg.get("information_bars")
    return bool(isinstance(ib, dict) and ib.get("enabled"))


def resolved_source_timeframe_for_ml(cfg: dict[str, Any]) -> str:
    """Timeframe of the on-disk CSV used before optional information-bar aggregation."""
    ib = cfg.get("information_bars") or {}
    if information_bars_enabled(cfg):
        return str(ib.get("source_timeframe") or cfg.get("timeframe", "daily"))
    return str(cfg.get("timeframe", "daily"))


def information_bars_meta_from_cfg(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return a JSON-serializable recipe for ``meta.json``, or None if bars are off."""
    if not information_bars_enabled(cfg):
        return None
    ib = cfg.get("information_bars") or {}
    src_tf = resolved_source_timeframe_for_ml(cfg)
    out: dict[str, Any] = {
        "enabled": True,
        "bar_type": str(ib["bar_type"]),
        "threshold": ib["threshold"],
        "source_timeframe": src_tf,
    }
    return out


def load_ohlcv_for_ml(cfg: dict[str, Any], symbol: str) -> pd.DataFrame:
    """Load OHLCV for ML / triple-barrier / features: optional information bars first.

    When ``information_bars.enabled`` is true, loads
    ``source_timeframe`` (or top-level ``timeframe``) from disk, then aggregates.
    Otherwise loads the top-level ``timeframe`` CSV only.
    """
    data_dir = Path(cfg["data_dir"]) if cfg.get("data_dir") else None
    if information_bars_enabled(cfg):
        ib = cfg.get("information_bars") or {}
        src_tf = resolved_source_timeframe_for_ml(cfg)
        raw = load_price_data(symbol, src_tf, data_dir=data_dir)
        return apply_information_bars_from_config(raw, ib)
    tf = str(cfg.get("timeframe", "daily"))
    return load_price_data(symbol, tf, data_dir=data_dir)
