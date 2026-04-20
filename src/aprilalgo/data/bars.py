"""Information-driven bars: tick, volume, and dollar bars."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    need = {"datetime", "open", "high", "low", "close", "volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required OHLCV columns: {sorted(miss)}")
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def _aggregate_chunk(chunk: pd.DataFrame, *, dollar_value: float | None = None) -> dict:
    rec: dict[str, float | pd.Timestamp | str] = {
        "datetime": chunk["datetime"].iloc[-1],
        "open": float(chunk["open"].iloc[0]),
        "high": float(chunk["high"].max()),
        "low": float(chunk["low"].min()),
        "close": float(chunk["close"].iloc[-1]),
        "volume": float(chunk["volume"].sum()),
    }
    if dollar_value is not None:
        rec["dollar_value"] = float(dollar_value)
    return rec


def _build(
    df: pd.DataFrame,
    *,
    threshold: float,
    acc_fn: Callable[[pd.Series], float],
    bar_type: str,
) -> pd.DataFrame:
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    src = _validate_input(df)
    out_rows: list[dict] = []
    start = 0
    acc = 0.0
    for i in range(len(src)):
        row = src.iloc[i]
        acc += float(acc_fn(row))
        if acc >= threshold:
            chunk = src.iloc[start : i + 1]
            rec = _aggregate_chunk(
                chunk,
                dollar_value=float((chunk["close"] * chunk["volume"]).sum())
                if bar_type == "dollar"
                else None,
            )
            rec["bar_type"] = bar_type
            rec["threshold"] = float(threshold)
            rec["source_rows"] = int(len(chunk))
            out_rows.append(rec)
            start = i + 1
            acc = 0.0
    if start < len(src):
        chunk = src.iloc[start:]
        rec = _aggregate_chunk(
            chunk,
            dollar_value=float((chunk["close"] * chunk["volume"]).sum())
            if bar_type == "dollar"
            else None,
        )
        rec["bar_type"] = bar_type
        rec["threshold"] = float(threshold)
        rec["source_rows"] = int(len(chunk))
        out_rows.append(rec)
    out = pd.DataFrame(out_rows)
    return out.sort_values("datetime").reset_index(drop=True)


def build_tick_bars(df: pd.DataFrame, *, threshold: int) -> pd.DataFrame:
    """Aggregate every ``threshold`` rows into one bar."""
    return _build(df, threshold=float(threshold), acc_fn=lambda _r: 1.0, bar_type="tick")


def build_volume_bars(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Aggregate rows until cumulative volume >= threshold."""
    return _build(df, threshold=float(threshold), acc_fn=lambda r: float(r["volume"]), bar_type="volume")


def build_dollar_bars(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Aggregate rows until cumulative dollar value ``close * volume`` >= threshold."""
    return _build(
        df,
        threshold=float(threshold),
        acc_fn=lambda r: float(r["close"]) * float(r["volume"]),
        bar_type="dollar",
    )


def apply_information_bars_from_config(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Build tick / volume / dollar bars from *df* using YAML ``information_bars`` fields.

    Expects ``bar_type`` and ``threshold``. ``enabled`` is ignored (caller decides).
    """
    kind = str(spec["bar_type"])
    th = spec["threshold"]
    if kind == "tick":
        return build_tick_bars(df, threshold=int(th))
    if kind == "volume":
        return build_volume_bars(df, threshold=float(th))
    if kind == "dollar":
        return build_dollar_bars(df, threshold=float(th))
    raise ValueError(f"Unknown information_bars.bar_type: {kind!r}")
