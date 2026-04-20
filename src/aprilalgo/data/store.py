"""Simple I/O helpers for DataFrames (CSV and pickle)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write *df* to a CSV file (no index column)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, auto-parsing a ``datetime`` column if present."""
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def save_pickle(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def load_pickle(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(path)  # nosec B301  # trusted local artifact paths only
