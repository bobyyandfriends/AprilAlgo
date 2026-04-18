"""Edge-case tests for aprilalgo.data.loader."""

from __future__ import annotations

import pandas as pd
import pytest

from aprilalgo.data.loader import (
    information_bars_meta_from_cfg,
    load_ohlcv_for_ml,
    load_price_data,
)


def test_load_price_data_renames_date_column(tmp_path: Path) -> None:
    p = tmp_path / "daily_data"
    p.mkdir(parents=True)
    csv = p / "ZZ_daily.csv"
    csv.write_text("Date,open,high,low,close,volume\n2020-01-01,1,1,1,1,10\n", encoding="utf-8")
    df = load_price_data("ZZ", "daily", data_dir=tmp_path)
    assert "datetime" in df.columns
    assert len(df) == 1


def test_load_price_data_raises_without_datetime_column(tmp_path: Path) -> None:
    p = tmp_path / "daily_data"
    p.mkdir(parents=True)
    csv = p / "AA_daily.csv"
    csv.write_text("open,high,low,close,volume\n1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No datetime column"):
        load_price_data("AA", "daily", data_dir=tmp_path)


def test_information_bars_meta_from_cfg_disabled() -> None:
    assert information_bars_meta_from_cfg({}) is None
    assert information_bars_meta_from_cfg({"information_bars": {"enabled": False}}) is None


def test_load_ohlcv_for_ml_unknown_information_bar_type(tmp_path: Path) -> None:
    p = tmp_path / "daily_data"
    p.mkdir(parents=True)
    (p / "X_daily.csv").write_text(
        "datetime,open,high,low,close,volume\n2020-01-01,10,11,9,10,100\n2020-01-02,10,11,9,10,100\n",
        encoding="utf-8",
    )
    cfg = {
        "symbol": "X",
        "timeframe": "daily",
        "data_dir": str(tmp_path),
        "information_bars": {"enabled": True, "bar_type": "not_a_bar", "threshold": 1},
    }
    with pytest.raises(ValueError, match="Unknown information_bars"):
        load_ohlcv_for_ml(cfg, "X")
