"""Tests for aprilalgo.ui.helpers (pure functions; no Streamlit session)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aprilalgo.ui.helpers as helpers


def test_discover_symbols_empty_when_data_dir_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(helpers, "_DATA_DIR", tmp_path / "no_such_data")
    assert helpers.discover_symbols() == {}


def test_discover_symbols_skips_non_directory_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "data_root"
    root.mkdir()
    (root / "hourly_data").write_text("not_a_dir", encoding="utf-8")
    daily = root / "daily_data"
    daily.mkdir()
    (daily / "ZZ_daily.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(helpers, "_DATA_DIR", root)
    assert helpers.discover_symbols() == {"daily": ["ZZ"]}


def test_discover_symbols_groups_csv_by_timeframe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "data_root"
    daily = root / "daily_data"
    daily.mkdir(parents=True)
    (daily / "AAPL_daily.csv").write_text("x", encoding="utf-8")
    (daily / "MSFT_daily.csv").write_text("x", encoding="utf-8")
    monkeypatch.setattr(helpers, "_DATA_DIR", root)
    out = helpers.discover_symbols()
    assert out == {"daily": ["AAPL", "MSFT"]}


def test_format_metric_none_and_nan() -> None:
    assert helpers.format_metric("total_return_pct", None) == "—"
    assert helpers.format_metric("sharpe_ratio", float("nan")) == "—"


@pytest.mark.parametrize(
    ("key", "value", "expected_substr"),
    [
        ("total_return_pct", 1.234, "1.23%"),
        ("win_rate_pct", 50.0, "50.00%"),
        ("sharpe_ratio", 1.5, "1.50"),
        ("profit_factor", 2.0, "2.00"),
        ("total_pnl", 1234.56, "$"),
        ("num_trades", 7, "7"),
    ],
)
def test_format_metric_typed_branches(key: str, value: float | int, expected_substr: str) -> None:
    s = helpers.format_metric(key, value)
    assert expected_substr in s


def test_format_metric_fallback_str_for_bad_float() -> None:
    bad = object()
    assert helpers.format_metric("sharpe_ratio", bad) == str(bad)


def test_format_metric_plain_str_when_no_rule_matches() -> None:
    assert helpers.format_metric("custom_metric_xyz", 42) == "42"
