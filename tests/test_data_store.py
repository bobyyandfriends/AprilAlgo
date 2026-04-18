"""Tests for aprilalgo.data.store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aprilalgo.data.store import load_csv, load_pickle, save_csv, save_pickle


def test_save_load_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b.csv"
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=3, freq="D"), "x": [1, 2, 3]})
    save_csv(df, p)
    out = load_csv(p)
    assert pd.api.types.is_datetime64_any_dtype(out["datetime"])
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df)


def test_load_csv_invalid_datetime(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("datetime,x\nnot-a-date,1\n", encoding="utf-8")
    out = load_csv(p)
    assert pd.isna(out["datetime"].iloc[0])


def test_save_load_pickle_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.pkl"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    save_pickle(df, p)
    out = load_pickle(p)
    pd.testing.assert_frame_equal(out, df)


def test_save_csv_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deep" / "f.csv"
    save_csv(pd.DataFrame({"a": [1]}), p)
    assert p.is_file()
