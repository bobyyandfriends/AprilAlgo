"""Tests for aprilalgo.data.fetcher (mocked REST; no network)."""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import aprilalgo.data.fetcher as fetcher


def test_fetch_bars_unknown_timeframe() -> None:
    with pytest.raises(ValueError, match="Unknown timeframe"):
        fetcher.fetch_bars("AAPL", timeframe="quarterly", save=False)


def test_fetch_bars_missing_massive_dep() -> None:
    real_import = builtins.__import__

    def _block(name: str, *a, **kw):  # noqa: ANN002
        if name == "massive":
            raise ImportError("simulated missing massive")
        return real_import(name, *a, **kw)

    with (
        patch.object(builtins, "__import__", side_effect=_block),
        pytest.raises(ImportError, match="massive|uv add"),
    ):
        fetcher.fetch_bars("AAPL", timeframe="daily", save=False)


def test_fetch_bars_no_rows_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import massive

    class _C:
        def __init__(self, api_key=None):
            pass

        def list_aggs(self, **kwargs):
            yield from ()

    monkeypatch.setattr(massive, "RESTClient", _C)
    with pytest.raises(ValueError, match="No data returned"):
        fetcher.fetch_bars("AAPL", timeframe="daily", save=False)


def test_fetch_bars_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import massive

    class _Bar:
        def __init__(self, ts: int):
            self.timestamp = ts
            self.open = self.high = self.low = self.close = 100.0
            self.volume = 10.0

    class _C:
        def __init__(self, api_key=None):
            pass

        def list_aggs(self, **kwargs):
            yield _Bar(1609459200000)
            yield _Bar(1609545600000)

    monkeypatch.setattr(massive, "RESTClient", _C)
    df = fetcher.fetch_bars(
        "ZZZ", timeframe="daily", start="2020-01-01", end="2020-01-10", save=True, data_dir=tmp_path
    )
    assert len(df) == 2
    assert "datetime" in df.columns


def test_fetch_universe_accumulates_errors(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"i": 0}

    def _fake_fetch(
        symbol: str,
        timeframe: str = "daily",
        start: str = "2020-01-01",
        end: str | None = None,
        api_key: str | None = None,
        save: bool = True,
        data_dir: Path | None = None,
    ):
        calls["i"] += 1
        if calls["i"] == 1:
            raise RuntimeError("boom")
        return pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2020-01-01")],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )

    monkeypatch.setattr(fetcher, "fetch_bars", _fake_fetch)
    out = fetcher.fetch_universe(["BAD", "GOOD"], timeframe="daily")
    assert "GOOD" in out
    assert "BAD" not in out
    captured = capsys.readouterr()
    assert "FAILED" in captured.out or "FAILED" in captured.err
