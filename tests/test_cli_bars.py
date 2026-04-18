"""Tests for aprilalgo.cli bar build + walk-forward + error paths (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

import aprilalgo.cli as cli_mod
from aprilalgo.cli import cmd_bars, cmd_predict, cmd_shap, cmd_walk_forward


def _tiny_ohlcv(n: int = 30) -> pd.DataFrame:
    t = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "datetime": t,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 500.0,
        }
    )


@pytest.mark.parametrize("bar_type", ["tick", "volume", "dollar"])
def test_cmd_bars_writes_csv(tmp_path: Path, bar_type: str, capsys: pytest.CaptureFixture[str]) -> None:
    inp = tmp_path / "in.csv"
    _tiny_ohlcv(40).to_csv(inp, index=False)
    out = tmp_path / "out.csv"
    thr = 3.0 if bar_type != "tick" else 5.0
    ns = SimpleNamespace(input=str(inp), bar_type=bar_type, threshold=thr, output=str(out))
    cmd_bars(ns)
    assert out.is_file()
    df = pd.read_csv(out)
    assert "datetime" in df.columns
    assert len(df) >= 1
    assert "Wrote" in capsys.readouterr().out


def test_cmd_bars_drops_nat_datetimes(tmp_path: Path) -> None:
    # Keep datetime as plain strings so one row parses to NaT under errors="coerce"
    rows = []
    for i in range(5):
        ts = f"2020-01-01 09:{30 + i:02d}:00"
        if i == 2:
            ts = "not-a-date"
        rows.append(f"{ts},100,101,99,100.5,500")
    inp = tmp_path / "bad_dates.csv"
    inp.write_text("datetime,open,high,low,close,volume\n" + "\n".join(rows), encoding="utf-8")
    out = tmp_path / "vol.csv"
    ns = SimpleNamespace(input=str(inp), bar_type="volume", threshold=1000.0, output=str(out))
    cmd_bars(ns)
    loaded = pd.read_csv(out)
    assert len(loaded) >= 1


def test_cmd_walk_forward_json_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = {"symbol": "X", "walk_forward": {"n_folds": 3, "min_train": 20}}
    cfg_path = tmp_path / "wf.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def _fake_load(_cfg: dict, sym: str) -> pd.DataFrame:
        assert sym == "X"
        return _tiny_ohlcv(80)

    monkeypatch.setattr(cli_mod, "load_ohlcv_for_ml", _fake_load)
    ns = SimpleNamespace(config=str(cfg_path))
    cmd_walk_forward(ns)
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert "n_bars" in payload and payload["n_bars"] == 80
    assert "summary" in payload
    assert "splits" in payload and len(payload["splits"]) >= 1
    for sp in payload["splits"]:
        assert sp["train_size"] >= 1 and sp["test_size"] >= 1


def test_cmd_shap_per_regime_requires_regime_index(tmp_path: Path) -> None:
    cfg = {"model": {"out_dir": str(tmp_path / "models")}, "symbol": "AAPL"}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "models").mkdir(parents=True)
    ns = SimpleNamespace(
        config=str(p),
        model_dir=None,
        max_samples=10,
        per_regime=True,
        output=str(tmp_path / "v.csv"),
        importance_output=str(tmp_path / "i.csv"),
    )
    with pytest.raises(ValueError, match="regime_index.json"):
        cmd_shap(ns)


def test_cmd_predict_raises_without_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = {"model": {"out_dir": str(tmp_path / "m")}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "m").mkdir()

    def _fake_bundle(_path: Path):
        b = MagicMock()
        b.meta = {}
        return b

    monkeypatch.setattr("aprilalgo.cli.load_model_bundle", _fake_bundle)
    ns = SimpleNamespace(config=str(cfg_path), model_dir=None, output=str(tmp_path / "pred.csv"))
    with pytest.raises(ValueError, match="symbol"):
        cmd_predict(ns)


def test_cli_main_bars_invocation(tmp_path: Path) -> None:
    inp = tmp_path / "x.csv"
    _tiny_ohlcv(25).to_csv(inp, index=False)
    out = tmp_path / "tb.csv"
    argv = [
        "bars",
        "--input",
        str(inp),
        "--bar-type",
        "tick",
        "--threshold",
        "3",
        "--output",
        str(out),
    ]
    cli_mod.main(argv)
    assert out.is_file()


def test_cmd_bars_unknown_bar_type_raises(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    _tiny_ohlcv(10).to_csv(inp, index=False)
    out = tmp_path / "out.csv"
    ns = SimpleNamespace(input=str(inp), bar_type="not_a_kind", threshold=1.0, output=str(out))
    with pytest.raises(ValueError, match="Unknown bar type"):
        cmd_bars(ns)
