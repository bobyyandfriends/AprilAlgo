"""Tests for aprilalgo.config.load_config."""

from __future__ import annotations

from pathlib import Path

import yaml

import aprilalgo.config as cfgmod


def test_load_config_missing_path_returns_defaults() -> None:
    out = cfgmod.load_config("/nonexistent/path/that/does/not/exist.yaml")
    assert out == cfgmod._defaults()


def test_load_config_defaults_when_none() -> None:
    out = cfgmod.load_config(None)
    assert isinstance(out, dict)
    d = cfgmod._defaults()
    for k in d:
        assert k in out
    assert out["initial_capital"] == 100_000.0


def test_load_config_merges_overrides(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"strategy": "custom", "commission": 0.01}), encoding="utf-8")
    out = cfgmod.load_config(p)
    assert out["strategy"] == "custom"
    assert out["commission"] == 0.01
    assert out["slippage"] == cfgmod._defaults()["slippage"]


def test_load_config_empty_yaml(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    out = cfgmod.load_config(p)
    assert out == cfgmod._defaults()
