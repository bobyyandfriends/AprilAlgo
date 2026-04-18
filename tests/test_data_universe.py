"""Tests for aprilalgo.data.universe."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import aprilalgo.data.universe as u


def test_default_when_missing(tmp_path: Path) -> None:
    assert u.load_universe(tmp_path / "missing.yaml") == u._default_symbols()


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"symbols": ["aapl", "msft"]}), encoding="utf-8")
    assert u.load_universe(p) == ["AAPL", "MSFT"]


def test_load_txt(tmp_path: Path) -> None:
    p = tmp_path / "s.txt"
    p.write_text("aapl\n\nMsft\n", encoding="utf-8")
    assert u.load_universe(p) == ["AAPL", "MSFT"]


def test_load_yaml_missing_symbols_key(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"other": 1}), encoding="utf-8")
    assert u.load_universe(p) == []


def test_load_default_from_project_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    sym_path = cfg_dir / "symbols.yaml"
    sym_path.write_text(yaml.safe_dump({"symbols": ["x"]}), encoding="utf-8")
    monkeypatch.setattr(u, "_PROJECT_ROOT", tmp_path)
    assert u.load_universe(None) == ["X"]
    sym_path.unlink()
    assert u.load_universe(None) == u._default_symbols()
