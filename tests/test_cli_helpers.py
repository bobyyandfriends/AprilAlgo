"""Unit tests for pure helpers in aprilalgo.cli."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import pandas as pd

from aprilalgo.cli import (
    _cfg_for_inference,
    _load_cfg,
    _regime_bucket_key_series,
    _regime_groupby_training,
    _regime_meta_from_cfg,
    _sampling_meta,
    _symbols_for_cfg,
    _wf_tune_grid_from_yaml,
)


def test_load_cfg_roundtrip(tmp_path: Path) -> None:
    d = {"a": 1, "b": [1, 2]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert _load_cfg(p) == d


def test_symbols_for_cfg_precedence() -> None:
    assert _symbols_for_cfg({"symbol": "X", "symbols": ["A", "B"]}) == ["A", "B"]
    assert _symbols_for_cfg({"symbol": "Z"}) == ["Z"]


def test_symbols_for_cfg_keyerror() -> None:
    with pytest.raises(KeyError):
        _symbols_for_cfg({})


def test_sampling_meta() -> None:
    assert _sampling_meta({})["strategy"] == "none"
    out = _sampling_meta({"sampling": {"strategy": "bootstrap", "random_state": 7, "n_draw": 50}})
    assert out["strategy"] == "bootstrap"
    assert out["random_state"] == 7
    assert out["n_draw"] == 50
    uq = _sampling_meta({"sampling": {"strategy": "uniqueness"}})
    assert uq["strategy"] == "uniqueness"
    rs_only = _sampling_meta({"sampling": {"strategy": "uniqueness", "random_state": 99}})
    assert rs_only["random_state"] == 99


def test_regime_bucket_key_series_nan_and_unknown() -> None:
    idx = {"buckets": {"0": "regime_0", "1": "regime_1"}, "default": "regime_0"}
    vr = pd.Series([float("nan"), 1.0, 1.4, 99.0], index=list("abcd"))
    out = _regime_bucket_key_series(vr, idx)
    assert list(out) == ["0", "1", "1", "0"]
    assert list(out.index) == list("abcd")


def test_regime_meta_from_cfg() -> None:
    d = _regime_meta_from_cfg({})
    assert d["enabled"] is False
    d2 = _regime_meta_from_cfg({"regime": {"enabled": True, "window": 9}})
    assert d2["enabled"] is True
    assert d2["window"] == 9


def test_cfg_for_inference_overrides_regime() -> None:
    cfg = {"regime": {"enabled": True, "window": 1, "n_buckets": 2, "use_hmm": False}}
    bundle = MagicMock()
    bundle.meta = {"regime": {"enabled": True, "window": 9, "n_buckets": 4, "use_hmm": False, "groupby": False}}
    merged = _cfg_for_inference(cfg, bundle)
    assert int(merged["regime"]["window"]) == 9


def test_regime_groupby_training() -> None:
    assert _regime_groupby_training({}) is False
    assert _regime_groupby_training({"regime": {"enabled": True, "groupby": True}}) is True


def test_wf_tune_grid_from_yaml_dict() -> None:
    wf = {"grid": {"max_depth": [2, 3], "learning_rate": [0.1, 0.2]}}
    g = _wf_tune_grid_from_yaml(wf)
    assert len(g) == 4
    depths = {x["max_depth"] for x in g}
    assert depths == {2, 3}


def test_wf_tune_grid_from_yaml_list() -> None:
    wf = {"grid": [{"max_depth": 2}, {"max_depth": 3}]}
    g = _wf_tune_grid_from_yaml(wf)
    assert g == [{"max_depth": 2}, {"max_depth": 3}]
