"""Tests for JSONL signal logger."""

from __future__ import annotations

import pytest

from aprilalgo.backtest.logger import (
    FULL_SIGNAL_EVENT_KEYS,
    SignalJsonlLogger,
    events_to_dataframe,
    log_event,
    validate_event,
)


def test_validate_event_requires_ts_and_symbol():
    with pytest.raises(ValueError):
        validate_event({"ts": "2020-01-01"})
    validate_event({"ts": "x", "symbol": "AAPL"})


def test_full_schema_validate_all_keys_present():
    evt = {k: None for k in FULL_SIGNAL_EVENT_KEYS}
    evt["ts"] = "2024-01-01"
    evt["symbol"] = "X"
    evt["pred"] = 1.0
    evt["pred_proba"] = [0.2, 0.8]
    validate_event(evt, schema="full")


def test_log_event_full_schema_roundtrip(tmp_path):
    p = tmp_path / "full.jsonl"
    log = SignalJsonlLogger(p)
    evt = {k: None for k in FULL_SIGNAL_EVENT_KEYS}
    evt.update(
        {
            "ts": "2024-01-01",
            "symbol": "X",
            "tf": "daily",
            "model_id": "m1",
            "features_hash": "abc",
            "pred": 1.0,
            "pred_proba": [0.4, 0.6],
            "label_multiclass": 1,
            "label_binary": 1,
            "meta_pred": 0.7,
            "pred_proba_meta": 0.55,
            "outcome": "pending",
            "pnl": 0.0,
        }
    )
    log_event(log, evt, schema="full")
    rows = log.read_all()
    assert len(rows) == 1
    assert rows[0]["features_hash"] == "abc"


def test_logger_roundtrip(tmp_path):
    p = tmp_path / "sig.jsonl"
    log = SignalJsonlLogger(p)
    log.log({"ts": "2020-01-01", "symbol": "AAPL", "pred": 1})
    log.log({"ts": "2020-01-02", "symbol": "AAPL", "pred": 0})
    rows = log.read_all()
    assert len(rows) == 2
    df = events_to_dataframe(rows)
    assert len(df) == 2
    assert "pred" in df.columns
