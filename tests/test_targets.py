"""Tests for unified triple-barrier target frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aprilalgo.labels.targets import (
    barrier_hit_name,
    build_triple_barrier_targets,
    targets_from_triple_barrier_result,
)
from aprilalgo.labels.triple_barrier import (
    LABEL_STOP_LOSS,
    LABEL_TAKE_PROFIT,
    LABEL_VERTICAL_TIMEOUT,
    apply_triple_barrier,
)


def _ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    o, h, l, c = zip(*rows, strict=True)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c},
        index=pd.RangeIndex(len(rows)),
        dtype=float,
    )


def test_binary_is_one_only_for_take_profit():
    df = _ohlc(
        [
            (100, 101, 99.5, 100),
            (100, 103, 99.8, 102),
            (102, 102, 101, 101.5),
        ]
    )
    t = build_triple_barrier_targets(
        df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2
    )
    assert t["label_multiclass"].iloc[0] == LABEL_TAKE_PROFIT
    assert t["label_binary"].iloc[0] == 1.0
    assert t["barrier_hit"].iloc[0] == "take_profit"
    assert t["label_t0"].iloc[0] == 0
    assert t["label_t1"].iloc[0] == 1.0


def test_binary_zero_for_stop_loss():
    df = _ohlc(
        [
            (100, 101, 99.5, 100),
            (100, 100.5, 97, 98),
            (98, 99, 97.5, 98.5),
        ]
    )
    t = build_triple_barrier_targets(
        df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2
    )
    assert t["label_multiclass"].iloc[0] == LABEL_STOP_LOSS
    assert t["label_binary"].iloc[0] == 0.0
    assert t["barrier_hit"].iloc[0] == "stop_loss"


def test_vertical_timeout_binary_zero():
    df = _ohlc(
        [
            (100, 101, 99.5, 100),
            (100, 101, 99.6, 100.2),
            (100.2, 101, 99.7, 100.1),
            (100.1, 101, 99.8, 100.0),
        ]
    )
    t = build_triple_barrier_targets(
        df, upper_pct=0.02, lower_pct=0.02, vertical_bars=3
    )
    assert t["label_multiclass"].iloc[0] == LABEL_VERTICAL_TIMEOUT
    assert t["label_binary"].iloc[0] == 0.0
    assert t["barrier_hit"].iloc[0] == "vertical_timeout"


def test_nan_multiclass_yields_nan_binary_and_barrier():
    df = _ohlc([(100, 101, 99, 100)])
    t = build_triple_barrier_targets(
        df, upper_pct=0.02, lower_pct=0.02, vertical_bars=5
    )
    assert pd.isna(t["label_multiclass"].iloc[0])
    assert pd.isna(t["label_binary"].iloc[0])
    assert pd.isna(t["barrier_hit"].iloc[0])
    assert pd.isna(t["label_t1"].iloc[0])


def test_label_t1_matches_barrier_offset():
    df = _ohlc(
        [
            (100, 101, 99.5, 100),
            (100, 103, 99.8, 102),
            (102, 102, 101, 101.5),
        ]
    )
    t = build_triple_barrier_targets(
        df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2
    )
    assert t["label_t0"].iloc[0] == 0
    off = t["barrier_hit_offset"].iloc[0]
    assert t["label_t1"].iloc[0] == 0 + off


def test_barrier_hit_name_non_finite_and_unknown() -> None:
    assert barrier_hit_name(float("nan")) is None
    assert barrier_hit_name(99.0) is None


def test_targets_from_triple_barrier_result_matches_build() -> None:
    df = _ohlc(
        [
            (100, 101, 99.5, 100),
            (100, 103, 99.8, 102),
        ]
    )
    tb = apply_triple_barrier(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
    a = targets_from_triple_barrier_result(df.index, tb)
    b = build_triple_barrier_targets(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
    pd.testing.assert_frame_equal(a, b)
