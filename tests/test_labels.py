"""Tests for triple-barrier labeling (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.labels.triple_barrier import (
    LABEL_STOP_LOSS,
    LABEL_TAKE_PROFIT,
    LABEL_VERTICAL_TIMEOUT,
    apply_triple_barrier,
)


def _ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Rows as (open, high, low, close); synthetic index; float columns for safe mutation."""
    o, h, l, c = zip(*rows, strict=True)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c},
        index=pd.RangeIndex(len(rows)),
        dtype=float,
    )


class TestTripleBarrierAssignment:
    def test_take_profit_hit_first(self):
        # entry 100 at i=0; +2% -> 102; next bar high touches 102, low stays above 98
        df = _ohlc(
            [
                (100, 101, 99.5, 100),
                (100, 103, 99.8, 102),
                (102, 102, 101, 101.5),
            ]
        )
        # vertical_bars must be < n - i so horizon exists (here n=3, i=0 -> max V is 2)
        out = apply_triple_barrier(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
        assert out.label.iloc[0] == LABEL_TAKE_PROFIT
        assert out.barrier_hit_offset.iloc[0] == 1

    def test_stop_loss_hit_first(self):
        # entry 100; -2% -> 98; first future bar dips to 97
        df = _ohlc(
            [
                (100, 101, 99.5, 100),
                (100, 100.5, 97, 98),
                (98, 99, 97.5, 98.5),
            ]
        )
        out = apply_triple_barrier(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
        assert out.label.iloc[0] == LABEL_STOP_LOSS
        assert out.barrier_hit_offset.iloc[0] == 1

    def test_vertical_timeout(self):
        # entry 100; barriers 102 / 98; price wanders inside band for 3 bars
        df = _ohlc(
            [
                (100, 101, 99.5, 100),
                (100, 101, 99.6, 100.2),
                (100.2, 101, 99.7, 100.1),
                (100.1, 101, 99.8, 100.0),
            ]
        )
        out = apply_triple_barrier(
            df, upper_pct=0.02, lower_pct=0.02, vertical_bars=3
        )
        assert out.label.iloc[0] == LABEL_VERTICAL_TIMEOUT
        assert out.barrier_hit_offset.iloc[0] == 3

    def test_both_barriers_same_bar_stop_loss_first(self):
        # entry 100 -> upper 102, lower 98; one bar spans both (n=2 -> only V=1 future bar)
        df = _ohlc(
            [
                (100, 101, 99.5, 100),
                (100, 105, 95, 100),
            ]
        )
        out = apply_triple_barrier(
            df,
            upper_pct=0.02,
            lower_pct=0.02,
            vertical_bars=1,
            both_hit_policy="stop_loss_first",
        )
        assert out.label.iloc[0] == LABEL_STOP_LOSS

    def test_both_barriers_same_bar_take_profit_first(self):
        df = _ohlc(
            [
                (100, 101, 99.5, 100),
                (100, 105, 95, 100),
            ]
        )
        out = apply_triple_barrier(
            df,
            upper_pct=0.02,
            lower_pct=0.02,
            vertical_bars=1,
            both_hit_policy="take_profit_first",
        )
        assert out.label.iloc[0] == LABEL_TAKE_PROFIT

    def test_insufficient_future_bars_nan(self):
        df = _ohlc(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),
            ]
        )
        out = apply_triple_barrier(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=3)
        assert np.isnan(out.label.iloc[0])
        assert np.isnan(out.barrier_hit_offset.iloc[0])


class TestNoLookahead:
    def test_decision_uses_close_at_t_first_scan_at_t_plus_1(self):
        # Bar 1 would trigger TP if entry were taken from bar 1's close; decision at 0 uses 100 -> 102
        df = _ohlc(
            [
                (100, 101, 99, 100),
                (100, 103, 99, 102),
                (102, 103, 101, 102),
            ]
        )
        out = apply_triple_barrier(df, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
        # First bar after decision must be index 1, not 0
        assert out.label.iloc[0] == LABEL_TAKE_PROFIT
        assert out.barrier_hit_offset.iloc[0] == 1

    def test_label_stable_if_only_post_barrier_path_changes(self):
        """Mutating bars strictly after the first hit must not change the label."""
        base = _ohlc(
            [
                (100, 101, 99, 100),
                (100, 103, 99.5, 101),
                (101, 102, 100, 101),
                (101, 102, 100, 101),
            ]
        )
        out1 = apply_triple_barrier(
            base, upper_pct=0.02, lower_pct=0.02, vertical_bars=3
        )
        mutated = base.copy()
        mutated.loc[3, ["high", "low", "close"]] = [200.0, 199.0, 199.5]
        out2 = apply_triple_barrier(
            mutated, upper_pct=0.02, lower_pct=0.02, vertical_bars=3
        )
        assert out1.label.iloc[0] == out2.label.iloc[0] == LABEL_TAKE_PROFIT
        assert out1.barrier_hit_offset.iloc[0] == out2.barrier_hit_offset.iloc[0]

    def test_changing_future_before_barrier_changes_label(self):
        # Four bars: three future steps stay inside [98, 102] -> timeout when V=3
        df1 = _ohlc(
            [
                (100, 101, 99, 100),
                (100, 101, 99.5, 100),
                (100, 101, 99.6, 100),
                (100, 101, 99.7, 100),
            ]
        )
        # First future bar quiet; second reaches TP — different outcome than df1
        df2 = _ohlc(
            [
                (100, 101, 99, 100),
                (100, 100.5, 99.5, 100),
                (100, 103, 99, 101),
            ]
        )
        o1 = apply_triple_barrier(df1, upper_pct=0.02, lower_pct=0.02, vertical_bars=3)
        o2 = apply_triple_barrier(df2, upper_pct=0.02, lower_pct=0.02, vertical_bars=2)
        assert o1.label.iloc[0] == LABEL_VERTICAL_TIMEOUT
        assert o2.label.iloc[0] == LABEL_TAKE_PROFIT


class TestValidation:
    def test_bad_vertical_bars(self):
        df = _ohlc([(1, 1, 1, 1)])
        with pytest.raises(ValueError, match="vertical_bars"):
            apply_triple_barrier(df, upper_pct=0.01, lower_pct=0.01, vertical_bars=0)

    def test_bad_pct(self):
        df = _ohlc([(1, 1, 1, 1), (1, 1, 1, 1)])
        with pytest.raises(ValueError, match="upper_pct"):
            apply_triple_barrier(df, upper_pct=-0.01, lower_pct=0.01, vertical_bars=1)
