"""Tests for aprilalgo.backtest.position_sizer."""

from __future__ import annotations

import pytest

from aprilalgo.backtest.position_sizer import (
    ATRBased,
    FixedFraction,
    FractionalKelly,
    SIZERS,
)


def test_fixed_fraction_size() -> None:
    ff = FixedFraction(fraction=0.02)
    assert ff.size(100_000.0, 100.0) == 20
    assert ff.size(50.0, 10_000.0) >= 1


def test_fractional_kelly_edges() -> None:
    fk = FractionalKelly(kelly_fraction=0.5, max_position_pct=0.25)
    assert fk.size(100_000.0, 50.0, win_prob=0.3, avg_win=1.0, avg_loss=1.0) == 0
    assert fk.size(100_000.0, 50.0, win_prob=0.6, avg_win=1.0, avg_loss=0.0) == 1
    big = FractionalKelly(kelly_fraction=1.0, max_position_pct=0.05)
    n = big.size(1_000_000.0, 10.0, win_prob=0.9, avg_win=2.0, avg_loss=1.0)
    assert n * 10.0 <= 1_000_000.0 * 0.05 + 10.0


def test_atr_based() -> None:
    a = ATRBased(risk_per_trade=1000.0, atr_multiplier=2.0)
    n = a.size(100_000.0, 100.0, atr=2.0)
    assert n >= 1
    assert a.size(100_000.0, 100.0, atr=-1.0) == 1
    m = a.size(1_000_000.0, 10.0, atr=0.01)
    assert m >= 1


def test_sizers_registry() -> None:
    assert SIZERS.keys() == {"fixed_fraction", "fractional_kelly", "atr_based"}
    assert SIZERS["fixed_fraction"] is FixedFraction
    assert SIZERS["fractional_kelly"] is FractionalKelly
    assert SIZERS["atr_based"] is ATRBased
