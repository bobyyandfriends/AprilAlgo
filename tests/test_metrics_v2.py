"""Coverage for :mod:`aprilalgo.backtest.metrics_v2` (AUDIT B2 / BUG_REPORT C1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aprilalgo.backtest.metrics_v2 import (
    _annualised_sharpe,
    _annualised_sortino,
    _empty_metrics,
    compute_metrics_from_equity,
    infer_periods_per_year,
)


def test_empty_metrics_shape() -> None:
    m = _empty_metrics()
    assert m["sharpe_ratio"] == 0.0
    assert m["n_observations"] == 0
    assert m["num_trades"] == 0


def test_compute_empty_or_missing_equity() -> None:
    assert compute_metrics_from_equity(pd.DataFrame()) == _empty_metrics()
    bad = pd.DataFrame({"datetime": [1], "x": [1.0]})
    assert compute_metrics_from_equity(bad) == _empty_metrics()


def test_compute_single_point_equity() -> None:
    df = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=1), "equity": [100_000.0]})
    out = compute_metrics_from_equity(df)
    assert out["total_return_pct"] == 0.0
    assert out["n_observations"] == 0


def test_flat_equity_zero_drawdown_near_zero_cagr() -> None:
    n = 100
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = np.full(n, 100_000.0)
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    out = compute_metrics_from_equity(df, initial_capital=100_000.0)
    assert abs(float(out["total_return_pct"])) < 1e-6
    assert abs(float(out["cagr_pct"])) < 0.01
    assert float(out["max_drawdown_pct"]) == 0.0
    assert float(out["sharpe_ratio"]) == 0.0
    assert float(out["sortino_ratio"]) == 0.0


def test_monotone_up_positive_cagr_and_sharpe() -> None:
    n = 252
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = np.linspace(100_000.0, 110_000.0, n)
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    out = compute_metrics_from_equity(df, initial_capital=100_000.0)
    assert float(out["total_return_pct"]) > 5.0
    assert float(out["cagr_pct"]) > 3.0
    assert float(out["sharpe_ratio"]) > 0.0
    assert float(out["max_drawdown_pct"]) == 0.0


def test_hand_computed_daily_fixture() -> None:
    """252 trading days, +10% linear equity (no vol in simple returns except endpoints)."""
    n = 253
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = np.linspace(100_000.0, 110_000.0, n, dtype=float)
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    out = compute_metrics_from_equity(df, periods_per_year=252.0, risk_free_rate=0.0)
    total = float(out["total_return_pct"]) / 100.0
    assert abs(total - 0.1) < 1e-3
    years = (n - 1) / 252.0
    expected_cagr = (1.1 ** (1.0 / years) - 1.0) * 100.0
    assert abs(float(out["cagr_pct"]) - expected_cagr) < 0.05


def test_risk_free_rate_reduces_sharpe() -> None:
    n = 120
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    eq = 100_000.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, size=n))
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    base = compute_metrics_from_equity(df, periods_per_year=252.0, risk_free_rate=0.0)
    rf = compute_metrics_from_equity(df, periods_per_year=252.0, risk_free_rate=0.10)
    assert float(rf["sharpe_ratio"]) <= float(base["sharpe_ratio"]) + 1e-6


def test_periods_per_year_override() -> None:
    n = 60
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = np.linspace(100_000.0, 105_000.0, n)
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    out = compute_metrics_from_equity(df, periods_per_year=52.0)
    assert float(out["periods_per_year"]) == 52.0


def test_infer_periods_daily_weekly_monthly_sparse() -> None:
    # Daily (~1 business day)
    d_daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
            "equity": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert abs(infer_periods_per_year(d_daily) - 252.0) < 1.0

    # Weekly (~7 calendar days between points)
    d_week = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22", "2020-01-29"]
            ),
            "equity": np.ones(5),
        }
    )
    assert abs(infer_periods_per_year(d_week) - 52.0) < 1.0

    # Monthly-ish (~30 days)
    d_mon = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "equity": np.ones(4),
        }
    )
    assert abs(infer_periods_per_year(d_mon) - 12.0) < 1.0

    # Very sparse (>180d) -> 1.0 periods/year bucket
    d_sparse = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]),
            "equity": np.ones(4),
        }
    )
    assert infer_periods_per_year(d_sparse) == 1.0


def test_infer_periods_hourly_and_five_minute() -> None:
    # Hourly bars
    t0 = pd.Timestamp("2020-01-02 09:30")
    dt_h = pd.DataFrame(
        {
            "datetime": [t0 + pd.Timedelta(hours=i) for i in range(10)],
            "equity": np.linspace(1.0, 1.05, 10),
        }
    )
    ppy_h = infer_periods_per_year(dt_h)
    expected_h = 252.0 * min(86400.0 / 3600.0, 6.5 * 3600.0 / 3600.0)
    assert abs(ppy_h - expected_h) < 0.01

    # 5-minute bars
    dt_5 = pd.DataFrame(
        {
            "datetime": [t0 + pd.Timedelta(minutes=5 * i) for i in range(20)],
            "equity": np.linspace(1.0, 1.02, 20),
        }
    )
    ppy_5 = infer_periods_per_year(dt_5)
    sec = 300.0
    intraday = min(86400.0 / sec, 6.5 * 3600.0 / sec)
    expected_5 = 252.0 * intraday
    assert abs(ppy_5 - expected_5) < 0.01


def test_infer_fallbacks_short_or_bad_index() -> None:
    assert infer_periods_per_year(pd.DataFrame()) == 252.0
    assert infer_periods_per_year(pd.DataFrame({"equity": [1.0]})) == 252.0
    two = pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=2), "equity": [1.0, 1.0]})
    assert infer_periods_per_year(two) == 252.0

    bad_dt = pd.DataFrame({"datetime": [pd.NaT, pd.NaT, pd.NaT], "equity": [1.0, 1.0, 1.0]})
    assert infer_periods_per_year(bad_dt) == 252.0


def test_infer_zero_or_negative_spacing_falls_back() -> None:
    dup = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"]),
            "equity": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert infer_periods_per_year(dup) == 252.0


def test_compute_with_trades_win_loss_profit_factor() -> None:
    n = 30
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = pd.Series(np.linspace(100_000.0, 101_000.0, n))
    df = pd.DataFrame({"datetime": dt, "equity": eq})
    trades = pd.DataFrame({"realized_pnl": [100.0, -40.0, 50.0, -10.0]})
    out = compute_metrics_from_equity(df, trades=trades, periods_per_year=252.0)
    assert int(out["num_trades"]) == 4
    assert float(out["win_rate_pct"]) == 50.0
    assert float(out["profit_factor"]) == pytest.approx(150.0 / 50.0)


def test_compute_trades_all_wins_infinite_pf_rounded() -> None:
    n = 10
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"datetime": dt, "equity": np.ones(n) * 100_000.0})
    trades = pd.DataFrame({"realized_pnl": [10.0, 20.0]})
    out = compute_metrics_from_equity(df, trades=trades)
    assert out["profit_factor"] == float("inf")


def test_compute_trades_empty_ignored() -> None:
    n = 10
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"datetime": dt, "equity": np.linspace(1.0, 1.1, n)})
    out = compute_metrics_from_equity(df, trades=pd.DataFrame())
    assert out["num_trades"] == 0


def test_zero_starting_equity_guard() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="B"),
            "equity": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    out = compute_metrics_from_equity(df, periods_per_year=252.0)
    assert float(out["total_return_pct"]) == 0.0


@pytest.mark.parametrize(
    ("excess", "ppy", "expected"),
    [
        (np.array([0.01], dtype=float), 252.0, 0.0),
        (np.array([0.01, 0.02], dtype=float), 0.0, 0.0),
        (np.zeros(5, dtype=float), 252.0, 0.0),
    ],
)
def test_annualised_sharpe_edges(excess: np.ndarray, ppy: float, expected: float) -> None:
    assert _annualised_sharpe(excess, ppy) == expected


def test_annualised_sortino_edges() -> None:
    assert _annualised_sortino(np.array([0.01], dtype=float), 252.0) == 0.0
    assert _annualised_sortino(np.array([0.01, 0.02], dtype=float), 0.0) == 0.0
    # All positive excess -> no downside obs -> inf when mean > 0
    pos = np.array([0.01, 0.02], dtype=float)
    assert math.isinf(_annualised_sortino(pos, 252.0))
    # Mean <= 0 with <2 downside observations -> 0
    neg_mean = np.array([-0.01, 0.001], dtype=float)
    assert _annualised_sortino(neg_mean, 252.0) == 0.0
    # Downside std == 0 but mean > 0 -> inf
    assert math.isinf(_annualised_sortino(np.array([0.01, 0.01], dtype=float), 252.0))


def test_annualised_sortino_normal() -> None:
    xs = np.array([0.02, -0.01, 0.015, -0.005, 0.01], dtype=float)
    s = _annualised_sortino(xs, 252.0)
    assert s > 0.0 and math.isfinite(s)


def test_annualised_sortino_downside_std_zero_positive_mean() -> None:
    """Two identical losses -> downside std 0; positive mean excess -> inf Sortino."""
    xs = np.array([0.05, -0.01, -0.01], dtype=float)
    assert math.isinf(_annualised_sortino(xs, 252.0))
