"""Tests for volatility regime tagging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aprilalgo.meta.regime import add_vol_regime, realized_vol


def test_realized_vol_positive():
    c = pd.Series(np.linspace(100, 110, 50))
    v = realized_vol(c, window=5)
    assert (v.dropna() >= 0).all()


def test_add_vol_regime_hmm_smoke():
    # hmmlearn has no wheels / requires MSVC on some Windows+Python combos; use optional extra.
    pytest.importorskip("hmmlearn")
    n = 120
    df = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.default_rng(2).normal(0, 0.4, n)),
        }
    )
    out = add_vol_regime(df, window=10, n_buckets=3, use_hmm=True)
    assert "vol_regime" in out.columns
    assert out["vol_regime"].notna().sum() == n


def test_add_vol_regime_column():
    n = 80
    df = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n)),
        }
    )
    out = add_vol_regime(df, window=10, n_buckets=3)
    assert "vol_regime" in out.columns
    assert out["vol_regime"].notna().sum() > 0
