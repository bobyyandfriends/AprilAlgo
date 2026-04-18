"""Streamlit: multi-symbol backtest summary (v0.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from aprilalgo.backtest.portfolio_runner import run_multi_symbol_backtests
from aprilalgo.strategies.rsi_sma import RsiSmaStrategy

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def render() -> None:
    st.title("Portfolio lab")
    st.caption("Runs independent backtests per symbol (same initial capital each).")
    syms = st.text_input("Symbols (comma-separated)", value="TEST")
    use_fixture = st.checkbox("Use tests/fixtures", value=True)
    data_dir = _PROJECT_ROOT / "tests" / "fixtures" if use_fixture else None

    symbol_list = [s.strip().upper() for s in syms.split(",") if s.strip()]

    def strat(_sym: str) -> RsiSmaStrategy:
        return RsiSmaStrategy(rsi_period=14, sma_period=20, rsi_buy=35, rsi_sell=65)

    if st.button("Run"):
        rows = []
        try:
            results = run_multi_symbol_backtests(
                strat,
                symbol_list,
                timeframe="daily",
                data_dir=data_dir,
            )
        except Exception as e:
            st.error(str(e))
            return
        for sym, res in results.items():
            m = res["metrics"]
            rows.append(
                {
                    "symbol": sym,
                    "num_trades": m.get("num_trades"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                    "total_return_pct": m.get("total_return_pct"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
