"""RSI + SMA crossover strategy — the first working strategy for AprilAlgo."""

from __future__ import annotations

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.indicators.rsi import rsi
from aprilalgo.indicators.sma import sma
from aprilalgo.strategies.base import BaseStrategy


class RsiSmaStrategy(BaseStrategy):
    """Long-only strategy combining RSI oversold/overbought with SMA trend filter.

    **Buy** when RSI crosses below ``rsi_buy`` AND close is above the SMA.
    **Sell** when RSI crosses above ``rsi_sell`` OR close drops below the SMA.
    """

    name = "rsi_sma"

    def __init__(
        self,
        rsi_period: int = 14,
        sma_period: int = 50,
        rsi_buy: float = 30.0,
        rsi_sell: float = 70.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self._data: pd.DataFrame = pd.DataFrame()

    def init(self, price_data: pd.DataFrame) -> None:
        df = rsi(price_data, period=self.rsi_period)
        df = sma(df, period=self.sma_period)
        self._data = df

    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        enriched = self._data.iloc[idx]
        rsi_val = enriched.get(f"rsi_{self.rsi_period}")
        sma_val = enriched.get(f"sma_{self.sma_period}")

        if pd.isna(rsi_val) or pd.isna(sma_val):
            return

        close = row["close"]
        time = row["datetime"]

        if not portfolio.has_open_position:
            if rsi_val < self.rsi_buy and close > sma_val:
                shares = int(portfolio.cash * 0.95 / close)
                if shares > 0:
                    portfolio.open_trade(time, close, side="long", quantity=shares)
        else:
            if rsi_val > self.rsi_sell or close < sma_val:
                for trade in list(portfolio.open_positions):
                    portfolio.close_trade(trade, time, close)
