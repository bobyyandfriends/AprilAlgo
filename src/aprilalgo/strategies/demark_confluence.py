"""DeMark + Multi-Indicator Confluence strategy.

Uses DeMark Sequential as the primary trigger, confirmed by confluence
scoring across RSI, Bollinger Bands, Volume Trend, and optional
Super Smoother trend filter.

Buys when: DeMark fires a bullish signal AND confluence score exceeds threshold.
Sells when: DeMark fires a bearish signal OR confluence flips negative OR stop hit.
"""

from __future__ import annotations

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.confluence.scorer import score_confluence
from aprilalgo.indicators.bollinger import bollinger_bands
from aprilalgo.indicators.demark import demark
from aprilalgo.indicators.ehlers import super_smoother
from aprilalgo.indicators.rsi import rsi
from aprilalgo.indicators.sma import sma
from aprilalgo.indicators.volume_trend import volume_trend
from aprilalgo.strategies.base import BaseStrategy


class DeMarkConfluenceStrategy(BaseStrategy):
    """DeMark exhaustion signals confirmed by multi-indicator confluence.

    Opens long when td_bull fires and confluence_net > threshold.
    Closes on td_bear, confluence flip, or stop loss.
    """

    name = "demark_confluence"

    def __init__(
        self,
        rsi_period: int = 14,
        sma_period: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ss_period: int = 10,
        vol_period: int = 20,
        confluence_threshold: float = 0.3,
        stop_loss_pct: float = 0.03,
        position_pct: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ss_period = ss_period
        self.vol_period = vol_period
        self.confluence_threshold = confluence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.position_pct = position_pct
        self._data: pd.DataFrame = pd.DataFrame()

    def init(self, price_data: pd.DataFrame) -> None:
        df = demark(price_data)
        df = rsi(df, period=self.rsi_period)
        df = sma(df, period=self.sma_period)
        df = bollinger_bands(df, period=self.bb_period, std_dev=self.bb_std)
        df = volume_trend(df, vol_period=self.vol_period)
        df = super_smoother(df, period=self.ss_period)
        df = score_confluence(df)
        self._data = df
        self._backtest_bars_df = df

    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        enriched = self._data.iloc[idx]
        close = row["close"]
        time = row["datetime"]

        conf_net = enriched.get("confluence_net", 0.0)
        if pd.isna(conf_net):
            return

        td_bull = enriched.get("td_bull", False)
        td_bear = enriched.get("td_bear", False)

        if not portfolio.has_open_position:
            if td_bull and conf_net > self.confluence_threshold:
                shares = int(portfolio.cash * self.position_pct / close)
                if shares > 0:
                    portfolio.open_trade(time, close, side="long", quantity=shares)
        else:
            for trade in list(portfolio.open_positions):
                entry = trade.entry_price
                drawdown = (entry - close) / entry

                should_exit = td_bear or conf_net < -self.confluence_threshold or drawdown >= self.stop_loss_pct

                if should_exit:
                    portfolio.close_trade(trade, time, close)
