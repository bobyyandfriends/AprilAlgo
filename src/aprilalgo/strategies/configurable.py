"""Configurable strategy — compose indicator pipelines from config, not code.

Instead of writing a new Python class for every indicator combination,
define which indicators to use via a list of dicts. The strategy applies
them all, scores confluence, and trades based on the net score.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.indicators.registry import IndicatorRegistry
from aprilalgo.confluence.scorer import score_confluence
from aprilalgo.strategies.base import BaseStrategy


class ConfigurableStrategy(BaseStrategy):
    """Strategy driven by confluence scoring across a configurable indicator set.

    Parameters
    ----------
    indicators : List of dicts, e.g. ``[{"name": "rsi", "period": 14}, ...]``.
                 If None, uses a sensible default set.
    entry_threshold : Minimum ``confluence_net`` to open a long (or ``-threshold`` for short).
    exit_threshold : Close when ``confluence_net`` reverses past this level.
    stop_loss_pct : Maximum drawdown before stop-loss exit.
    direction : ``"long"``, ``"short"``, or ``"both"``.
    position_pct : Fraction of cash to deploy per trade.
    """

    name = "configurable"

    def __init__(
        self,
        indicators: list[dict[str, Any]] | None = None,
        entry_threshold: float = 0.3,
        exit_threshold: float = -0.1,
        stop_loss_pct: float = 0.03,
        direction: str = "long",
        position_pct: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.indicator_configs = indicators or [
            {"name": "rsi", "period": 14},
            {"name": "sma", "period": 20},
            {"name": "bollinger_bands", "period": 20},
            {"name": "volume_trend"},
            {"name": "demark"},
        ]
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.direction = direction
        self.position_pct = position_pct
        self._data: pd.DataFrame = pd.DataFrame()

    def init(self, price_data: pd.DataFrame) -> None:
        pipeline = IndicatorRegistry.from_config(
            [dict(cfg) for cfg in self.indicator_configs]
        )
        df = pipeline.apply(price_data)
        df = score_confluence(df)
        self._data = df

    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        enriched = self._data.iloc[idx]
        close = row["close"]
        time = row["datetime"]

        conf_net = enriched.get("confluence_net", 0.0)
        if pd.isna(conf_net):
            return

        if not portfolio.has_open_position:
            should_long = (
                self.direction in ("long", "both")
                and conf_net >= self.entry_threshold
            )
            should_short = (
                self.direction in ("short", "both")
                and conf_net <= -self.entry_threshold
            )

            if should_long:
                shares = int(portfolio.cash * self.position_pct / close)
                if shares > 0:
                    portfolio.open_trade(time, close, side="long", quantity=shares)
            elif should_short:
                shares = int(portfolio.cash * self.position_pct / close)
                if shares > 0:
                    portfolio.open_trade(time, close, side="short", quantity=shares)
        else:
            for trade in list(portfolio.open_positions):
                entry = trade.entry_price
                if trade.side == "long":
                    drawdown = (entry - close) / entry
                    exit_signal = conf_net <= self.exit_threshold
                else:
                    drawdown = (close - entry) / entry
                    exit_signal = conf_net >= -self.exit_threshold

                if exit_signal or drawdown >= self.stop_loss_pct:
                    portfolio.close_trade(trade, time, close)
