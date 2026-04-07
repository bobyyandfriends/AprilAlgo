"""Abstract base strategy — all strategies inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio


class BaseStrategy(ABC):
    """Every strategy must implement ``init`` and ``on_bar``."""

    name: str = "BaseStrategy"

    def __init__(self, **params) -> None:
        self.params = params

    def init(self, price_data: pd.DataFrame) -> None:
        """Called once before the backtest loop.

        Use this to pre-compute indicators on the full DataFrame so ``on_bar``
        can look them up by index.
        """

    @abstractmethod
    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        """Called once per bar during the backtest.

        *idx* is the integer position in the pre-computed DataFrame.
        *row* contains OHLCV plus any columns added during ``init``.
        Use ``portfolio.open_trade`` / ``portfolio.close_trade`` to act.
        """
