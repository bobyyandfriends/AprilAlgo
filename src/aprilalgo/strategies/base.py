"""Abstract base strategy — all strategies inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from aprilalgo.backtest.portfolio import Portfolio


class BaseStrategy(ABC):
    """Every strategy must implement ``init`` and ``on_bar``.

    Loop-frame contract
    -------------------
    After :meth:`init` returns, :attr:`_backtest_bars_df` **must** be set to the
    DataFrame the engine should iterate over. This is the frame whose
    ``(idx, row)`` pairs are passed to :meth:`on_bar`, and it must be identical
    in length to whatever internal enriched frame the strategy uses for lookups
    (typically ``self._data``) — otherwise ``on_bar`` will mis-index silently.

    * Strategies that preserve the input row count (identity indicator
      pipelines) should set ``self._backtest_bars_df = self._data`` and leave
      :attr:`_backtest_frame_matches_input` at its default ``True``. The engine
      will assert ``len(_backtest_bars_df) == len(price_data)``.
    * Strategies that legitimately resample or filter bars (e.g. information
      bars inside :class:`~aprilalgo.strategies.ml_strategy.MLStrategy`) must
      set :attr:`_backtest_frame_matches_input` to ``False`` so the engine
      skips the length check.
    """

    name: str = "BaseStrategy"
    _backtest_frame_matches_input: bool = True

    def __init__(self, **params) -> None:
        self.params = params
        self._backtest_bars_df: pd.DataFrame | None = None

    def init(self, price_data: pd.DataFrame) -> None:
        """Called once before the backtest loop.

        Use this to pre-compute indicators on the full DataFrame so ``on_bar``
        can look them up by index. The default implementation sets
        :attr:`_backtest_bars_df` to *price_data* (identity loop); subclasses
        that override it must either call ``super().init(price_data)`` or set
        :attr:`_backtest_bars_df` themselves before returning.
        """
        self._backtest_bars_df = price_data

    @abstractmethod
    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        """Called once per bar during the backtest.

        *idx* is the integer position in :attr:`_backtest_bars_df`.
        *row* contains OHLCV plus any columns added during ``init``.
        Use ``portfolio.open_trade`` / ``portfolio.close_trade`` to act.
        """
