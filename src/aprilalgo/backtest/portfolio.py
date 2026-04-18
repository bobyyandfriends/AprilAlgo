"""Portfolio — tracks cash, open positions, equity curve, and trade log."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from aprilalgo.backtest.trade import Trade


class Portfolio:
    """Simulates a brokerage account during a backtest.

    Parameters
    ----------
    initial_capital
        Starting cash balance.
    commission
        Flat per-trade commission applied on entry and exit.
    slippage
        Fractional slippage (e.g. ``0.0005`` = 5 bps) applied to ``price * qty``.
    margin_ratio
        Optional short-sale initial margin fraction. When set (e.g. ``0.5`` for
        US Reg-T), opening a short requires existing cash collateral of at
        least ``entry_price * quantity * margin_ratio``; otherwise the call
        raises :class:`ValueError`. ``None`` (the default) disables the check
        and reproduces the legacy "unlimited short" behaviour — accurate only
        for strategies that never go short.
    borrow_rate_bps_per_day
        Optional daily stock-borrow cost in basis points of short notional.
        Accrued at each :meth:`record_equity` call based on the elapsed time
        since the previous snapshot and the mark-to-market notional of every
        open short. Zero by default (back-compat).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        *,
        margin_ratio: float | None = None,
        borrow_rate_bps_per_day: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_ratio = margin_ratio
        self.borrow_rate_bps_per_day = float(borrow_rate_bps_per_day)

        self.open_positions: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self._last_equity_time: datetime | None = None
        self._accrued_borrow: float = 0.0

    # ------------------------------------------------------------------
    def open_trade(self, time: datetime, price: float, side: str, quantity: float = 1.0) -> Trade:
        """Open a new position and deduct cost from cash.

        When :attr:`margin_ratio` is set, a short open consumes
        ``price * quantity * margin_ratio`` of available cash as collateral
        (implicit: ``cash`` doubles as margin equity in this simplified model).
        """
        cost = price * quantity
        slip = price * self.slippage * quantity
        comm = self.commission

        if side == "long":
            self.cash -= cost + slip + comm
        else:
            if self.margin_ratio is not None:
                required = cost * self.margin_ratio
                if self.cash < required:
                    raise ValueError(
                        f"Insufficient cash for short margin: need "
                        f"{required:.2f} collateral ({self.margin_ratio:.2%} of "
                        f"{cost:.2f} notional), have {self.cash:.2f}."
                    )
            self.cash += cost - slip - comm

        trade = Trade(
            entry_time=time,
            entry_price=price,
            side=side,
            quantity=quantity,
            commission=comm,
            slippage=slip,
        )
        self.open_positions.append(trade)
        return trade

    def close_trade(self, trade: Trade, time: datetime, price: float) -> None:
        """Close an existing open trade and credit proceeds to cash.

        The exit commission and slippage are accumulated onto the *Trade* before
        :meth:`Trade.close` computes ``realized_pnl``, so reported P&L reflects the
        full round-trip cost instead of only the entry-side charges.
        """
        exit_slip = price * self.slippage * trade.quantity
        exit_comm = self.commission
        trade.commission += exit_comm
        trade.slippage += exit_slip
        trade.close(time, price)

        if trade.side == "long":
            self.cash += price * trade.quantity - exit_slip - exit_comm
        else:
            self.cash -= price * trade.quantity + exit_slip + exit_comm

        # Identity-match removal — dataclass __eq__ is structural, so ``list.remove``
        # could otherwise pick up a differently-intended trade with identical fields.
        for i, open_t in enumerate(self.open_positions):
            if open_t is trade:
                del self.open_positions[i]
                break
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------
    def record_equity(self, time: datetime, price: float) -> None:
        """Snapshot total equity (cash + mark-to-market) at *time*.

        Equity convention:

        * For long opens, :meth:`open_trade` already debited ``entry_price * qty`` from
          cash, so we add the long entry value back (cancelling the debit) and apply
          the mark-to-market P&L.
        * For short opens, :meth:`open_trade` credited ``entry_price * qty`` to cash
          as short-sale proceeds; the outstanding liability equal to that same amount
          must be subtracted so the snapshot reflects true equity, not inflated cash.

        When :attr:`borrow_rate_bps_per_day` is positive, the elapsed time since the
        previous snapshot is used to debit cash by
        ``short_notional * rate_bps / 10_000`` per day. The current mark-to-market
        price is used as the short notional, matching how real brokers re-price the
        borrow daily.
        """
        if self.borrow_rate_bps_per_day > 0.0 and self._last_equity_time is not None:
            elapsed_days = (time - self._last_equity_time).total_seconds() / 86400.0
            if elapsed_days > 0.0:
                short_notional = sum(price * t.quantity for t in self.open_positions if t.side == "short")
                charge = short_notional * (self.borrow_rate_bps_per_day / 10_000.0) * elapsed_days
                self.cash -= charge
                self._accrued_borrow += charge

        mtm = sum(t.unrealized_pnl(price) for t in self.open_positions)
        open_adjust = sum(t.entry_price * t.quantity * (1 if t.side == "long" else -1) for t in self.open_positions)
        self.equity_curve.append(
            {
                "time": time,
                "equity": self.cash + mtm + open_adjust,
            }
        )
        self._last_equity_time = time

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.closed_trades])

    @property
    def has_open_position(self) -> bool:
        return len(self.open_positions) > 0
