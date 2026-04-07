"""Portfolio — tracks cash, open positions, equity curve, and trade log."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from aprilalgo.backtest.trade import Trade


class Portfolio:
    """Simulates a brokerage account during a backtest."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.open_positions: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    # ------------------------------------------------------------------
    def open_trade(self, time: datetime, price: float, side: str, quantity: float = 1.0) -> Trade:
        """Open a new position and deduct cost from cash."""
        cost = price * quantity
        slip = price * self.slippage * quantity
        comm = self.commission

        if side == "long":
            self.cash -= cost + slip + comm
        else:
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
        """Close an existing open trade and credit proceeds to cash."""
        trade.close(time, price)

        slip = price * self.slippage * trade.quantity
        comm = self.commission

        if trade.side == "long":
            self.cash += price * trade.quantity - slip - comm
        else:
            self.cash -= price * trade.quantity + slip + comm

        if trade in self.open_positions:
            self.open_positions.remove(trade)
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------
    def record_equity(self, time: datetime, price: float) -> None:
        """Snapshot total equity (cash + mark-to-market) at *time*."""
        mtm = sum(t.unrealized_pnl(price) for t in self.open_positions)
        self.equity_curve.append({
            "time": time,
            "equity": self.cash + mtm + sum(
                t.entry_price * t.quantity for t in self.open_positions if t.side == "long"
            ),
        })

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.closed_trades])

    @property
    def has_open_position(self) -> bool:
        return len(self.open_positions) > 0
