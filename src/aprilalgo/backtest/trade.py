"""Trade object — represents a single completed or open trade."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """A single trade with entry/exit info and P&L tracking."""

    entry_time: datetime
    entry_price: float
    side: str  # "long" or "short"
    quantity: float = 1.0
    exit_time: datetime | None = None
    exit_price: float | None = None
    commission: float = 0.0
    slippage: float = 0.0
    closed: bool = False
    realized_pnl: float = 0.0

    def close(self, exit_time: datetime, exit_price: float) -> None:
        """Close the trade and compute realized P&L."""
        if self.closed:
            return
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.closed = True

        if self.side == "long":
            raw = (exit_price - self.entry_price) * self.quantity
        else:
            raw = (self.entry_price - exit_price) * self.quantity

        self.realized_pnl = raw - self.commission - self.slippage

    def unrealized_pnl(self, current_price: float) -> float:
        """Mark-to-market P&L for an open position."""
        if self.closed:
            return self.realized_pnl
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "side": self.side,
            "quantity": self.quantity,
            "realized_pnl": self.realized_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "closed": self.closed,
        }
