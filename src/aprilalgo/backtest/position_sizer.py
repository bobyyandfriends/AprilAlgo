"""Position sizing methods — Fractional Kelly, fixed percentage, ATR-based."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PositionSizer(ABC):
    """Abstract base for position sizing strategies."""

    @abstractmethod
    def size(self, capital: float, price: float, **context) -> int:
        """Return the number of shares/units to trade."""


class FixedFraction(PositionSizer):
    """Risk a fixed fraction of capital per trade.

    Parameters
    ----------
    fraction : Fraction of capital to allocate (e.g. 0.02 = 2%).
    """

    def __init__(self, fraction: float = 0.02) -> None:
        self.fraction = fraction

    def size(self, capital: float, price: float, **context) -> int:
        risk_amount = capital * self.fraction
        return max(1, int(risk_amount / price))


class FractionalKelly(PositionSizer):
    """Fractional Kelly Criterion — size based on win probability and edge.

    Full Kelly: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio

    We use a fraction of Kelly (default half) to reduce variance.

    Parameters
    ----------
    kelly_fraction : Fraction of full Kelly to use (0.5 = half-Kelly).
    max_position_pct : Maximum fraction of capital in one trade.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.25,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def size(self, capital: float, price: float, **context) -> int:
        win_prob = context.get("win_prob", 0.5)
        avg_win = context.get("avg_win", 1.0)
        avg_loss = context.get("avg_loss", 1.0)

        if avg_loss == 0:
            return 1

        b = abs(avg_win / avg_loss)
        q = 1.0 - win_prob
        kelly_f = (win_prob * b - q) / b

        if kelly_f <= 0:
            return 0

        position_frac = kelly_f * self.kelly_fraction
        position_frac = min(position_frac, self.max_position_pct)

        dollar_amount = capital * position_frac
        return max(0, int(dollar_amount / price))


class ATRBased(PositionSizer):
    """Size positions so that 1 ATR of movement = a fixed dollar risk.

    Parameters
    ----------
    risk_per_trade : Dollar amount willing to risk per trade.
    atr_multiplier : How many ATRs define the stop distance.
    """

    def __init__(
        self,
        risk_per_trade: float = 1000.0,
        atr_multiplier: float = 2.0,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier

    def size(self, capital: float, price: float, **context) -> int:
        atr = context.get("atr", price * 0.02)
        if atr <= 0:
            return 1
        stop_distance = atr * self.atr_multiplier
        shares = int(self.risk_per_trade / stop_distance)
        max_shares = int(capital * 0.25 / price)
        return max(1, min(shares, max_shares))


SIZERS: dict[str, type[PositionSizer]] = {
    "fixed_fraction": FixedFraction,
    "fractional_kelly": FractionalKelly,
    "atr_based": ATRBased,
}
