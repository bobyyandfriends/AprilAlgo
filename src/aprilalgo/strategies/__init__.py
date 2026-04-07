"""Trading strategies that plug into the backtesting engine."""

from aprilalgo.strategies.base import BaseStrategy
from aprilalgo.strategies.rsi_sma import RsiSmaStrategy
from aprilalgo.strategies.demark_confluence import DeMarkConfluenceStrategy
from aprilalgo.strategies.configurable import ConfigurableStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "rsi_sma": RsiSmaStrategy,
    "demark_confluence": DeMarkConfluenceStrategy,
    "configurable": ConfigurableStrategy,
}

__all__ = [
    "BaseStrategy", "RsiSmaStrategy", "DeMarkConfluenceStrategy",
    "ConfigurableStrategy", "STRATEGIES",
]
