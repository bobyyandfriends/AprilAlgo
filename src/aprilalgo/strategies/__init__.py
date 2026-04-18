"""Trading strategies that plug into the backtesting engine."""

from aprilalgo.strategies.base import BaseStrategy
from aprilalgo.strategies.rsi_sma import RsiSmaStrategy
from aprilalgo.strategies.demark_confluence import DeMarkConfluenceStrategy
from aprilalgo.strategies.configurable import ConfigurableStrategy
from aprilalgo.strategies.ml_strategy import MLStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "rsi_sma": RsiSmaStrategy,
    "demark_confluence": DeMarkConfluenceStrategy,
    "configurable": ConfigurableStrategy,
    "ml_xgboost": MLStrategy,
}

__all__ = [
    "BaseStrategy",
    "RsiSmaStrategy",
    "DeMarkConfluenceStrategy",
    "ConfigurableStrategy",
    "MLStrategy",
    "STRATEGIES",
]
