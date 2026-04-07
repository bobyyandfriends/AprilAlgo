"""Run backtests across parameter grid combinations."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from aprilalgo.backtest.engine import run_backtest
from aprilalgo.strategies.base import BaseStrategy
from aprilalgo.tuner.grid import ParameterGrid


class TunerRunner:
    """Execute a strategy backtest for every parameter combination in a grid.

    Parameters
    ----------
    strategy_class : The strategy class to instantiate for each combo.
    price_data : OHLCV DataFrame to backtest on.
    grid : ParameterGrid defining parameter ranges.
    initial_capital : Starting cash.
    commission : Per-trade commission.
    slippage : Per-trade slippage fraction.
    metric : Which metric to optimize (key from calculate_metrics output).
    """

    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        price_data: pd.DataFrame,
        grid: ParameterGrid,
        initial_capital: float = 100_000.0,
        commission: float = 0.0,
        slippage: float = 0.0005,
        metric: str = "sharpe_ratio",
    ) -> None:
        self.strategy_class = strategy_class
        self.price_data = price_data
        self.grid = grid
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.metric = metric
        self.results: list[dict[str, Any]] = []

    def run(self, progress_callback: Callable[[int, int], None] | None = None) -> pd.DataFrame:
        """Execute all combinations and return a results DataFrame.

        Columns: one per parameter + all metric columns + ``combo_id``.
        """
        combos = self.grid.generate()
        total = len(combos)
        self.results.clear()

        for i, combo in enumerate(combos):
            flat_params = self._flatten_params(combo)

            try:
                strategy = self.strategy_class(**flat_params)
                result = run_backtest(
                    strategy=strategy,
                    price_data=self.price_data.copy(),
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage,
                )
                row = {"combo_id": i, **flat_params, **result["metrics"]}
            except Exception as exc:
                row = {"combo_id": i, **flat_params, "error": str(exc)}

            self.results.append(row)

            if progress_callback:
                progress_callback(i + 1, total)

        return pd.DataFrame(self.results)

    @staticmethod
    def _flatten_params(combo: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Flatten nested indicator params into a single dict for strategy __init__."""
        flat: dict[str, Any] = {}
        for _indicator, params in combo.items():
            flat.update(params)
        return flat
