"""Define parameter grids for indicator optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any


@dataclass
class ParameterGrid:
    """Define parameter ranges for one or more indicators.

    Example:
        grid = ParameterGrid()
        grid.add("rsi", period=[10, 12, 14, 16], oversold=[25, 30, 35])
        grid.add("sma", period=[15, 20, 25, 50])
        combos = grid.generate()  # list of dicts
    """

    _specs: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def add(self, indicator: str, **param_ranges: list[Any]) -> "ParameterGrid":
        """Add parameter ranges for an indicator.

        Each keyword is a parameter name, and its value is a list of values to try.
        """
        self._specs[indicator] = param_ranges
        return self

    def generate(self) -> list[dict[str, dict[str, Any]]]:
        """Generate all combinations across all indicators.

        Returns a list where each element is a dict mapping
        indicator name → {param: value} for one specific combination.
        """
        if not self._specs:
            return [{}]

        indicator_names = list(self._specs.keys())
        per_indicator_combos: list[list[dict[str, Any]]] = []

        for name in indicator_names:
            params = self._specs[name]
            keys = list(params.keys())
            values = list(params.values())
            combos = [dict(zip(keys, v)) for v in product(*values)]
            per_indicator_combos.append(combos)

        all_combos = []
        for combo_tuple in product(*per_indicator_combos):
            entry = {}
            for name, combo in zip(indicator_names, combo_tuple):
                entry[name] = combo
            all_combos.append(entry)

        return all_combos

    @property
    def total_combinations(self) -> int:
        count = 1
        for params in self._specs.values():
            indicator_combos = 1
            for values in params.values():
                indicator_combos *= len(values)
            count *= indicator_combos
        return count

    def __repr__(self) -> str:
        parts = [f"{name}: {params}" for name, params in self._specs.items()]
        return f"ParameterGrid({', '.join(parts)}) → {self.total_combinations} combos"
