"""Central indicator registry — pipeline builder and catalog access."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

IndicatorFn = Callable[..., pd.DataFrame]


class IndicatorRegistry:
    """Collect indicator functions and apply them sequentially to a DataFrame.

    Supports both raw function references and catalog-based lookup by name.
    """

    def __init__(self) -> None:
        self._indicators: list[tuple[IndicatorFn, dict]] = []

    def add(self, fn: IndicatorFn, **kwargs: Any) -> "IndicatorRegistry":
        """Register an indicator function with optional keyword arguments."""
        self._indicators.append((fn, kwargs))
        return self

    def add_by_name(self, name: str, **kwargs: Any) -> "IndicatorRegistry":
        """Register an indicator by its catalog name (e.g., ``'rsi'``)."""
        from aprilalgo.indicators.descriptor import get_catalog

        catalog = get_catalog()
        if name not in catalog:
            raise KeyError(f"Unknown indicator '{name}'. Available: {list(catalog.keys())}")
        spec = catalog[name]
        merged = spec.default_params()
        merged.update(kwargs)
        if spec._param_transform:
            merged = spec._param_transform(merged)
        self._indicators.append((spec.fn, merged))
        return self

    @classmethod
    def from_config(cls, indicator_list: list[dict[str, Any]]) -> "IndicatorRegistry":
        """Build a pipeline from a list of dicts.

        Example::

            IndicatorRegistry.from_config([
                {"name": "rsi", "period": 14},
                {"name": "sma", "period": 20},
            ])
        """
        reg = cls()
        for item in indicator_list:
            item = dict(item)
            name = item.pop("name")
            reg.add_by_name(name, **item)
        return reg

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every registered indicator on *df*, returning the enriched copy."""
        result = df.copy()
        for fn, kwargs in self._indicators:
            result = fn(result, **kwargs)
        return result

    def clear(self) -> None:
        self._indicators.clear()

    def __len__(self) -> int:
        return len(self._indicators)


def apply_indicators(df: pd.DataFrame, indicators: list[tuple[IndicatorFn, dict]]) -> pd.DataFrame:
    """Convenience: apply a list of ``(fn, kwargs)`` pairs to *df*."""
    result = df.copy()
    for fn, kwargs in indicators:
        result = fn(result, **kwargs)
    return result
