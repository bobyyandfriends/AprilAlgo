"""Shared helpers for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"


def discover_symbols() -> dict[str, list[str]]:
    """Scan the data directory and return ``{timeframe: [symbols]}`` available."""
    result: dict[str, list[str]] = {}
    if not _DATA_DIR.exists():
        return result
    for folder in sorted(_DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        tf_name = folder.name.replace("_data", "")
        symbols = sorted(
            f.stem.replace(f"_{tf_name}", "").upper()
            for f in folder.glob("*.csv")
        )
        if symbols:
            result[tf_name] = symbols
    return result


def format_metric(key: str, value) -> str:
    """Nicely format a metric value for display.

    Guards against ``None`` and ``NaN`` values (e.g. metrics from failed tuner combos
    or pages that load missing columns) so UI renders a dash instead of crashing.
    """
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        if "pct" in key or "rate" in key or "return" in key or "drawdown" in key:
            return f"{float(value):.2f}%"
        if "ratio" in key or "factor" in key:
            return f"{float(value):.2f}"
        if "pnl" in key:
            return f"${float(value):,.2f}"
        if "trades" in key:
            return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)
    return str(value)


METRIC_DISPLAY_NAMES = {
    "total_pnl": "Total P&L",
    "total_return_pct": "Return",
    "num_trades": "Trades",
    "win_rate_pct": "Win Rate",
    "avg_win": "Avg Win",
    "avg_loss": "Avg Loss",
    "profit_factor": "Profit Factor",
    "max_drawdown_pct": "Max Drawdown",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
}
