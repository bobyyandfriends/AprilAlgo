"""Symbol universe management — load watchlists from YAML or text files."""

from __future__ import annotations

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_universe(path: str | Path | None = None) -> list[str]:
    """Load a symbol list from a YAML or text file.

    YAML format: ``symbols: [AAPL, MSFT, ...]``
    Text format: one symbol per line.

    Falls back to ``configs/symbols.yaml`` if *path* is None.
    """
    if path is None:
        path = _PROJECT_ROOT / "configs" / "symbols.yaml"
    path = Path(path)

    if not path.exists():
        return _default_symbols()

    if path.suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return [s.upper() for s in data.get("symbols", [])]

    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


def _default_symbols() -> list[str]:
    """Fallback symbol list if no config file found."""
    return [
        "AAPL", "AMZN", "GOOG", "META", "MSFT",
        "NVDA", "TSLA", "AMD", "AVGO", "INTC",
    ]
