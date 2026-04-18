"""JSONL signal / outcome logging for ML and research (v0.3).

Each line is one JSON object. Schema is documented in ``docs/DATA_SCHEMA.md`` §6.
v0.4: meta-labeling and reporting read the same files — extend keys, do not rename.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Minimum keys for a well-formed event
REQUIRED_EVENT_KEYS: frozenset[str] = frozenset({"ts", "symbol"})

# Full contract (v0.3+) — values may be ``null`` when unknown at log time.
FULL_SIGNAL_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "ts",
        "symbol",
        "tf",
        "model_id",
        "features_hash",
        "pred",
        "pred_proba",
        "label_multiclass",
        "label_binary",
        "meta_pred",
        "pred_proba_meta",
        "outcome",
        "pnl",
    }
)

SchemaMode = Literal["minimal", "full"]


def validate_event(
    event: dict[str, Any],
    *,
    schema: SchemaMode = "minimal",
) -> None:
    """Raise ``ValueError`` if required keys are missing."""
    required = FULL_SIGNAL_EVENT_KEYS if schema == "full" else REQUIRED_EVENT_KEYS
    missing = required - event.keys()
    if missing:
        raise ValueError(f"Signal event missing keys {sorted(missing)}: {event}")


def hash_features_row(row: pd.DataFrame) -> str:
    """Stable short hash for one feature row (first row of *row*).

    Hardened against two real-world failure modes seen in the UI/backtest:

    * Empty frames (``row`` has zero rows) — previously raised ``IndexError``
      on ``row.iloc[0]``. We now return a sentinel so callers can still log
      the event without crashing.
    * Non-numeric / mixed-dtype frames (e.g. a string ``regime`` column, or
      datetimes that leaked into the feature matrix) — ``to_numpy(float64)``
      would raise ``TypeError``. We coerce with ``pd.to_numeric`` first and
      fall back to the string representation if coercion still fails, so the
      hash stays stable for a given row shape.
    """
    if row is None or getattr(row, "empty", True) or len(row) == 0:
        return "empty"

    series = row.iloc[0]
    try:
        coerced = pd.to_numeric(series, errors="coerce")
        vals = np.ascontiguousarray(coerced.to_numpy(dtype=np.float64, copy=False))
        payload = vals.tobytes()
    except (TypeError, ValueError):
        # Last-ditch fallback: use the repr of each cell joined with NULs.
        payload = "\x00".join(repr(v) for v in series.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class SignalJsonlLogger:
    """Append-only JSONL writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any], *, strict_keys: bool = True) -> None:
        if strict_keys:
            validate_event(event)
        line = json.dumps(event, default=str, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    out.append(json.loads(raw))
        return out


def log_event(
    logger: SignalJsonlLogger,
    event: dict[str, Any],
    *,
    schema: SchemaMode = "minimal",
) -> None:
    """Validate and append one JSONL event (convenience for strategies)."""
    validate_event(event, schema=schema)
    logger.log(event, strict_keys=False)


def events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten JSON events to a DataFrame (best-effort)."""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)
