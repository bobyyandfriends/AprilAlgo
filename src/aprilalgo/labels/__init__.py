"""Labeling utilities for ML (v0.3+)."""

from aprilalgo.labels.targets import build_triple_barrier_targets
from aprilalgo.labels.triple_barrier import (
    LABEL_STOP_LOSS,
    LABEL_TAKE_PROFIT,
    LABEL_VERTICAL_TIMEOUT,
    TripleBarrierResult,
    apply_triple_barrier,
    label_inclusive_end_ix,
)

__all__ = [
    "LABEL_STOP_LOSS",
    "LABEL_TAKE_PROFIT",
    "LABEL_VERTICAL_TIMEOUT",
    "TripleBarrierResult",
    "apply_triple_barrier",
    "build_triple_barrier_targets",
    "label_inclusive_end_ix",
]
