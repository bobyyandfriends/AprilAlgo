"""Sequential bootstrap–style sample weights from label interval overlap (v0.4).

Uniqueness weight: inverse of how many samples' ``[t0,t1]`` intervals overlap
the sample's interval (including self). Heavier overlap -> lower weight.
"""

from __future__ import annotations

import numpy as np


def overlap_count_matrix(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """``M[i,j] = 1`` if intervals ``[t0_i,t1_i]`` and ``[t0_j,t1_j]`` overlap (inclusive)."""
    n = len(t0)
    t0i = t0.reshape(-1, 1)
    t1i = t1.reshape(-1, 1)
    t0j = t0.reshape(1, -1)
    t1j = t1.reshape(1, -1)
    overlap = (np.maximum(t0i, t0j) <= np.minimum(t1i, t1j)).astype(np.float64)
    return overlap


def uniqueness_weights(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Return nonnegative weights summing to ``n`` (normalize for mean weight1)."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    m = overlap_count_matrix(t0, t1)
    counts = m.sum(axis=1)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (len(w) / np.sum(w))
    return w


def sequential_bootstrap_sample(
    t0: np.ndarray,
    t1: np.ndarray,
    *,
    n_draw: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Draw row indices with replacement; probabilities from :func:`uniqueness_weights`.

    Non-overlapping labels (pairwise disjoint intervals) yield uniform weights.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    n = len(t0)
    n_draw = int(n if n_draw is None else n_draw)
    w = uniqueness_weights(t0, t1)
    p = w / w.sum()
    rng = np.random.default_rng(random_state)
    return rng.choice(np.arange(n, dtype=np.int64), size=n_draw, replace=True, p=p)
