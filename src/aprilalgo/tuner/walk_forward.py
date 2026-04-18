"""Walk-forward train/test index splits (v0.4).

Yields increasing train windows and forward test blocks — combine with
:class:`aprilalgo.ml.cv.PurgedKFold` inside each train window when fitting.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def walk_forward_splits(
    n: int,
    *,
    n_folds: int = 4,
    min_train: int = 50,
    test_size: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer position arrays.

    When *test_size* is ``None`` it is computed as
    ``ceil((n - min_train) / n_folds)`` so the iterator emits exactly
    *n_folds* splits that together tile ``[min_train, n)``. Previously the
    floor-based default could emit ``n_folds + 1`` windows when
    ``(n - min_train)`` was not divisible by ``n_folds`` (§AUDIT B9).

    When an explicit *test_size* is supplied, we preserve the legacy
    "cover-the-tail" behaviour: every window that fits in ``[min_train, n)``
    is yielded (caller is responsible if they want exactly *n_folds*).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if min_train >= n:
        raise ValueError("min_train must be < n")
    auto_sized = test_size is None
    if auto_sized:
        # Ceiling division so *n_folds* auto-sized windows cover the tail.
        test_size_eff: int = max(1, -(-(n - min_train) // n_folds))
    else:
        if test_size is None:
            raise ValueError("test_size must be set when relying on explicit window sizing")
        test_size_eff = test_size
    start_test = min_train
    emitted = 0
    while start_test < n:
        if auto_sized and emitted >= n_folds:
            break
        end_test = min(start_test + test_size_eff, n)
        train_idx = np.arange(0, start_test, dtype=np.int64)
        test_idx = np.arange(start_test, end_test, dtype=np.int64)
        if test_idx.size == 0:
            break
        yield train_idx, test_idx
        emitted += 1
        start_test = end_test


def walk_forward_summary(
    n: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    """Summarize split coverage and average train/test sizes."""
    if not splits:
        return {
            "n_splits": 0,
            "coverage_pct": 0.0,
            "mean_train_size": 0.0,
            "mean_test_size": 0.0,
        }
    tests = np.concatenate([te for _tr, te in splits])
    coverage = 100.0 * float(pd.Series(tests).nunique()) / float(max(n, 1))
    train_sizes = [int(tr.size) for tr, _te in splits]
    test_sizes = [int(te.size) for _tr, te in splits]
    return {
        "n_splits": int(len(splits)),
        "coverage_pct": float(coverage),
        "mean_train_size": float(np.mean(train_sizes)),
        "mean_test_size": float(np.mean(test_sizes)),
    }
