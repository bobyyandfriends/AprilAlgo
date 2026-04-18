"""Feature importance: XGBoost gain/cover/weight + sklearn permutation (v0.3).

SHAP is intentionally out of scope here; use permutation as a cheap alternative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def xgb_importance_table(
    classifier: Any,
    *,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Tidy frame ``feature``, ``score``, ``rank`` from Booster."""
    booster = classifier.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    # scores keys are often f0, f1 — map via feature_names if needed
    rows: list[tuple[str, float]] = []
    if not scores:
        return pd.DataFrame(columns=["feature", "score", "rank"])

    for j, name in enumerate(feature_names):
        key_f = f"f{j}"
        key_plain = name
        v = scores.get(key_plain, scores.get(key_f, 0.0))
        rows.append((name, float(v)))

    df = pd.DataFrame(rows, columns=["feature", "score"])
    df.insert(1, "method", importance_type)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def permutation_importance_table(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Permutation importance (mean decrease in metric)."""
    r = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "method": "permutation",
            "score": r.importances_mean,
            "std": r.importances_std,
        }
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
