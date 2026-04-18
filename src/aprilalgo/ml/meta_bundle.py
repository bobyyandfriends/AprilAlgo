"""Serialized meta-label logistic (v0.5 Sprint 4).

Persists a :class:`~sklearn.linear_model.LogisticRegression` fit from
:func:`~aprilalgo.labels.meta_label.fit_meta_logit_purged` for inference without sklearn
pickles (JSON + NumPy math only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

__all__ = ["MetaLogitBundle", "load_meta_logit_bundle", "save_meta_logit_bundle"]


@dataclass(frozen=True, slots=True)
class MetaLogitBundle:
    """Loaded meta logistic for ``predict_proba`` on the stacked feature matrix."""

    feature_names: tuple[str, ...]
    coef: np.ndarray  # (n_classes, n_features) sklearn layout
    intercept: np.ndarray  # (n_classes,)
    classes_: np.ndarray

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape ``(n_samples, n_classes)`` like sklearn."""
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError("X is missing columns required by the meta logit bundle: " + ", ".join(sorted(missing)))
        Xd = X[list(self.feature_names)].to_numpy(dtype=np.float64, copy=False)
        logits = Xd @ self.coef.T + self.intercept.reshape(1, -1)
        if logits.shape[1] == 1:
            p1 = _sigmoid(logits.ravel())
            return np.column_stack([1.0 - p1, p1])
        return _softmax(logits)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(np.clip(z, -60.0, 60.0))
    return ex / np.sum(ex, axis=1, keepdims=True)


def save_meta_logit_bundle(
    out_dir: str | Path,
    clf: LogisticRegression,
    *,
    feature_names: list[str],
) -> Path:
    """Write ``meta_logit.json`` under *out_dir*."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64).ravel()
    classes_list = [float(c) for c in np.asarray(clf.classes_).ravel()]
    payload: dict[str, Any] = {
        "feature_names": list(feature_names),
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "classes_": classes_list,
    }
    path = root / "meta_logit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_meta_logit_bundle(out_dir: str | Path, *, rel_path: str = "meta_logit.json") -> MetaLogitBundle:
    """Load :class:`MetaLogitBundle` from ``<out_dir>/<rel_path>`` (default ``meta_logit.json``)."""
    root = Path(out_dir)
    path = root / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing meta_logit bundle: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    coef = np.asarray(data["coef"], dtype=np.float64)
    intercept = np.asarray(data["intercept"], dtype=np.float64).ravel()
    classes_ = np.asarray(data["classes_"], dtype=np.float64)
    names = tuple(str(n) for n in data["feature_names"])
    return MetaLogitBundle(
        feature_names=names,
        coef=coef,
        intercept=intercept,
        classes_=classes_,
    )
