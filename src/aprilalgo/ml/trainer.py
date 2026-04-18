"""XGBoost training and model bundle save/load (v0.3).

Writes ``meta.json`` and ``xgboost.json`` under the output directory.
Inference uses :class:`xgboost.Booster` (sklearn wrapper does not reload ``classes_``).
v0.4: extend ``meta.json`` only; keep these filenames stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

Task = Literal["binary", "multiclass"]

_APRIL_LABEL_CLASSES = "_aprilalgo_label_classes_"


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Loaded Booster + metadata for inference and strategies."""

    booster: xgb.Booster
    feature_names: list[str]
    task: Task
    classes_: np.ndarray
    indicator_config: list[dict[str, Any]] | None
    meta: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape ``(n_samples, n_classes)`` probability matrix.

        Binary columns are anchored on :attr:`classes_`: column ``i`` always
        corresponds to ``classes_[i]``, even if a future ``meta.json`` persists
        classes as ``[1.0, 0.0]`` instead of the conventional ``[0.0, 1.0]``.
        """
        data = X[self.feature_names].to_numpy(dtype=np.float64, copy=False)
        dm = xgb.DMatrix(data, feature_names=self.feature_names)
        pred = np.asarray(self.booster.predict(dm))
        if self.task == "binary":
            if pred.ndim == 2 and pred.shape[1] == 2:
                return pred
            p = pred.ravel()
            classes = np.asarray(self.classes_, dtype=np.float64).ravel()
            if classes.shape[0] < 2:
                # Degenerate bundle (e.g. a per-regime slice where training
                # data saw only one class). Preserve legacy behaviour: treat
                # ``p`` as P(positive=1) and emit a canonical ``[P(0), P(1)]``
                # matrix so downstream callers keep working.
                out = np.empty((p.shape[0], 2), dtype=np.float64)
                out[:, 0] = 1.0 - p
                out[:, 1] = p
                return out
            if classes.shape[0] > 2:
                raise ValueError(f"Binary bundle must expose <=2 classes, got {classes.tolist()}")
            # Booster ``predict`` returns P(class=positive_label). Identify which
            # column in ``classes_`` is the positive (larger) label and place the
            # predicted probability there so column order always matches classes_.
            pos_ix = int(np.argmax(classes))
            neg_ix = 1 - pos_ix
            out = np.empty((p.shape[0], 2), dtype=np.float64)
            out[:, pos_ix] = p
            out[:, neg_ix] = 1.0 - p
            return out
        n = len(X)
        k = len(self.classes_)
        if pred.ndim == 1:
            return pred.reshape(n, k)
        return pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_, dtype=np.float64)[idx]

    def predict_proba_row(self, X_row: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X_row)[0]


def _default_xgb_params(task: Task) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    }
    return base


def train_xgb_classifier(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    task: Task,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> xgb.XGBClassifier:
    """Fit :class:`xgboost.XGBClassifier` on aligned *X*, *y*.

    Multiclass *y* may be ``{-1, 0, 1}`` etc.; labels are encoded with
    :class:`~sklearn.preprocessing.LabelEncoder` before fitting. Binary *y*
    should be ``{0, 1}``.
    """
    params = {**_default_xgb_params(task), **(xgb_params or {})}
    params["random_state"] = random_state
    y_raw = np.asarray(y)
    if task == "binary":
        params.setdefault("objective", "binary:logistic")
        clf = xgb.XGBClassifier(**params)
        clf.fit(X, y_raw, sample_weight=sample_weight)
        return clf

    params.setdefault("objective", "multi:softprob")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y_enc, sample_weight=sample_weight)
    setattr(clf, _APRIL_LABEL_CLASSES, le.classes_.astype(np.float64).copy())
    return clf


def save_model_bundle(
    out_dir: str | Path,
    clf: xgb.XGBClassifier,
    *,
    feature_names: list[str],
    task: Task,
    indicator_config: list[dict[str, Any]] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist booster + ``meta.json`` under *out_dir*."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "xgboost.json"
    clf.save_model(model_path)

    if task == "multiclass":
        enc = getattr(clf, _APRIL_LABEL_CLASSES, None)
        if enc is None:
            raw_classes = np.asarray(clf.classes_, dtype=np.float64).ravel()
            # When :func:`train_xgb_classifier` is skipped and the sklearn wrapper
            # was fit directly, ``clf.classes_`` for ``multi:softprob`` holds the
            # encoded label space ``[0, 1, ..., k-1]`` rather than the original
            # AprilAlgo labels ``{-1, 0, 1}``. Detect that exact shape and refuse
            # to persist the bundle — the caller must set ``_aprilalgo_label_classes_``
            # or use :func:`train_xgb_classifier` so inference returns the correct
            # class axis.
            k = raw_classes.shape[0]
            looks_like_encoded = k >= 2 and np.array_equal(raw_classes, np.arange(k, dtype=np.float64))
            if looks_like_encoded:
                raise ValueError(
                    "save_model_bundle multiclass fallback refused: clf.classes_ "
                    f"== {raw_classes.tolist()} which looks like an encoded index "
                    "space rather than the original labels (e.g. {-1, 0, 1}). "
                    "Train via train_xgb_classifier() or set the "
                    f"{_APRIL_LABEL_CLASSES!r} attribute on clf with the decoded "
                    "labels before saving."
                )
            classes_list = [float(c) for c in raw_classes]
        else:
            classes_list = [float(c) for c in np.asarray(enc).ravel()]
    else:
        classes_list = [float(c) for c in np.asarray(clf.classes_).ravel()]

    meta: dict[str, Any] = {
        "feature_names": feature_names,
        "task": task,
        "classes_": classes_list,
        "indicator_config": indicator_config,
    }
    if extra_meta:
        meta.update(extra_meta)
    (root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return root


def load_model_bundle(model_dir: str | Path) -> ModelBundle:
    """Load :class:`ModelBundle` from *model_dir*."""
    root = Path(model_dir)
    meta_path = root / "meta.json"
    model_path = root / "xgboost.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing xgboost.json: {model_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    task = meta["task"]
    if task not in ("binary", "multiclass"):
        raise ValueError(f"Unknown task in meta: {task}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    classes = np.asarray(meta["classes_"], dtype=np.float64)
    feature_names = list(meta["feature_names"])
    ind_cfg = meta.get("indicator_config")
    return ModelBundle(
        booster=booster,
        feature_names=feature_names,
        task=task,
        classes_=classes,
        indicator_config=ind_cfg,
        meta=meta,
    )


def proba_positive_takeprofit(
    bundle: ModelBundle,
    proba: np.ndarray,
) -> float:
    """Probability mass on class ``1`` (take-profit), for sizing / thresholds."""
    classes = [float(c) for c in np.asarray(bundle.classes_).ravel()]
    if bundle.task == "binary":
        if len(classes) == 2:
            pos_ix = int(np.argmax(classes))
            return float(proba[pos_ix])
        if len(classes) == 1:
            # Degenerate fit (only one label seen in training).
            return 1.0 if float(classes[0]) >= 0.5 else 0.0
        raise ValueError("Binary model expects 1 or 2 classes")
    ix_tp = _index_takeprofit_class(classes)
    return float(proba[ix_tp])


def _index_takeprofit_class(classes: list[float]) -> int:
    for i, c in enumerate(classes):
        if abs(float(c) - 1.0) < 1e-9:
            return i
    raise ValueError("Multiclass bundle must include class 1 (take-profit) in classes_")
