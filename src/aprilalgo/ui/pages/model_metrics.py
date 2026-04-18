"""Streamlit: purged CV metrics, confusion-style summary, ROC proxy (in-process)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

from aprilalgo.ml.evaluator import purged_cv_evaluate
from aprilalgo.ml.pipeline import prepare_xy as _prepare_xy

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def render() -> None:
    st.title("Model metrics")
    st.caption("Purged CV in-process (same pipeline as `cli evaluate`).")

    cfg_path = st.text_input(
        "Config path (relative to project root)",
        value="configs/ml/default.yaml",
    )
    if st.button("Run purged CV"):
        p = _PROJECT_ROOT / cfg_path
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        X, y, t0, t1, task = _prepare_xy(cfg)
        cv = cfg.get("cv", {})
        n_splits = int(cv.get("n_splits", 3))
        embargo = int(cv.get("embargo", 0))
        seed = int(cfg.get("random_state", 42))
        xgb_params = cfg.get("model", {}).get("xgb", {})

        def factory() -> XGBClassifier:
            if task == "binary":
                return XGBClassifier(
                    objective="binary:logistic",
                    random_state=seed,
                    n_estimators=int(xgb_params.get("n_estimators", 50)),
                    max_depth=int(xgb_params.get("max_depth", 3)),
                    learning_rate=float(xgb_params.get("learning_rate", 0.1)),
                )
            return XGBClassifier(
                objective="multi:softprob",
                random_state=seed,
                n_estimators=int(xgb_params.get("n_estimators", 50)),
                max_depth=int(xgb_params.get("max_depth", 3)),
                learning_rate=float(xgb_params.get("learning_rate", 0.1)),
            )

        with st.spinner("Evaluating…"):
            res = purged_cv_evaluate(
                factory,
                X,
                y,
                sample_t0=t0,
                sample_t1=t1,
                n_splits=n_splits,
                embargo=embargo,
            )
        st.subheader("Mean metrics")
        st.json(res.get("mean", {}))
        if "f1" in res.get("mean", {}):
            st.metric("Mean F1", f"{float(res['mean']['f1']):.3f}")
        if "f1_macro" in res.get("mean", {}):
            st.metric("Mean F1 macro", f"{float(res['mean']['f1_macro']):.3f}")
        fd = res.get("folds_df")
        if isinstance(fd, pd.DataFrame) and not fd.empty:
            st.subheader("Per-fold")
            st.dataframe(fd, use_container_width=True)

        # Confusion from last fold (illustrative)
        folds = res.get("folds", [])
        if folds:
            last_cm = folds[-1].get("confusion_matrix")
            if last_cm is not None:
                cm = np.asarray(last_cm, dtype=float)
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    title="Confusion matrix (last CV fold)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Binary ROC curve + proxy equity curve. Both are IN-SAMPLE (the model is
        # fit and scored on the same rows). They are visual quick-looks only and
        # must not be treated as out-of-sample performance — CV metrics above are
        # the ground truth for that.
        if task == "binary":
            st.caption(
                ":warning: ROC + proxy equity below are in-sample on the training "
                "rows; refer to the purged-CV metrics above for held-out performance."
            )
            clf = factory()
            clf.fit(X, y)
            prob = clf.predict_proba(X)[:, 1]
            fpr, tpr, _thr = roc_curve(y, prob)
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            st.subheader("ROC (in-sample quick view)")
            st.line_chart(roc_df.set_index("fpr"))

            pred = clf.predict(X)
            pnl = np.where(pred == y.to_numpy(), 1.0, -1.0)
            eq = pd.Series(pnl).cumsum()
            st.subheader("Proxy equity curve (in-sample classification correctness)")
            st.line_chart(eq)

        # Optional feature importance preview from latest CLI outputs. Guard
        # against unexpected CSV schemas so a bad file doesn't blow up the page.
        out_dir = _PROJECT_ROOT / cfg.get("model", {}).get("out_dir", "outputs/ml")
        gain_csv = out_dir / "importance_gain.csv"
        if gain_csv.is_file():
            try:
                imp = pd.read_csv(gain_csv).head(20)
                if {"feature", "score"}.issubset(imp.columns):
                    st.subheader("Feature importance (latest gain CSV)")
                    st.bar_chart(imp.set_index("feature")["score"])
                else:
                    st.warning(
                        f"importance_gain.csv exists but is missing required columns (found: {list(imp.columns)})"
                    )
            except Exception as exc:  # pragma: no cover - UI best-effort
                st.warning(f"Could not read importance_gain.csv: {exc}")

        st.download_button(
            "Download JSON",
            data=json.dumps(
                {
                    "mean": res.get("mean"),
                    "folds_df": fd.to_dict(orient="records") if isinstance(fd, pd.DataFrame) else [],
                },
                indent=2,
                default=str,
            ),
            file_name="cv_metrics.json",
        )
