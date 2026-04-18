"""Streamlit: walk-forward splits + WF tuner results (CLI ``walk-forward`` / ``wf-tune``)."""

from __future__ import annotations

import json
import subprocess  # nosec B404  # trusted local operator-driven CLI from Streamlit UI
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Walk-forward listing itself is fast, but the subprocess still needs a hard
# cap so a hung CLI invocation cannot block the Streamlit worker forever.
_CLI_TIMEOUT_SECONDS = 60 * 60


def render() -> None:
    st.title("Walk-forward")
    st.caption("Split preview from CLI JSON; tuner charts from wf_tune_results.csv (wf-tune).")

    cfg = st.text_input(
        "Config path (relative to project root)",
        value="configs/ml/default.yaml",
    )
    out_rel = st.text_input(
        "Model out_dir (for auto-loading wf_tune_results.csv)",
        value="models/xgboost/latest",
    )

    tab_splits, tab_tuner = st.tabs(["Splits", "Tuner"])

    with tab_splits:
        st.subheader("Walk-forward index ranges")
        if st.button("Show splits"):
            cmd = [sys.executable, "-m", "aprilalgo.cli", "walk-forward", "--config", cfg]
            try:
                proc = subprocess.run(  # nosec B603  # trusted local operator-driven CLI from Streamlit UI
                    cmd,
                    cwd=_PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_CLI_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as e:
                st.error(
                    f"walk-forward CLI timed out after {_CLI_TIMEOUT_SECONDS}s.\nPartial stdout:\n{e.stdout or ''}"
                )
                return
            st.code(proc.stdout or proc.stderr or "(empty)", language="json")
            try:
                data = json.loads(proc.stdout)
                st.subheader("Summary")
                st.json(data.get("summary", {}))
                folds = pd.DataFrame(data.get("splits", []))
                if not folds.empty:
                    st.subheader("Folds")
                    st.dataframe(folds, use_container_width=True)
                    st.line_chart(folds[["fold", "test_return"]].set_index("fold"))
                    st.download_button(
                        "Download folds CSV",
                        data=folds.to_csv(index=False),
                        file_name="walk_forward_folds.csv",
                    )
                st.json({"n_bars": data.get("n_bars")})
            except json.JSONDecodeError:
                # Non-JSON stdout is expected when the CLI prints a plain error
                # (e.g. missing config file); keep the stdout/stderr already
                # rendered above and move on.
                pass
            except (KeyError, ValueError) as err:
                # Schema drift (missing expected keys) or pandas-side validation
                # error — surface it so the user can tell the JSON parsed but
                # didn't match the expected shape.
                st.warning(f"Could not render walk-forward output: {err}")

    with tab_tuner:
        st.subheader("Walk-forward ML tuner")
        st.caption(
            "Per-fold scores by grid_id (from wf_tune_results.csv). "
            "Run: uv run python -m aprilalgo.cli wf-tune --config <yaml>"
        )
        uploaded = st.file_uploader(
            "Upload wf_tune_results.csv (optional)",
            type=["csv"],
        )
        discover = _PROJECT_ROOT / out_rel / "wf_tune_results.csv"
        df: pd.DataFrame | None = None
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.caption("Using uploaded CSV.")
        elif discover.is_file():
            df = pd.read_csv(discover)
            st.caption(f"Loaded `{discover.relative_to(_PROJECT_ROOT)}`")
        else:
            st.info("No `wf_tune_results.csv` under the model path above. Run **wf-tune** or upload a CSV.")

        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
            if "grid_id" in df.columns and "score" in df.columns:
                chart = st.radio("Chart", ("Box", "Violin"), horizontal=True)
                plot_df = df.copy()
                plot_df["grid_id"] = plot_df["grid_id"].astype(str)
                if chart == "Box":
                    fig = px.box(
                        plot_df,
                        x="grid_id",
                        y="score",
                        points="all",
                        title="Score by grid_id (per walk-forward fold)",
                    )
                else:
                    fig = px.violin(
                        plot_df,
                        x="grid_id",
                        y="score",
                        box=True,
                        points="all",
                        title="Score by grid_id (per walk-forward fold)",
                    )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("CSV needs at least `grid_id` and `score` columns for the chart.")
