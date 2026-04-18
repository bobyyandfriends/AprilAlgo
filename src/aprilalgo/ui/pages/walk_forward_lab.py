"""Streamlit: walk-forward split + summary preview (CLI ``walk-forward``)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def render() -> None:
    st.title("Walk-forward")
    st.caption("Shows split summary, per-fold ranges, and export from CLI JSON output.")

    cfg = st.text_input(
        "Config path (relative to project root)",
        value="configs/ml/default.yaml",
    )
    if st.button("Show splits"):
        cmd = [sys.executable, "-m", "aprilalgo.cli", "walk-forward", "--config", cfg]
        proc = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
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
        except Exception:
            pass
