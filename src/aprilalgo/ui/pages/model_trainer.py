"""Streamlit: model training (CLI ``train`` / ``importance`` / ``shap``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def render() -> None:
    st.title("Model trainer")
    st.caption("Runs `python -m aprilalgo.cli train`, `importance`, and `shap`.")

    cfg = st.text_input(
        "Config path (relative to project root)",
        value="configs/ml/default.yaml",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Train"):
            _run(["train", "--config", cfg])
    with col2:
        if st.button("Feature importance"):
            _run(["importance", "--config", cfg])
    with col3:
        if st.button("SHAP"):
            _run(["shap", "--config", cfg])


def _run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "aprilalgo.cli", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        st.code(proc.stdout or "(no stdout)", language="text")
        if proc.stderr:
            st.code(proc.stderr, language="text")
        st.caption(f"exit {proc.returncode}")
    except Exception as e:
        st.error(str(e))
