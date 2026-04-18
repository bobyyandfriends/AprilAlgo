"""Streamlit: run ML CLI flows (train / evaluate / importance) from the UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Avoid pinning the Streamlit worker thread to a hung child CLI process. One
# hour is generous enough for purged-CV evaluation on daily bars while still
# guaranteeing eventual release of the UI if something deadlocks.
_CLI_TIMEOUT_SECONDS = 60 * 60


def render() -> None:
    st.title("ML lab")
    st.caption("Wraps `python -m aprilalgo.cli` — config-first, same as terminal.")

    cfg = st.text_input(
        "Config path (relative to project root)",
        value="configs/ml/default.yaml",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Train"):
            _run(["train", "--config", cfg])
    with col2:
        if st.button("Evaluate (purged CV)"):
            _run(["evaluate", "--config", cfg])
    with col3:
        if st.button("Feature importance"):
            _run(["importance", "--config", cfg])


def _run(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "aprilalgo.cli", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=_CLI_TIMEOUT_SECONDS,
        )
        st.code(proc.stdout or "(no stdout)", language="text")
        if proc.stderr:
            st.code(proc.stderr, language="text")
        st.caption(f"exit {proc.returncode}")
    except subprocess.TimeoutExpired as e:
        st.error(
            f"CLI timed out after {_CLI_TIMEOUT_SECONDS}s; check for hangs or "
            f"increase the UI timeout. Partial stdout:\n{e.stdout or ''}"
        )
    except (OSError, ValueError) as e:
        st.error(str(e))
