"""AprilAlgo — Streamlit UI entry point.

Run with:  uv run streamlit run src/aprilalgo/ui/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AprilAlgo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- sidebar navigation ---------------

PAGES = {
    "Charts": "charts",
    "Signal Feed": "signals",
    "Dashboard": "dashboard",
    "Parameter Tuner": "tuner",
}

st.sidebar.title("AprilAlgo")
st.sidebar.caption("Quantitative Trading Intelligence")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

# --------------- page routing ---------------

if page == "Charts":
    from aprilalgo.ui.pages.charts import render
    render()
elif page == "Signal Feed":
    from aprilalgo.ui.pages.signals import render
    render()
elif page == "Dashboard":
    from aprilalgo.ui.pages.dashboard import render
    render()
elif page == "Parameter Tuner":
    from aprilalgo.ui.pages.tuner import render
    render()
