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
    "ML lab": "model_lab",
    "Model trainer": "model_trainer",
    "Model metrics": "model_metrics",
    "Walk-forward": "walk_forward_lab",
    "Regime lab": "regime_lab",
    "Portfolio lab": "portfolio_lab",
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
elif page == "ML lab":
    from aprilalgo.ui.pages.model_lab import render
    render()
elif page == "Model trainer":
    from aprilalgo.ui.pages.model_trainer import render
    render()
elif page == "Model metrics":
    from aprilalgo.ui.pages.model_metrics import render
    render()
elif page == "Walk-forward":
    from aprilalgo.ui.pages.walk_forward_lab import render
    render()
elif page == "Regime lab":
    from aprilalgo.ui.pages.regime_lab import render
    render()
elif page == "Portfolio lab":
    from aprilalgo.ui.pages.portfolio_lab import render
    render()
