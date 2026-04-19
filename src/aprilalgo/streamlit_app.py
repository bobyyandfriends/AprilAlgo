"""AprilAlgo — Streamlit UI entry point.

Run::

    uv run streamlit run src/aprilalgo/streamlit_app.py

This file lives **outside** ``aprilalgo/ui/`` so the Python package directory
``ui/pages/`` is not a sibling ``pages/`` folder next to the entry script.
Streamlit otherwise auto-discovers that folder as multipage routes, which adds
a second sidebar navigator that conflicts with :data:`aprilalgo.ui.app.PAGES`
and can import page modules as standalone scripts (circular imports / errors).

Page registry: :mod:`aprilalgo.ui.app`.
"""

from __future__ import annotations

import streamlit as st

from aprilalgo.ui.app import PAGES

st.set_page_config(
    page_title="AprilAlgo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("AprilAlgo")
st.sidebar.caption("Quantitative Trading Intelligence")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

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
