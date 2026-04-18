"""Streamlit runner: dashboard with no symbols (for AppTest.from_file)."""

import aprilalgo.ui.pages.dashboard as dash
from aprilalgo.ui.pages.dashboard import render

dash.discover_symbols = lambda: {}  # type: ignore[method-assign]

render()
