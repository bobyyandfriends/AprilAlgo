"""Streamlit runner: dashboard idle state with fixture symbol list."""

import aprilalgo.ui.pages.dashboard as dash
from aprilalgo.ui.pages.dashboard import render

dash.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]

render()
