"""Streamlit runner: tuner page with symbols."""

import aprilalgo.ui.pages.tuner as tun
from aprilalgo.ui.pages.tuner import render

tun.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]

render()
