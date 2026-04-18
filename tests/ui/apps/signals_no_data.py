"""Streamlit runner: signals page with no data."""

import aprilalgo.ui.pages.signals as sig
from aprilalgo.ui.pages.signals import render

sig.discover_symbols = lambda: {}  # type: ignore[method-assign]

render()
