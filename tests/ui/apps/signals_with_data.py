"""Streamlit runner: signals page with AAPL daily."""

import aprilalgo.ui.pages.signals as sig
from aprilalgo.ui.pages.signals import render

sig.discover_symbols = lambda: {"daily": ["AAPL"]}  # type: ignore[method-assign]

render()
