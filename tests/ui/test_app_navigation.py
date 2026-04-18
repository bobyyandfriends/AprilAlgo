"""Navigate every Streamlit page via AppTest (no exceptions; header or warning)."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

import aprilalgo.ui.app as app_module


@pytest.mark.parametrize("label", list(app_module.PAGES.keys()))
def test_app_navigation_each_page(label: str) -> None:
    at = AppTest.from_file("src/aprilalgo/ui/app.py", default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value(label)
    at.run()
    assert not at.exception
    # Some pages use st.title (AppTest.title) instead of st.header.
    assert (
        at.header or at.warning or at.title or at.subheader
    ), f"page {label!r} produced no header, title, subheader, or warning"
