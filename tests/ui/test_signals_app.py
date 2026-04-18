"""AppTest: signal feed page."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_signals_warning_without_data() -> None:
    at = AppTest.from_file(str(_APPS / "signals_no_data.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.warning


def test_signals_renders_tail() -> None:
    at = AppTest.from_file(str(_APPS / "signals_with_data.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.dataframe or at.markdown


def test_signals_min_conf_filter_empty() -> None:
    at = AppTest.from_file(str(_APPS / "signals_with_data.py"), default_timeout=120)
    at.run()
    for w in at.sidebar.slider:
        if w.key == "sig_min":
            w.set_value(1.0)
            break
    else:
        pytest.skip("sig_min slider not found in this Streamlit version")
    at.run()
    assert not at.exception
    assert at.info
    assert "No signals meet" in at.info[0].value
