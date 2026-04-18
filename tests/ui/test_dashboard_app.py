"""AppTest: dashboard page (via runner scripts — from_function omits imports on some Streamlit builds)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_dashboard_no_data() -> None:
    at = AppTest.from_file(str(_APPS / "dash_no_data.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.warning
    assert at.warning[0].value == "No data found."


def test_dashboard_idle_shows_info() -> None:
    at = AppTest.from_file(str(_APPS / "dash_idle.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.info
    assert "Run Backtest" in at.info[0].value


def test_dashboard_run_button_exists() -> None:
    at = AppTest.from_file(str(_APPS / "dash_idle.py"), default_timeout=60)
    at.run()
    labels = [b.label for b in at.sidebar.button]
    assert any("Run Backtest" in (lb or "") for lb in labels)


def test_dashboard_run_backtest_renders_summary() -> None:
    at = AppTest.from_file(str(_APPS / "dash_run_success.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for b in at.sidebar.button:
        if b.label and "Run Backtest" in b.label:
            b.click()
            break
    else:
        pytest.fail("Run Backtest button not found")
    at.run()
    assert not at.exception
    subs = [s.value for s in at.subheader] if at.subheader else []
    assert any("Performance" in (s or "") for s in subs)
