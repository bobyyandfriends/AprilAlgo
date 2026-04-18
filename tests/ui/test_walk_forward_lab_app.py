"""AppTest: walk-forward lab with subprocess mocked via runner script."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_walk_forward_splits_success() -> None:
    at = AppTest.from_file(str(_APPS / "walk_forward_mocked.py"), default_timeout=60)
    at.run()
    for b in at.button:
        if b.label == "Show splits":
            b.click()
            break
    else:
        pytest.fail("Show splits button not found")
    at.run()
    assert not at.exception
    assert at.dataframe


def test_walk_forward_splits_cli_error() -> None:
    at = AppTest.from_file(str(_APPS / "walk_forward_err.py"), default_timeout=60)
    at.run()
    for b in at.button:
        if b.label == "Show splits":
            b.click()
            break
    at.run()
    assert not at.exception
