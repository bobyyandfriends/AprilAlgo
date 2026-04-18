"""AppTest: charts page (mocked loaders)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_charts_page_smoke() -> None:
    at = AppTest.from_file(str(_APPS / "charts_smoke.py"), default_timeout=180)
    at.run()
    assert not at.exception
    titles = [h.value for h in at.header] if at.header else []
    assert any("Charts" in (t or "") or "Price" in (t or "") for t in titles)
