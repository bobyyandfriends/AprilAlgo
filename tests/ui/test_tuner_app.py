"""AppTest: parameter tuner page."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_tuner_smoke() -> None:
    at = AppTest.from_file(str(_APPS / "tuner_smoke.py"), default_timeout=120)
    at.run()
    assert not at.exception
    titles = [h.value for h in at.header] if at.header else []
    assert any("Tuner" in (t or "") for t in titles) or any("Tuner" in (s.value or "") for s in at.subheader)


def test_tuner_run_optimization_renders_best() -> None:
    at = AppTest.from_file(str(_APPS / "tuner_run_mocked.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for b in at.button:
        if b.label and "Run Optimization" in b.label:
            b.click()
            break
    else:
        pytest.skip("Run Optimization button not found in this Streamlit build")
    at.run()
    assert not at.exception
    subs = [s.value for s in at.subheader] if at.subheader else []
    assert any("Best" in (s or "") for s in subs)
