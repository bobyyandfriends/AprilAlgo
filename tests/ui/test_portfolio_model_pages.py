"""AppTest: portfolio lab + ML pages (runner scripts)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.56")
from streamlit.testing.v1 import AppTest

_APPS = Path(__file__).resolve().parent / "apps"


def test_portfolio_lab_runs_backtest_on_fixture() -> None:
    at = AppTest.from_file(str(_APPS / "portfolio_lab_fixture.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for b in at.button:
        if b.label == "Run":
            b.click()
            break
    else:
        pytest.fail("Run button not found")
    at.run()
    assert not at.exception
    assert at.dataframe


def test_model_lab_evaluate_mocked_cli() -> None:
    at = AppTest.from_file(str(_APPS / "model_lab_mocked.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for b in at.button:
        if "Evaluate" in (b.label or ""):
            b.click()
            break
    else:
        pytest.fail("Evaluate button not found")
    at.run()
    assert not at.exception
    assert at.code
    assert "cli_ok" in at.code[0].value


def test_model_trainer_importance_mocked_cli() -> None:
    at = AppTest.from_file(str(_APPS / "model_trainer_mocked.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for b in at.button:
        if "importance" in (b.label or "").lower():
            b.click()
            break
    else:
        pytest.fail("importance button not found")
    at.run()
    assert not at.exception
    assert at.code
    assert "trained" in at.code[0].value


def test_model_metrics_purged_cv_mocked() -> None:
    at = AppTest.from_file(str(_APPS / "model_metrics_mocked.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for b in at.button:
        if "Run purged CV" in (b.label or ""):
            b.click()
            break
    else:
        pytest.fail("Run purged CV button not found")
    at.run()
    assert not at.exception
    assert at.json
    assert at.dataframe
