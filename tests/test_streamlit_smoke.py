"""Streamlit: import smoke + render entry points (AppTest in tests/ui/)."""

from __future__ import annotations


def test_ui_app_importable():
    import aprilalgo.ui.app as app

    assert hasattr(app, "PAGES")


def test_streamlit_pages_importable():
    import importlib

    import aprilalgo.ui.pages.charts as charts_pkg

    assert callable(charts_pkg.render)

    for mod in (
        "aprilalgo.ui.pages.dashboard",
        "aprilalgo.ui.pages.model_lab",
        "aprilalgo.ui.pages.model_metrics",
        "aprilalgo.ui.pages.model_trainer",
        "aprilalgo.ui.pages.portfolio_lab",
        "aprilalgo.ui.pages.regime_lab",
        "aprilalgo.ui.pages.signals",
        "aprilalgo.ui.pages.tuner",
        "aprilalgo.ui.pages.walk_forward_lab",
    ):
        m = importlib.import_module(mod)
        assert callable(getattr(m, "render", None)), f"{mod} must define render()"
