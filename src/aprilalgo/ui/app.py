"""AprilAlgo Streamlit page registry (no Streamlit calls at import time).

The Streamlit **entry script** is :mod:`aprilalgo.streamlit_app` — run::

    uv run streamlit run src/aprilalgo/streamlit_app.py

Keeping this module free of ``streamlit`` imports avoids side effects when
tests do ``import aprilalgo.ui.app`` and prevents accidentally using
``src/aprilalgo/ui/app.py`` as the Streamlit entrypoint (which would sit next to
the ``pages/`` package and trigger Streamlit's automatic multipage discovery).
"""

from __future__ import annotations

PAGES: dict[str, str] = {
    "Charts": "charts",
    "Signal Feed": "signals",
    "Dashboard": "dashboard",
    "Parameter Tuner": "tuner",
    "ML lab": "model_lab",
    "Model trainer": "model_trainer",
    "Model metrics": "model_metrics",
    "Walk-forward": "walk_forward_lab",
    "Regime lab": "regime_lab",
    "Portfolio lab": "portfolio_lab",
}
