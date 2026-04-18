"""Charts page package.

Public surface: ``render`` (called by the Streamlit router in ``ui/app.py``).

Internals are split across submodules:

* :mod:`page` — top-level Streamlit layout (sidebar, tabs, orchestration)
* :mod:`figure` — :class:`plotly.graph_objects.Figure` builder (subplots, layout, theme)
* :mod:`layers.overlays` — indicator traces drawn on the candlestick row
* :mod:`layers.panels` — indicator traces drawn in sub-panels
* :mod:`layers.demark_counts` — TD Sequential integer annotations
* :mod:`layers.ml_proba` — XGBoost probability area chart + price band
* :mod:`layers.shap_local` — signed stacked SHAP bars + right-column local / global
* :mod:`data.ml_artifacts` — cached loaders for OOF probabilities & SHAP matrices
"""

from aprilalgo.ui.pages.charts.page import render

__all__ = ["render"]
