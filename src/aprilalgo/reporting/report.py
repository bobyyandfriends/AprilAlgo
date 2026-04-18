"""HTML report: metrics, equity, importance, SHAP, regime, walk-forward (v0.4+)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{{ title }}</title></head>
<body>
  <h1 id="section-title">{{ title }}</h1>

  <section id="section-metrics">
    <h2>Metrics</h2>
    <table border="1">
    {% for k, v in metrics.items() %}
      <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
    </table>
  </section>

  {% if equity is not none %}
  <section id="section-equity">
    <h2>Equity</h2>
    <table border="1">
 <thead><tr>{% for c in equity.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in equity.head(500).iterrows() %}
        <tr>{% for c in equity.columns %}<td>{{ row[c] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if importance is not none %}
  <section id="section-importance">
    <h2>Feature importance</h2>
    <table border="1">
      <thead><tr>{% for c in importance.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in importance.head(50).iterrows() %}
        <tr>{% for c in importance.columns %}<td>{{ row[c] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if shap is not none %}
  <section id="section-shap">
    <h2>SHAP importance</h2>
    <table border="1">
      <thead><tr>{% for c in shap.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in shap.head(50).iterrows() %}
        <tr>{% for c in shap.columns %}<td>{{ row[c] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if regime is not none %}
  <section id="section-regime">
    <h2>Regime timeline</h2>
    <table border="1">
      <thead><tr>{% for c in regime.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in regime.head(500).iterrows() %}
        <tr>{% for c in regime.columns %}<td>{{ row[c] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if walk_forward is not none %}
  <section id="section-walk-forward">
    <h2>Walk-forward</h2>
    <table border="1">
      <thead><tr>{% for c in walk_forward.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in walk_forward.head(200).iterrows() %}
        <tr>{% for c in walk_forward.columns %}<td>{{ row[c] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if notes %}<p id="section-notes">{{ notes }}</p>{% endif %}
</body>
</html>
"""


def render_backtest_html(
    metrics: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "AprilAlgo backtest",
    notes: str = "",
    equity: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
    shap: pd.DataFrame | None = None,
    regime: pd.DataFrame | None = None,
    walk_forward: pd.DataFrame | None = None,
) -> Path:
    """Write a single-page HTML report with stable ``section-*`` ids for tests/UI."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    html = Template(_HTML).render(
        title=title,
        metrics=metrics,
        notes=notes,
        equity=equity,
        importance=importance,
        shap=shap,
        regime=regime,
        walk_forward=walk_forward,
    )
    p.write_text(html, encoding="utf-8")
    return p
