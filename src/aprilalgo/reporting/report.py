"""HTML report: metrics, equity, importance, SHAP, regime, walk-forward (v0.4+)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template, select_autoescape

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
    {% if equity|length > 500 %}
    <p class="truncation-note" style="font-size:0.9em;color:#666;">
      Showing first 500 of {{ equity|length }} equity rows (truncated for report size).
    </p>
    {% else %}
    <p class="truncation-note" style="font-size:0.9em;color:#666;">
      Showing all {{ equity|length }} equity rows.
    </p>
    {% endif %}
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
  <section id="section-regime-timeline">
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


def render_sampling_section(
    sample_weight: pd.Series | None,
    *,
    bins: int = 12,
) -> str:
    """Emit ``section-sampling`` with a simple histogram of non-uniform row weights (e.g. uniqueness)."""
    if sample_weight is None or len(sample_weight) == 0:
        return ""
    s = pd.to_numeric(sample_weight, errors="coerce").dropna()
    if s.empty:
        return ""
    counts, edges = np.histogram(s.to_numpy(dtype=np.float64), bins=bins)
    hist_df = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts.astype(int)}
    )
    inner = hist_df.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}")
    return (
        '<section id="section-sampling"><h2>Sampling weights</h2>'
        f"<p>Rows: {len(s)}</p>{inner}</section>"
    )


def render_meta_section(
    coefficients: pd.DataFrame | None,
    *,
    oof_coverage: float | None = None,
    oof_rows: int | None = None,
    oof_nonnull: int | None = None,
) -> str:
    """Emit ``section-meta`` with meta-label coefficients and optional OOF coverage stats."""
    inner: list[str] = []
    if coefficients is not None and not coefficients.empty:
        inner.append(coefficients.to_html(index=False, escape=True))
    if oof_coverage is not None:
        inner.append(
            f"<p>OOF prediction coverage: <strong>{html.escape(f'{float(oof_coverage):.4f}')}</strong></p>"
        )
    if oof_rows is not None:
        inner.append(f"<p>OOF table rows: <strong>{int(oof_rows)}</strong></p>")
    if oof_nonnull is not None:
        inner.append(f"<p>Rows with non-null OOF pred: <strong>{int(oof_nonnull)}</strong></p>")
    if not inner:
        return ""
    return (
        '<section id="section-meta"><h2>Meta-label &amp; OOF</h2>'
        + "".join(inner)
        + "</section>"
    )


def render_regime_section(
    bucket_counts: pd.DataFrame | None,
    per_regime_accuracy: pd.DataFrame | None,
) -> str:
    """Emit ``section-regime`` with bucket row counts and optional per-bucket accuracy."""
    if (bucket_counts is None or bucket_counts.empty) and (
        per_regime_accuracy is None or per_regime_accuracy.empty
    ):
        return ""
    parts: list[str] = ['<section id="section-regime"><h2>Regime buckets</h2>']
    if bucket_counts is not None and not bucket_counts.empty:
        parts.append("<h3>Counts</h3>")
        parts.append(bucket_counts.to_html(index=False, escape=True))
    if per_regime_accuracy is not None and not per_regime_accuracy.empty:
        parts.append("<h3>Per-regime accuracy</h3>")
        parts.append(per_regime_accuracy.to_html(index=False, escape=True))
    parts.append("</section>")
    return "".join(parts)


def render_wf_tuner_section(
    wf_tune_results: pd.DataFrame | None,
    *,
    top_n: int = 5,
) -> str:
    """Emit ``section-wf-tuner`` with aggregated top grid points and raw per-fold scores."""
    if wf_tune_results is None or wf_tune_results.empty:
        return ""
    from aprilalgo.tuner.ml_walk_forward import aggregate_grid

    agg = aggregate_grid(wf_tune_results, "score").sort_values("mean", ascending=False).head(
        int(top_n)
    )
    parts: list[str] = [
        '<section id="section-wf-tuner"><h2>Walk-forward tuner</h2>',
        f"<h3>Top {int(top_n)} by mean score</h3>",
        agg.to_html(index=False, escape=True),
        "<h3>Per-fold scores (sample)</h3>",
        wf_tune_results.head(500).to_html(index=False, escape=True),
        "</section>",
    ]
    return "".join(parts)


def _metrics_section_html(metrics: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in metrics.items()
    )
    return (
        '<section id="section-metrics"><h2>Metrics</h2>'
        f'<table border="1">{rows}</table></section>'
    )


def render_full_ml_report_html(
    metrics: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "AprilAlgo ML report",
    sampling_weights: pd.Series | None = None,
    meta_coef: pd.DataFrame | None = None,
    oof_coverage: float | None = None,
    oof_rows: int | None = None,
    oof_nonnull: int | None = None,
    regime_bucket_counts: pd.DataFrame | None = None,
    regime_accuracy: pd.DataFrame | None = None,
    wf_tuner_results: pd.DataFrame | None = None,
    notes: str = "",
) -> Path:
    """Single-page ML report when optional Sprint 1–9 artifacts are available."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        _metrics_section_html(metrics),
        render_sampling_section(sampling_weights),
        render_meta_section(
            meta_coef,
            oof_coverage=oof_coverage,
            oof_rows=oof_rows,
            oof_nonnull=oof_nonnull,
        ),
        render_regime_section(regime_bucket_counts, regime_accuracy),
        render_wf_tuner_section(wf_tuner_results, top_n=5),
    ]
    if notes:
        body.append(f'<p id="section-notes">{html.escape(notes)}</p>')
    body.append("</body></html>")
    p.write_text("\n".join(body), encoding="utf-8")
    return p


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
    # Autoescape guards against stray HTML characters in metric / column names
    # (feature names, notes) corrupting the rendered document. ``rendered`` is
    # named explicitly to avoid shadowing the stdlib ``html`` module imported
    # at the top, which is used by the sibling ``render_full_ml_report_html``.
    template = Template(_HTML, autoescape=select_autoescape(["html", "xml"]))
    rendered = template.render(
        title=title,
        metrics=metrics,
        notes=notes,
        equity=equity,
        importance=importance,
        shap=shap,
        regime=regime,
        walk_forward=walk_forward,
    )
    p.write_text(rendered, encoding="utf-8")
    return p
