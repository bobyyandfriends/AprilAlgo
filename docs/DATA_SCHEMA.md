# DATA_SCHEMA.md — Interface Map (AprilAlgo)

This document defines **data states** and **column contracts** between layers. Use it to implement or call modules without spelunking implementation files. Column names and types here are authoritative for integration and reviews (including look-ahead and leakage checks).

---

## 1. Raw OHLCV (Input)

**Role:** Canonical market bars loaded from CSV/Parquet or APIs, before indicators.

| Column      | Type   | Notes |
|------------|--------|--------|
| `timestamp` | string (ISO 8601) | Bar open time; timezone-aware preferred; sorted ascending per series |
| `open`     | float  | Open price |
| `high`     | float  | High price |
| `low`      | float  | Low price |
| `close`    | float  | Close price |
| `volume`   | float/int | Volume |

**Invariants:** One row per bar; no duplicate `(symbol, timestamp)` keys in a single series.

---

## 2. Enriched Features (Intermediate)

**Role:** Indicator outputs appended to OHLCV. Feeds confluence, strategies, and (v0.3) ML feature matrices.

**Column naming:** Parameterized bull/bear (and auxiliary) columns:

- Pattern: `{indicator_name}_{param_token}_bull` / `{indicator_name}_{param_token}_bear`
- Examples: `rsi_14_bull`, `rsi_14_bear`, `demark_9_bull`, `sma_20_bear`
- Additional non-signal columns may exist (e.g., `rsi_14`, band values) per indicator; they still carry parameters in the name where applicable.

**Types:** Typically `float64` for signals (often 0.0–1.0 or signed strength); align with `descriptor.py` / this doc when wiring new code.

**Temporal rule:** Values at bar *t* must depend only on data available at or before *t* (see project no-look-ahead rules).

---

## 3. Confluence Output (Scoring)

**Role:** Single scalar summary of multi-timeframe / multi-indicator agreement.

| Column             | Type  | Range / semantics |
|--------------------|-------|-------------------|
| `confluence_score` | float | **−1.0** (pure bearish) … **+1.0** (pure bullish); 0.0 = neutral / balanced |

May coexist with diagnostic columns in specific implementations; `confluence_score` is the primary handoff to strategies and reporting.

---

## 4. Backtest Metrics (Output)

**Role:** Aggregated performance after a simulation run — **scalars**, not per-bar series.

| Metric (concept)   | Typical name        | Type  | Notes |
|--------------------|---------------------|-------|--------|
| Sharpe Ratio       | e.g. `sharpe_ratio` | float | Risk-adjusted return (definition matches `backtest/metrics.py`) |
| Win Rate           | e.g. `win_rate`     | float | 0.0–1.0 fraction of winning trades |
| Max Drawdown       | e.g. `max_drawdown` | float | Peak-to-trough equity decline (negative or positive convention per implementation) |

Other metrics (profit factor, total return, etc.) may appear in the same result object; the three above are the minimum documented contract for cross-module discussion.

---

## 5. ML Labels (v0.3 Target)

**Role:** Supervised learning targets derived from **triple-barrier** events (take-profit, stop-loss, vertical/time barrier). Used to train models on enriched features without inventing ad hoc labels.

| Concept | Description |
|---------|-------------|
| **Barriers** | Upper (TP), lower (SL), vertical (max holding time / timeout) |
| **Label** | Binary or multiclass: e.g., which barrier was hit first, or {TP, SL, timeout} |

**Types:** Integer or small categorical encoding; stored alongside the **event decision time** (bar index or timestamp) used for labeling.

**Temporal rule:** Labels must be generated with barriers anchored at a **decision time** *t* using **only** price path strictly **after** *t* until first hit or timeout. Training rows must align features at *t* with labels that do not peek at pre-*t* future returns beyond what the barrier definition allows (purged CV / embargo in training splits).

### 5a. Unified triple-barrier targets (`labels/targets.py`)

| Column | Type | Notes |
|--------|------|--------|
| `label_multiclass` | float | ``1`` / ``-1`` / ``0`` / ``NaN`` (same codes as triple-barrier) |
| `label_binary` | float | ``1`` = take-profit hit first; ``0`` = stop or vertical; ``NaN`` if multiclass NaN |
| `label_t0` | int | Bar **position** ``0..n-1`` of decision row |
| `label_t1` | int | Inclusive end position of label window (for :class:`~aprilalgo.ml.cv.PurgedKFold`) |
| `barrier_hit` | str | ``take_profit`` \| ``stop_loss`` \| ``vertical_timeout`` or missing |
| `entry_price` | float | Barrier anchor |
| `barrier_hit_offset` | float | Bars until event |

---

## 6. Signal log (JSONL)

**Role:** Append-only research log under ``outputs/signals/*.jsonl`` (path chosen by caller).

Each line is one JSON object. **Minimal keys:** ``ts``, ``symbol`` (see :func:`~aprilalgo.backtest.logger.validate_event`).

**Full contract** (all keys present; use ``null`` when unknown at log time): ``ts``, ``symbol``, ``tf``, ``model_id``, ``features_hash``, ``pred``, ``pred_proba`` (list aligned with model classes), ``label_multiclass``, ``label_binary``, ``meta_pred``, ``outcome``, ``pnl``. Strategies may also include legacy fields such as ``bar_index``, ``pred_proba_tp``, ``event``.

Use :class:`~aprilalgo.backtest.logger.SignalJsonlLogger` and :func:`~aprilalgo.backtest.logger.log_event`.

---

## 7. Regime features (v0.4)

| Column | Type | Notes |
|--------|------|--------|
| `realized_vol` | float | Rolling stdev of log returns |
| `vol_regime` | float | Quantile bucket index ``0..k-1`` (``NaN`` where vol undefined) |

**HMM path:** when ``use_hmm=True`` in :func:`~aprilalgo.meta.regime.add_vol_regime`, install ``hmmlearn`` via the project extra ``hmm`` (``uv sync --extra hmm`` / ``pip install aprilalgo[hmm]``); wheels may not exist for every Python version.

---

## 8. Explainability artifacts (v0.4.x)

| Artifact | Type | Notes |
|----------|------|-------|
| `shap_values.csv` | table | Long-form rows: ``sample_idx``, ``feature``, ``shap_value`` |
| `shap_importance.csv` | table | ``feature``, ``mean_abs_shap``, ``rank`` |

Generated by CLI: ``python -m aprilalgo.cli shap --config ...``.

---

## 9. Information-driven bars (v0.4.x)

Built from source OHLCV rows by event thresholds (tick/volume/dollar).

| Column | Type | Notes |
|--------|------|-------|
| `datetime` | datetime | Close time of aggregated bar chunk |
| `open/high/low/close/volume` | float | Standard OHLCV aggregation |
| `bar_type` | str | `tick` / `volume` / `dollar` |
| `threshold` | float | Trigger threshold used to build bars |
| `source_rows` | int | Number of source rows in aggregate |
| `dollar_value` | float | Present for dollar bars |

CLI usage: ``python -m aprilalgo.cli bars --input ... --bar-type volume --threshold ... --output ...``.

**ML pipeline (train / evaluate / predict / importance / SHAP / walk-forward index counts):** When the ML YAML includes ``information_bars.enabled: true``, the loader reads OHLCV from ``source_timeframe`` if set, otherwise from the top-level ``timeframe``, then applies the same aggregation as the CLI builders **in memory** before triple-barrier labels and indicator features. Bar spacing is **irregular in clock time**; triple-barrier ``vertical_bars`` counts **information bars**, not calendar bars. The recipe written to ``meta.json`` under the key ``information_bars`` must match at inference time; ``predict`` / ``shap`` merge the saved recipe from the bundle over the config. For ``ml_xgboost`` backtests, pass the **same raw** OHLCV series (matching ``source_timeframe``) so the strategy can rebuild identical bars; the backtest loop steps over the bar frame when the strategy sets ``_backtest_bars_df``.

---

## 10. Walk-forward output (CLI JSON)

| Key | Type | Notes |
|-----|------|-------|
| `summary.n_splits` | int | Number of generated folds |
| `summary.coverage_pct` | float | Test-index coverage over input bars |
| `summary.mean_train_size` | float | Average train window length |
| `summary.mean_test_size` | float | Average test window length |
| `splits[].test_return` | float | Close-to-close return on each test window |

---

## 11. Model bundle — sampling metadata (v0.5)

**Role:** Record how row-level ``sample_weight`` was chosen during training and document the YAML contract for reproducibility.

### YAML: top-level ``sampling`` (optional)

When the key is **absent**, training uses **uniform** row weights (same as ``strategy: none``).

| Field | Type | Applies to | Notes |
|-------|------|------------|--------|
| `strategy` | str | all | ``none`` (default) \| ``uniqueness`` \| ``bootstrap`` |
| `random_state` | int | ``bootstrap`` | RNG seed for ``sequential_bootstrap_sample``; if omitted, falls back to top-level ``random_state`` in the ML config, then ``42`` |
| `n_draw` | int or null | ``bootstrap`` | Number of **with-replacement** index draws; default ``null`` means ``n`` = number of training rows after the label/feature mask |

**Semantics:**

- **``none``:** no ``sample_weight`` passed to XGBoost (library default = uniform).
- **``uniqueness``:** ``sample_weight[i]`` = :func:`~aprilalgo.ml.sampling.uniqueness_weights` from triple-barrier ``label_t0`` / ``label_t1`` (overlap-aware; weights sum to ``n``).
- **``bootstrap``:** one draw of ``n_draw`` indices via :func:`~aprilalgo.ml.sampling.sequential_bootstrap_sample`; each row’s weight is its **multiplicity** in that draw, **normalized** so weights sum to ``n`` (zeros allowed for rows never drawn).

### ``meta.json`` key ``sampling`` (object, written on every ``train`` in v0.5+)

| Field | Type | When present |
|-------|------|----------------|
| `strategy` | str | always: ``none`` \| ``uniqueness`` \| ``bootstrap`` |
| `random_state` | int | ``bootstrap`` (resolved seed used for the draw) |
| `n_draw` | int or ``null`` | ``bootstrap`` only (``null`` = used default ``n``) |

**Inference:** ``predict`` / ``shap`` do not reapply row weights; they only need the trained booster. The ``sampling`` block in ``meta.json`` is for audit and for future tooling that replays training.

**Compatibility:** Bundles from v0.4.x without this key are still loadable; consumers treat missing ``sampling`` as ``{"strategy": "none"}``.

---

## 12. OOF artifacts — primary model (v0.5 Sprint 3)

**Role:** Store **out-of-fold** predictions from a **purged** k-fold pass (same splitter as ``evaluate`` / :class:`~aprilalgo.ml.cv.PurgedKFold`) for meta-labeling and research. Rows align 1:1 with the masked training matrix used by ``train`` / ``_prepare_xy``.

**CLI:** ``python -m aprilalgo.cli oof --config <ml.yaml>`` writes ``oof_primary.csv`` under ``model.out_dir`` and, if ``meta.json`` already exists there (e.g. after ``train``), sets ``meta.json`` key ``oof`` to ``{"path": "oof_primary.csv"}``.

**File ``oof_primary.csv`` (CSV):**

| Column | Type | Notes |
|--------|------|--------|
| `row_idx` | int | Row position ``0..n-1`` in the filtered ``X`` / ``y`` frame |
| `y` | float | Label used for fitting |
| `oof_pred` | float | ``predict`` on the held-out fold |
| `oof_proba_<c>` | float | One column per class ``c`` in sklearn ``classes_`` order (names use the float string, e.g. ``oof_proba_1.0`` for binary positive) |

**Dependencies:** Built with the same optional ``sampling`` weights as ``train`` when ``sampling`` is set in YAML.

**Optional dependency (HMM regimes):** ``add_vol_regime(..., use_hmm=True)`` requires ``hmmlearn``, declared as the project extra ``hmm`` (``uv sync --extra hmm`` / ``pip install aprilalgo[hmm]``). Wheels exist for common CPython versions; source builds need a C toolchain.

---

## Handoff summary

```
Raw OHLCV → Enriched Features → Confluence (optional path) → Backtest Metrics
                    ↓
              ML Labels (v0.3) ← triple-barrier on future path only after feature time
```

When in doubt, verify column names here before writing integration code.
