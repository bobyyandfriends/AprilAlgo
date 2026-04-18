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

**Full contract** (all keys present; use ``null`` when unknown at log time): ``ts``, ``symbol``, ``tf``, ``model_id``, ``features_hash``, ``pred``, ``pred_proba`` (list aligned with model classes), ``label_multiclass``, ``label_binary``, ``meta_pred``, ``pred_proba_meta`` (``null`` when the ``ml_xgboost`` meta gate is off; otherwise ``float`` in ``[0, 1]`` — probability that the primary prediction is **correct** per the stacked meta logit, same target as ``meta_logit.json``), ``outcome``, ``pnl``. Strategies may also include legacy fields such as ``bar_index``, ``pred_proba_tp``, ``event``.

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

### 11.1 YAML: top-level ``sampling`` (optional)

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

### 11.2 ``meta.json`` key ``sampling`` (object, written on every ``train`` in v0.5+)

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

---

## 13. Meta-label bundle (v0.5 Sprint 4)

**Role:** Secondary logistic predicts whether the **primary** model’s OOF prediction matched ``y`` (meta target ``z``). Used for gating in Sprint 5+.

**Pipeline:** ``train`` → ``oof`` → ``meta-train`` (CLI). ``meta-train`` reads ``oof_primary.csv`` (must align row-for-row with :func:`~aprilalgo.cli._prepare_xy` for the same config + symbol).

**Degenerate case:** if primary OOF matches ``y`` on every row (or never), meta labels ``z`` are a single class and :func:`~aprilalgo.labels.meta_label.fit_meta_logit_purged` raises ``ValueError`` — regenerate OOF or use a non-trivial primary model.

**Artifacts under ``model.out_dir``:**

| File | Notes |
|------|--------|
| ``meta_logit.json`` | JSON: ``feature_names`` (includes trailing ``primary_pred``), ``coef``, ``intercept``, ``classes_`` — load via :func:`~aprilalgo.ml.meta_bundle.load_meta_logit_bundle` |
| ``meta_oof.csv`` | Purged meta-model OOF prob of ``z=1``: columns ``row_idx``, ``y_true``, ``z``, ``meta_oof_proba`` |

**``meta.json`` key ``meta_logit`` (after ``meta-train``):**

| Field | Type | Notes |
|-------|------|--------|
| `enabled` | bool | ``true`` when bundle was written |
| `path` | str | Relative path, e.g. ``meta_logit.json`` |
| `threshold` | float | Default ``0.5``; override with YAML ``meta_label.threshold`` |

**YAML (optional) ``meta_label`` block:** ``n_splits``, ``embargo`` (default to top-level ``cv``), ``threshold`` for persistence in ``meta.json``.

**Compatibility:** Bundles without ``meta_logit`` treat meta-gate as disabled (Sprint 5).

---

## 14. Regime conditioning in the ML pipeline (v0.5 Sprint 6)

**Role:** Optional **volatility regime** columns are added to the OHLCV frame **before** indicator features are built, so ``vol_regime`` can enter the XGBoost feature matrix while ``realized_vol`` stays a diagnostic-only column (excluded from ``X`` by default).

### YAML: top-level ``regime`` (optional)

| Field | Type | Notes |
|-------|------|--------|
| `enabled` | bool | When ``true``, :func:`~aprilalgo.meta.regime.add_vol_regime` runs on the loaded bar frame immediately after optional information bars and **before** triple-barrier labels and ``build_feature_matrix`` |
| `window` | int | Rolling window for realized vol (default ``20``) |
| `n_buckets` | int | Quantile buckets for rule-based regimes (default ``3``) |
| `use_hmm` | bool | If ``true``, HMM path (requires ``hmmlearn``); default ``false`` |

### ``meta.json`` key ``regime`` (object, written on every ``train`` in v0.5+)

Mirrors the resolved YAML (or defaults when the block is absent): ``enabled``, ``window``, ``n_buckets``, ``use_hmm``.

**Inference:** ``predict`` and ``shap`` call :func:`~aprilalgo.cli._cfg_for_inference`, which merges ``meta.regime`` from the bundle over the YAML so the **same window and bucket count** are used even if the local config drifts.

**Backtests:** ``ml_xgboost`` applies :func:`~aprilalgo.meta.regime.add_vol_regime` to ``work`` after optional information bars when ``bundle.meta.regime.enabled`` is true, then runs the indicator pipeline — matching train order.

**Feature matrix:** ``vol_regime`` is a normal numeric feature column; ``realized_vol`` is listed in :data:`~aprilalgo.ml.features.DEFAULT_EXCLUDED_FROM_FEATURES` and is not part of ``X``.

**Compatibility:** Bundles without a ``regime`` key are treated as ``{"enabled": false, ...}``; training without the YAML block matches the pre–Sprint 6 feature matrix.

---

## 15. Per-regime model routing (v0.5 Sprint 7)

**Role:** When ``regime.enabled`` and ``regime.groupby: true``, ``train`` fits **one XGBoost bundle per ``vol_regime`` bucket** instead of a single combined model. Inference routes each row to the bundle trained on that bucket (with undefined ``vol_regime`` mapped to the **default** bucket).

### Artifacts under ``model.out_dir``

| File / directory | Notes |
|------------------|--------|
| ``regime_index.json`` | JSON: ``buckets`` maps string bucket id ``"0"``, ``"1"``, … → subdirectory name ``regime_<k>``; ``default`` is one of those subdirectory names (the smallest trained bucket id) |
| ``regime_<k>/`` | Full model bundle (``meta.json`` + ``xgboost.json``); ``meta.json`` includes ``regime.bucket`` (int) and ``regime.groupby: true`` |

### Training semantics

- Rows with NaN ``vol_regime`` are included when training the **smallest** observed bucket id (same default used at inference for NaN rows).
- Buckets with too few rows or a single-class ``y`` (binary) are skipped.
- All successful sub-bundles must share **identical** ``feature_names`` (same ``X`` columns).

### CLI / strategy

- ``predict`` reads ``regime_index.json`` if present; builds ``X`` once, then batches predictions per bucket. CSV ``proba_<c>`` columns use the **sorted union** of ``classes_`` across buckets; for **binary** tasks, two-column ``predict_proba`` rows are mapped to canonical classes ``0.0`` and ``1.0`` even when a bucket’s saved ``classes_`` lists only one label.
- ``shap`` uses the **default** sub-bundle from ``regime_index.json`` when the index exists, unless ``shap --per-regime`` is used (see §16).
- ``ml_xgboost`` loads every sub-bundle at ``init`` and selects the bundle per bar from ``vol_regime`` on the feature row.

**Compatibility:** Absent ``regime_index.json``, behavior matches a single-bundle layout (Sprint 6 and earlier).

---

## 16. Per-regime SHAP artifacts (v0.5 Sprint 10)

**Role:** When ``regime_index.json`` exists, optional **per-bucket** SHAP tables align each explanation with the bundle actually used for rows in that ``vol_regime`` bucket.

### CLI

| Invocation | Output (under ``model.out_dir``) |
|------------|----------------------------------|
| ``python -m aprilalgo.cli shap --config …`` (default) | Single ``shap_values.csv`` / ``shap_importance.csv`` paths from flags, using the **default** sub-bundle only (unchanged from §15). |
| ``python -m aprilalgo.cli shap --config … --per-regime`` | For each bucket *k* with at least one routed row: ``regime_<k>_shap_values.csv`` and ``regime_<k>_shap_importance.csv`` (same column contracts as global SHAP tables in §8). |

**Requirements:** Feature matrix must include ``vol_regime`` (same routing rules as ``predict``). Buckets with no rows in the current config slice are skipped.

### Library

- :func:`~aprilalgo.ml.explain.shap_values_per_regime` accepts a ``dict`` of :class:`~aprilalgo.ml.trainer.ModelBundle` per bucket id and a matching ``dict`` of feature :class:`~pandas.DataFrame` slices.
- :func:`~aprilalgo.ml.explain.load_regime_bundles_shap` loads all sub-bundles listed in ``regime_index.json``.

### Reporting

- HTML section ``section-wf-tuner`` summarizes ``wf_tune_results.csv``; regime **bucket counts** and **per-regime accuracy** use ``section-regime`` in :mod:`aprilalgo.reporting.report` (distinct from the legacy backtest ``section-regime-timeline`` block).

---

## Handoff summary

```
Raw OHLCV → Enriched Features → Confluence (optional path) → Backtest Metrics
                    ↓
              ML Labels (v0.3) ← triple-barrier on future path only after feature time
```

When in doubt, verify column names here before writing integration code.
