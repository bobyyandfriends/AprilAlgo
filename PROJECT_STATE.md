# AprilAlgo — Project State

The single living tracker. Edited at the end of every session. For any other question, see `AGENTS.md` §9 "Canonical references".

---

## 1. Status at a glance

| Field | Value |
|---|---|
| Current version | `0.5.0` (shipped 2026-04-17) |
| Next target | `0.6.0+` — planning in §3 below |
| Active branch | `main` |
| Test count (live) | run `uv run pytest tests/ --collect-only -q` — **184 collected** at last check |
| Last full green run | `uv run pytest tests/ -q` → **183 passed / 1 skipped / 0 failed** (2026-04-18, after §B sweep; skip = `hmmlearn` unless `uv sync --extra hmm`) |

---

## 2. Today's handoff

### Handoff — 2026-04-18 (session 2: architectural follow-ups)

- **Sprint / theme**: Clear every remaining `AUDIT_FINDINGS.md` §B item in a single pass — swept all 23 architectural/design concerns left over from Phase 1's bug hunt.
- **Tasks completed**: All 23 §B items (B1–B23) addressed. Highlights:
  - **Backtest accuracy** (B1, B2, B3): `Portfolio` now supports `margin_ratio` + `borrow_rate_bps_per_day`; new `backtest/metrics_v2.py` computes CAGR / Sharpe / Sortino from the equity curve with auto-inferred annualisation; `BaseStrategy._backtest_bars_df` contract is standardised and the engine validates shape + length before the loop.
  - **ML pipeline** (B4, B7, B8): `sample_weight` threads through `purged_cv_evaluate` and `ml_walk_forward_tune` via a new `ml/pipeline.py`; `ModelBundle.predict_proba` anchors binary columns on `self.classes_` and tolerates degenerate one-class bundles; `save_model_bundle` guards against persisting encoded-index `clf.classes_` as real labels.
  - **Purged CV** (B5, B6): opt-in symmetric embargo; `_purge_train`, `_embargo_train`, `fold_train_test_interval_disjoint` vectorised — quadratic pure-Python loops removed.
  - **Walk-forward** (B9): auto-computed `test_size` uses ceiling division so `walk_forward_splits` now emits exactly `n_folds` tiles of `[min_train, n)`; explicit `test_size` preserves legacy cover-the-tail behaviour.
  - **Indicators** (B12, B13, B14, B15, B16, B17): numeric coercion in `confluence/scorer`, uniqueness assertions in `timeframe_aligner`, HMM / qcut / log-return edge cases in `meta/regime`, opt-in Wilder RSI, Hurst docstring aligned with implementation, NaN-safe Ehlers recursion.
  - **UX / ops** (B18, B19, B20, B21, B22, B23): align log delta, HTML equity footer, `hover_data` column intersection in tuner UI, narrow exception handlers in walk-forward UI, 1h `subprocess` timeouts on all CLI wrapper pages, `hash_features_row` hardened against empty frames + mixed dtypes.
  - **Docstrings** (B11, B16): binary label TP-vs-rest semantics + Hurst trend / mean-revert interpretation now match the implementation.
- **Tasks deferred**: none from §B. Outstanding follow-ups are now new work (see §3).
- **Commits / file diffs**: **new** `src/aprilalgo/ml/pipeline.py`, `src/aprilalgo/backtest/metrics_v2.py`; **modified** `src/aprilalgo/cli.py`, `src/aprilalgo/backtest/{engine.py,portfolio.py,logger.py}`, `src/aprilalgo/strategies/{base.py,configurable.py,rsi_sma.py,demark_confluence.py,ml_strategy.py}`, `src/aprilalgo/ml/{cv.py,evaluator.py,trainer.py,features.py}`, `src/aprilalgo/tuner/{walk_forward.py,ml_walk_forward.py}`, `src/aprilalgo/indicators/{rsi.py,hurst.py,ehlers.py}`, `src/aprilalgo/confluence/{scorer.py,timeframe_aligner.py}`, `src/aprilalgo/meta/regime.py`, `src/aprilalgo/labels/targets.py`, `src/aprilalgo/reporting/report.py`, `src/aprilalgo/ui/pages/{tuner.py,walk_forward_lab.py,model_lab.py,model_trainer.py,model_metrics.py}`.
- **Test result**: `uv run pytest -q` → **183 passed / 1 skipped / 0 failed** (same total as Phase 1). Two transient regressions from this session (over-aggressive `walk_forward_splits` cap; over-strict binary `classes_` assertion) were caught on the first full-suite run and fixed in-session — the final green run is the recorded baseline.
- **New public symbols**:
  - `aprilalgo.ml.pipeline`: `prepare_xy`, `xgb_estimator_factory`, `weights_for_training`, `apply_regime_if_enabled`.
  - `aprilalgo.backtest.metrics_v2`: `compute_metrics_from_equity`, `infer_periods_per_year`.
  - `aprilalgo.indicators.rsi.rsi`: new `mode` kwarg (`"sma"` default, `"wilder"`).
  - `aprilalgo.ml.cv.PurgedKFold`: new `symmetric_embargo` constructor flag (default `False`, back-compat).
  - `aprilalgo.ml.evaluator.purged_cv_evaluate`: new `sample_weight` kwarg.
  - `aprilalgo.backtest.Portfolio`: new `margin_ratio` + `borrow_rate_bps_per_day` constructor kwargs.
  - `aprilalgo.strategies.BaseStrategy`: new `_backtest_bars_df` / `_backtest_frame_matches_input` attributes.
- **`meta.json` / bundle schema changes**: none. `docs/DATA_SCHEMA.md` unchanged. Bundles from v0.5.0 keep loading.
- **Docs updated**: `CHANGELOG.md` `[Unreleased]` Fixed block gains §B1–§B23 detail. This file (§2 rewritten for today, §3 next-sprint list updated, §4 warning zone pruned). `AUDIT_FINDINGS.md` is now fully actioned.
- **Warning zone deltas**: §4.1–§4.18, §4.19, §4.20, §4.21, §4.22, §4.23 all removed (shipped). §4.11 kept with a different wording — the docstring fix does not change the underlying TP-vs-rest design decision. Stray `ui/pages/model_metrics.py` import reference removed.
- **Next gate**: **approved** — repo is test-green, warning zone is empty of audit debt, pipeline is ready for the next feature sprint.

---

## 3. Next sprint / active work

### Theme (proposed)
The §B audit backlog is cleared. The next sprint is a greenfield choice between product polish (SHAP narrative UX, WF analytics polish) and new research (information-bars presets, per-regime feature expansion). Pick one below and reshape §2 / §3 at kick-off.

### Candidate task list

- [ ] **Metrics v2 migration** — now that `backtest/metrics_v2.py` exists, decide whether `report.py` / UI pages / CHANGELOG metrics should switch over. Requires a side-by-side regression harness (legacy vs v2 on the same equity curves) so any shift in headline numbers is intentional and documented. **Tier: Heavy.**
- [ ] **Symmetric embargo as default** — gate behind a deprecation warning for one version, then flip `PurgedKFold(symmetric_embargo=True)` to default. Needs a test ensuring documented metric deltas on the main configs. **Tier: Standard.**
- [ ] **Borrow-cost / margin integration in CLI smoke configs** — wire `margin_ratio` + `borrow_rate_bps_per_day` into `configs/smoke_backtest*.yaml` and `main.py` YAML so short-heavy strategies are benchmarked with realistic costs. **Tier: Standard.**
- [ ] **SHAP narrative UX polish** — from the earlier `Planned` list; render per-regime summaries and top-N feature stories in `ui/pages/model_metrics.py`. **Tier: Standard.**
- [ ] **Walk-forward analytics deep polish** — visualise fold stability, grid heatmaps, meta-label calibration from `wf_tune_results.csv`. **Tier: Heavy.**
- [ ] **Information-bar presets research** — nail down default dollar/tick/volume thresholds per symbol-class; document in `docs/DATA_SCHEMA.md`. **Tier: Heavy.**

Pick a subset of the above for the next sprint; reshape this section at the start of that sprint.

---

## 4. Warning zone — technical debt & traps

The `AUDIT_FINDINGS.md` §B backlog (§4.1–§4.23) was cleared on 2026-04-18 (session 2). What remains below is either (a) a design decision that survives the fix, or (b) new follow-up debt introduced by the sweep itself. Add new items here as they are discovered.

### Design decisions worth revisiting

- **Binary label TP-vs-rest collapse** (`labels/targets.py::label_binary`). Fix B11 only clarified the docstring; the underlying choice to fold stop-loss + vertical-timeout into a single `0` class remains. If a future strategy needs directional gating (distinguishing "uncertain" from "downside"), switch primary to multiclass and have the strategy consume `P(class=+1)` directly — do **not** reintroduce a 3-state binary schema.
- **Legacy `metrics.py` still in the report path** (§B2). The new `backtest/metrics_v2.py` is import-available but no call-sites were migrated, so existing backtests keep their (mildly wrong for non-daily) numbers. Migration requires a regression harness before headlines can shift.
- **Symmetric embargo opt-in only** (§B5). Default is still one-sided for back-compat. Flip default only after publishing a metric-delta comparison on canonical configs.

### Operational reminders

- `BaseStrategy._backtest_bars_df` + `_backtest_frame_matches_input` are now a **public contract**. Any new strategy that resizes rows during `init` must set `_backtest_frame_matches_input = False` (see `MLStrategy`), otherwise `run_backtest` will raise on the length assertion.
- `PurgedKFold(symmetric_embargo=True)` drops more train rows — expect OOF coverage to shrink slightly; add a coverage assertion if a pipeline starts relying on exact row counts.
- `ModelBundle.predict_proba` now treats a 1-class bundle as `classes_ = [0.0]`-implied; downstream code relying on `len(bundle.classes_) == 2` must handle the degenerate case explicitly.
- `walk_forward_splits` with explicit `test_size` still emits as many folds as fit in `[min_train, n)` (not capped at `n_folds`). This is intentional (preserves the pre-B9 behaviour for callers who pick `test_size` manually).

---

## 5. Operational contracts

### `meta.json` target shape (v0.5)

```json
{
  "version": "0.5.0",
  "task": "binary",
  "feature_names": ["..."],
  "classes": [0, 1],
  "indicators": {"...": "..."},
  "information_bars": {"enabled": true, "bar_type": "dollar", "threshold": 1000000.0, "source_timeframe": "5min"},
  "sampling": {"strategy": "uniqueness", "random_state": 42, "n_draw": null},
  "oof": {"path": "oof_primary.csv"},
  "meta_logit": {"enabled": true, "path": "meta_logit.json", "threshold": 0.5},
  "regime": {"enabled": true, "window": 20, "n_buckets": 3, "use_hmm": false, "groupby": false},
  "wf_tuner": {"ran": true, "results_path": "wf_tune_results.csv", "metric": "f1_macro", "top": {"learning_rate": 0.05, "max_depth": 4}}
}
```

**Compatibility rule.** v0.4.1 bundles must keep loading — any missing key above is treated as "disabled" by both the CLI and `MLStrategy`.

### Artifact layout under `model.out_dir`

```
models/xgboost/<run>/
├── meta.json
├── xgboost.json
├── oof_primary.csv            # Sprint 3
├── meta_logit.json            # Sprint 4
├── meta_oof.csv               # Sprint 4
├── regime_index.json          # Sprint 7 (groupby=true)
├── regime_<k>/{meta.json, xgboost.json, ...}
├── wf_tune_results.csv        # Sprint 9
├── shap_values.csv            # v0.4.1
├── shap_importance.csv        # v0.4.1
└── regime_<k>_shap_*.csv      # Sprint 10
```

### CLI verbs (v0.5)

```
aprilalgo.cli train | evaluate | oof | meta-train | predict | importance | shap | walk-forward | wf-tune | bars
```

### Load-bearing invariants

- No look-ahead anywhere — `on_bar(idx, row)` must never read `data.iloc[idx+k]` for `k > 0`.
- Indicator columns follow `{name}_{params}_(bull|bear)`; no unparameterized collisions.
- Multi-timeframe alignment uses **forward-fill only**; higher-TF value persists until the next bar.
- Purged CV applies embargo around test block boundaries.
- `ml_xgboost` in backtest must iterate the **same bar series** as training when `information_bars` is in `meta.json` (see `_backtest_bars_df`).

---

## 6. Open questions

1. Default for `vol_regime` inclusion — always include, or stay behind `regime.include_as_feature: true`? (current behavior: always include when `regime.enabled`).
2. Default routing — single model with `vol_regime` column, or per-regime bundles? (current: single-model + `groupby=false`).
3. Meta-label family — keep logistic only (auditability) or allow XGB classifier for meta?
4. `wf_tuner.metric` default — currently `f1_macro`; confirm vs `neg_log_loss` for calibrated gating.
5. Binary label fusion (§4.11) — should primary stay multiclass and strategies consume TP probability only?

---

## Appendix A — Sprint handoff log (collapsed)

Chronological, append-only. One line per sprint. Detailed per-sprint narratives live in git history and in `CHANGELOG.md`.

- **Sprint 1** — 2026-04-17 — Sample-weight plumbing (trainer signature only) — shipped; `meta.json.sampling` default `none`.
- **Sprint 2** — 2026-04-18 — Sequential bootstrap + uniqueness weights — shipped.
- **Sprint 3** — 2026-04-18 — Primary OOF capture — shipped; `compute_primary_oof`, `oof_primary.csv`. Optional `hmm` extra added.
- **Sprint 4** — 2026-04-18 — Meta-label bundle (persisted) — shipped; `MetaLogitBundle`, `meta_logit.json`, `meta_oof.csv`. Degenerate-`z` guard added.
- **Sprint 5** — 2026-04-18 — `ml_xgboost` meta gate — shipped; stacks primary pred, `meta_proba_threshold`, `pred_proba_meta` logged.
- **Sprint 6** — 2026-04-18 — Regime as a feature — shipped; `add_vol_regime`, `vol_regime` in `X`, `meta.json.regime` persisted.
- **Sprint 7** — 2026-04-18 — Per-regime bundles + routing — shipped; `regime_index.json`, per-regime train + `predict`/`ml_xgboost` routing, default sub-bundle for SHAP.
- **Sprint 8** — 2026-04-18 — WF tuner core — shipped; `ml_walk_forward_tune`, `expand_grid`, `aggregate_grid`.
- **Sprint 9** — 2026-04-18 — `wf-tune` CLI + Streamlit Tuner tab — shipped; `wf_tune_results.csv`.
- **Sprint 10** — 2026-04-18 — Reporting + per-regime SHAP — shipped; stable section ids + `render_full_ml_report_html`; `shap --per-regime`.
- **Sprint 11** — 2026-04-18 — Documentation sync — shipped.
- **Sprint 12** — 2026-04-17 — Release prep + v0.5.0 tag — shipped.
- **Post-v0.5 audit** — 2026-04-18 — Principal-Staff bug hunt (14 fixes) + documentation consolidation (17 → 11 files).
- **Post-v0.5 audit, session 2** — 2026-04-18 — All 23 `AUDIT_FINDINGS.md` §B architectural follow-ups (B1–B23) shipped. New modules: `ml/pipeline.py`, `backtest/metrics_v2.py`. Full suite 183/1/0.

---

## Appendix B — CLI edge cases (from the old `DEBUG_LOG.md`)

These are CLI failures that are not simple code fixes — they are prerequisites, degenerate fixture data, or intentional guardrails.

### B.1 `meta-train` with `configs/ml/default.yaml` and symbol `TEST`

```bash
uv run python -m aprilalgo.cli meta-train --config configs/ml/default.yaml
# → ValueError: Meta labels z = (oof_pred == y) have a single class; need both
#   correct and incorrect primary OOF rows.
```

**Why.** On the `TEST` daily fixture with the default triple-barrier settings, the filtered binary labels `y` are a single class (`0` only). OOF primary predictions match `y` on every row, so `z = (oof_pred == y)` is all `1` and the meta logit cannot be fit. `cmd_meta_train` is correctly rejecting degenerate `z`.

**Fix.** Use a config that yields mixed labels (and typically some OOF errors) on the same fixture, then run `train` → `oof` → `meta-train`:
- `configs/ml/meta_train_smoke.yaml` (writes under `models/xgboost/meta_smoke/`).

### B.2 `shap --per-regime` without `regime_index.json`

```bash
uv run python -m aprilalgo.cli shap --config configs/ml/default.yaml --per-regime
# → ValueError: --per-regime requires regime_index.json (train with regime.groupby: true)
```

**Why.** Per-regime SHAP expects a `regime_index.json` + `regime_<k>/` sub-bundles, which only `train` produces when `regime.groupby: true`.

**Fix.** Train with `regime.groupby: true` (and `regime.enabled: true`) and point `--model-dir` at the bundle root:
- Config: `configs/ml/regime_groupby_smoke.yaml`
- Model dir: `models/xgboost/regime_smoke`

### B.3 Backtest `main.py` and `data_dir`

`main.py` previously ignored `data_dir` in YAML, so `symbol: TEST` with data under `tests/fixtures` could not load unless CSVs were copied into `data/`. **Fixed**: `main.py` now passes `cfg.get("data_dir")` to `load_price_data` as an optional `Path`. Smoke configs:
- `configs/smoke_backtest.yaml` — classic strategies on `TEST`
- `configs/smoke_backtest_ml.yaml` — `ml_xgboost` using the bundle from `configs/ml/default.yaml`
