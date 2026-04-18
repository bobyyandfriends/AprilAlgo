# AprilAlgo — Project State

> **Version:** `0.4.1` (baseline) + `Unreleased` information-bars ML integration  
> **Next target:** `0.5.0` — ML depth (sampling, meta-label, regime, walk-forward tuner)  
> **Tests (current):** 129 collected, 128 pass, 1 skipped (`hmmlearn` optional unless `uv sync --extra hmm`)  
> **Source of truth for tasks:** [`BACKLOG.md`](BACKLOG.md) (120 atomic tasks across 12 sprints)  
> **Updated:** 2026-04-18

---

## 1. Macro view — milestones × status

| Milestone | Status | Anchor files |
|---|---|---|
| **Data layer** | Shipped (v0.1.1 → v0.4.1) | [`src/aprilalgo/data/loader.py`](src/aprilalgo/data/loader.py), [`src/aprilalgo/data/store.py`](src/aprilalgo/data/store.py), [`src/aprilalgo/data/resampler.py`](src/aprilalgo/data/resampler.py), [`src/aprilalgo/data/bars.py`](src/aprilalgo/data/bars.py) |
| **Indicator engine** | Shipped (v0.2) | [`src/aprilalgo/indicators/registry.py`](src/aprilalgo/indicators/registry.py), 11 indicators with bull/bear dual signals |
| **Backtest engine** | Shipped (v0.1.1) + bar-aware loop (Unreleased) | [`src/aprilalgo/backtest/engine.py`](src/aprilalgo/backtest/engine.py), [`src/aprilalgo/backtest/portfolio.py`](src/aprilalgo/backtest/portfolio.py) |
| **Strategy framework** | Shipped + `ml_xgboost` | [`src/aprilalgo/strategies/base.py`](src/aprilalgo/strategies/base.py), [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py) |
| **Confluence / probability** | Shipped (v0.2) | [`src/aprilalgo/confluence/`](src/aprilalgo/confluence/) |
| **Grid tuner** | Shipped (v0.2) | [`src/aprilalgo/tuner/grid.py`](src/aprilalgo/tuner/grid.py), [`src/aprilalgo/tuner/runner.py`](src/aprilalgo/tuner/runner.py), [`src/aprilalgo/tuner/analyzer.py`](src/aprilalgo/tuner/analyzer.py) |
| **Triple-barrier labeling** | Shipped (v0.3) | [`src/aprilalgo/labels/triple_barrier.py`](src/aprilalgo/labels/triple_barrier.py), [`src/aprilalgo/labels/targets.py`](src/aprilalgo/labels/targets.py) |
| **XGBoost training + purged CV** | Shipped (v0.3) | [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py), [`src/aprilalgo/ml/cv.py`](src/aprilalgo/ml/cv.py), [`src/aprilalgo/ml/evaluator.py`](src/aprilalgo/ml/evaluator.py) |
| **Importance + SHAP** | Shipped (v0.4 / v0.4.1) | [`src/aprilalgo/ml/importance.py`](src/aprilalgo/ml/importance.py), [`src/aprilalgo/ml/explain.py`](src/aprilalgo/ml/explain.py) |
| **Information-driven bars** | Shipped into ML pipeline (Unreleased) | [`src/aprilalgo/data/bars.py`](src/aprilalgo/data/bars.py) + [`src/aprilalgo/data/loader.py`](src/aprilalgo/data/loader.py) `load_ohlcv_for_ml`; `meta.json.information_bars` contract |
| **Meta-label (fit)** | Partial — only `fit_meta_logit_purged` | [`src/aprilalgo/labels/meta_label.py`](src/aprilalgo/labels/meta_label.py); **not** persisted or consumed by `ml_xgboost` |
| **Regime detection** | Partial — `vol_regime` column | [`src/aprilalgo/meta/regime.py`](src/aprilalgo/meta/regime.py); **not** wired into training/inference |
| **Sequential bootstrap / uniqueness** | Shipped for **train** / **importance** via YAML `sampling` (Unreleased) | [`src/aprilalgo/ml/sampling.py`](src/aprilalgo/ml/sampling.py) + [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) `_weights_for_training`; **not** replayed at `predict` time (weights are training-only) |
| **Walk-forward splits** | Partial — index splits only | [`src/aprilalgo/tuner/walk_forward.py`](src/aprilalgo/tuner/walk_forward.py); **no** ML × WF tuner |
| **Reporting (HTML)** | Shipped (v0.4 / v0.4.1) | [`src/aprilalgo/reporting/report.py`](src/aprilalgo/reporting/report.py) |
| **Streamlit UI** | Shipped (core + labs) | [`src/aprilalgo/ui/app.py`](src/aprilalgo/ui/app.py), [`src/aprilalgo/ui/pages/*.py`](src/aprilalgo/ui/pages/) |
| **CLI** | Shipped verbs: `train`, `evaluate`, `oof`, `importance`, `predict`, `shap`, `walk-forward`, `bars` | [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) |
| **Release packaging** | Shipped (`pyproject.toml`, `hatchling`) | [`pyproject.toml`](pyproject.toml) |

**Legend:** *Shipped* = covered by tests and documented in `CHANGELOG.md`. *Partial* = implementation exists but not reachable through CLI / strategy / persisted artifacts.

---

## 2. Micro view — Deep Log

Every recent code touch, why it was made, and the test that locks it in.

### 2.1 Information-driven bars wired into ML pipeline (Unreleased)

| File | Change | Rationale | Anchoring test |
|---|---|---|---|
| [`src/aprilalgo/data/bars.py`](src/aprilalgo/data/bars.py) | `apply_information_bars_from_config(df, spec)` dispatch over `{tick, volume, dollar}` | Single entry point so `loader`, `strategy`, and CLI all agree on the bar recipe | [`tests/test_data_bars.py`](tests/test_data_bars.py) |
| [`src/aprilalgo/data/loader.py`](src/aprilalgo/data/loader.py) | `load_ohlcv_for_ml(cfg, symbol)` + `information_bars_enabled` / `resolved_source_timeframe_for_ml` / `information_bars_meta_from_cfg` | ML pipeline loads raw OHLCV at the **source** timeframe and aggregates; predict / SHAP / backtest reuse the identical series via `meta.json` | [`tests/test_loader_ml_bars.py`](tests/test_loader_ml_bars.py) |
| [`src/aprilalgo/data/__init__.py`](src/aprilalgo/data/__init__.py) | Export new symbols in `__all__` | Public API stability | import-time smoke |
| [`src/aprilalgo/ml/features.py`](src/aprilalgo/ml/features.py) | Added `bar_type`, `threshold`, `source_rows`, `dollar_value` to `DEFAULT_EXCLUDED_FROM_FEATURES` | Prevent metadata leakage into the feature matrix | [`tests/test_features.py`](tests/test_features.py) |
| [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) | `_cfg_for_inference` merges `meta.information_bars` over YAML so predict/SHAP re-aggregate the same way as training; `_train_and_save` persists bar recipe | Inference parity with training is a **correctness contract**, not a convenience | [`tests/test_cli_ml.py::test_cli_train_predict_roundtrip_information_bars`](tests/test_cli_ml.py) |
| [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py) | `init` applies `apply_information_bars_from_config` on `price_data` before indicators; exposes `_backtest_bars_df` | Backtest loop must iterate the aggregated series, not the raw one | covered via `tests/test_ml_strategy.py` |
| [`src/aprilalgo/backtest/engine.py`](src/aprilalgo/backtest/engine.py) | `run_backtest` duck-types `strategy._backtest_bars_df` and iterates it if present | Allows other strategies to opt into alternative bar series without changing the base class (intentionally informal until v0.5) | backtest smoke |

### 2.2 Documentation sweep (Unreleased)

| File | Change | Rationale |
|---|---|---|
| [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) | Header + "Today's Handoff" rewritten to v0.4.1 + bars; "Suggested next steps" now reflects v0.5 North Star | Eliminate drift with older v0.2 / v0.3 copy |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Header metadata → v1.1 living doc; v0.5 bullet for information bars | Align roadmap section with shipped modules |
| [`CHANGELOG.md`](CHANGELOG.md) | `[Unreleased]` → Added ML-bars block; Planned trimmed | Reflect what is actually shipped this session |
| [`README.md`](README.md) | CLI quick-commands table expanded (`predict`, `importance`, `shap`, `walk-forward`, `bars`) | Every shipped verb is discoverable |
| [`CLAUDE.md`](CLAUDE.md) | `Current version 0.4.1`; ML CLI description broadened | Primary context file for Cursor chats |
| [`configs/ml/default.yaml`](configs/ml/default.yaml) | Commented `information_bars` block | Self-documenting config |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | §9 subsection "Information-driven bars (v0.4.x)" | Column contract + `meta.json` implications |
| [`docs/TRIPLE_BARRIER_MATH.md`](docs/TRIPLE_BARRIER_MATH.md) | Note on vertical-barrier interpretation under bar-aggregated series | Prevents "days" mental model when using tick/volume bars |
| [`AGENTS.md`](AGENTS.md) | Test count reference `~129 tests` | Test count honesty |
| [`docs/MODEL_ROUTING.md`](docs/MODEL_ROUTING.md) | Sprint todo routing marked "(AprilAlgo v0.3 / v0.4 — historical)" | Stop re-executing old sprints |

### 2.3 Primary OOF (Sprint 3, Unreleased)

| File | Change | Rationale | Anchoring test |
|---|---|---|---|
| [`src/aprilalgo/ml/oof.py`](src/aprilalgo/ml/oof.py) | `compute_primary_oof` + optional `sample_weight` per fold | Stacked purged-OOF predictions for meta-labeling (Sprint 4) | [`tests/test_oof.py`](tests/test_oof.py) |
| [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) | `oof` subcommand; `_xgb_estimator_factory` shared with `evaluate` | One factory definition; CSV + `meta.json` `oof.path` | [`tests/test_cli_ml.py::test_cli_oof_writes_csv`](tests/test_cli_ml.py) |

### 2.4 ML sampling weights (Sprint 2, Unreleased)

| File | Change | Rationale | Anchoring test |
|---|---|---|---|
| [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) | `_weights_for_training` implements `uniqueness` and `bootstrap`; `_sampling_meta` persists bootstrap `random_state` / `n_draw` | Opt-in overlap-aware and sequential-bootstrap weights for XGBoost `fit` | [`tests/test_cli_ml.py`](tests/test_cli_ml.py) `test_cli_train_uniqueness_persists_meta`, `test_cli_train_bootstrap_persists_meta`; [`tests/test_sampling.py`](tests/test_sampling.py) |

### 2.5 Test count evolution

- `0.3.0`: ~60 tests (ML core)
- `0.4.0`: ~95 tests (meta-label, regime, WF splits, portfolio runner, reporting, UI smoke)
- `0.4.1`: ~113 tests (SHAP, information bars, walk-forward CLI polish)
- `Unreleased (post Sprint 3)`: **129 tests** (128 pass / 1 skipped — `hmmlearn` without `--extra hmm`)

---

## 3. Warning Zone — technical debt & traps

> Each item has: **What**, **Why it's dangerous**, **Pressure released by** (which sprint fixes or mitigates it).

1. **Fixture OHLCV is tiny (121 rows) — `tests/fixtures/daily_data/TEST_daily.csv`.**
   - **Danger:** v0.5 purged walk-forward tuner (Sprint 8) needs enough bars to fit ≥ 2 folds × ≥ 1 embargo window × a usable grid. With 121 rows the tuner tests will skip or produce meaningless means.
   - **Pressure released by:** Sprint 8 must introduce either a longer fixture (`tests/fixtures/daily_data/TEST_LONG_daily.csv`) or a parametric synthetic generator (`tests/helpers/synth_prices.py`).

2. **`_backtest_bars_df` is duck-typed on strategy instances.**
   - **Where:** [`src/aprilalgo/backtest/engine.py`](src/aprilalgo/backtest/engine.py) reads `getattr(strategy, "_backtest_bars_df", None)`; [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py) sets it.
   - **Danger:** Per-regime routing (Sprint 7) will add *another* per-bar attribute (bundle-per-row); an informal attribute soup gets unsafe.
   - **Pressure released by:** Sprint 7 elevates this to an explicit protocol on [`src/aprilalgo/strategies/base.py`](src/aprilalgo/strategies/base.py) (`iter_bars() -> pd.DataFrame`).

3. **`docs/SESSION_HANDOFF.md` interior (below "Today's Handoff") still reads like the original v0.2 handoff.**
   - **Danger:** Agents scroll past the fresh block and re-execute stale v0.2 sprints.
   - **Pressure released by:** Sprint 11-T09 deletes the v0.2 bootstrapping paragraphs and the template (this PR) forbids stale carry-over.

4. **`src/aprilalgo/meta/regime.py` HMM branch depends on `hmmlearn` (optional extra `hmm`).**
   - **Danger:** Without ``uv sync --extra hmm`` / a wheel for your Python, `use_hmm=True` raises at runtime and the HMM smoke test skips.
   - **Pressure released by:** Documented in §7 / §12 of [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) and [`README.md`](README.md); Sprint 6+ keeps `regime.use_hmm: false` as the safe default in shared configs.

5. **`ml_strategy.py` exit logic mixes entry threshold with a hardcoded `0.5 * entry_proba_threshold` exit heuristic.**
   - **Where:** [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py).
   - **Danger:** Changing `entry_proba_threshold` silently shifts exit semantics; meta-gate in Sprint 5 will multiply the surprise.
   - **Pressure released by:** Sprint 5-T03 adds explicit `meta_proba_threshold` and (nice-to-have Sprint 5-T04) separates `exit_proba_threshold` as a first-class ctor arg (tracked as backlog candidate S5-T04 if we decide).

6. **`information_bars` config only validated at call site, no schema.**
   - **Danger:** `bar_type: "ticks"` (plural typo) silently falls through to `raise ValueError` only at `_train_and_save` time.
   - **Pressure released by:** Sprint 11-T02 adds a DATA_SCHEMA entry; a later hardening pass can introduce pydantic or jsonschema validation. Not scheduled in v0.5.

7. **No public dtype guarantees on engineered feature columns.**
   - **Where:** [`src/aprilalgo/ml/features.py`](src/aprilalgo/ml/features.py) returns whatever `pandas` infers.
   - **Danger:** XGBoost warns on `object` columns; regime bucket stored as `int`/`category` is a looming mismatch when per-regime routing joins back on the test set.
   - **Pressure released by:** Sprint 7 enforces `vol_regime` as `int16` before split.

8. **`reporting/report.py` section ids are positional (no stable anchor).**
   - **Danger:** Adding Sprint 10 sections between existing ones could break external links.
   - **Pressure released by:** Sprint 10-T01..T04 introduces stable `section-sampling`, `section-meta`, `section-regime`, `section-wf-tuner` ids (test-locked).

9. **`tests.md` is manually regenerated.**
   - **Danger:** Drift between the file and CI truth; every sprint explicitly re-runs `pytest --collect-only -q` (each sprint final task T10).

---

## 4. Operational contracts — `meta.json` target shape after v0.5

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

**Compatibility rule:** v0.4.1 bundles must keep loading — any missing key above is treated as "disabled" by both the CLI and `MLStrategy`.

**Additional artifacts in `model.out_dir`:**

```
models/xgboost/<run>/
├── meta.json
├── xgboost.json
├── oof_primary.csv            # Sprint 3
├── meta_logit.json            # Sprint 4
├── meta_oof.csv               # Sprint 4
├── regime_index.json          # Sprint 7 (when groupby=true)
├── regime_0/                  # Sprint 7
│   ├── meta.json
│   └── xgboost.json
├── regime_1/
│   └── ...
├── wf_tune_results.csv        # Sprint 9
├── shap_values.csv            # v0.4.1
├── shap_importance.csv        # v0.4.1
├── regime_0_shap_values.csv   # Sprint 10
├── regime_0_shap_importance.csv
└── ...
```

**CLI verbs after v0.5 (new in bold):**

```
aprilalgo.cli train | evaluate | importance | predict | shap | walk-forward | bars
                | oof*       | meta-train* | wf-tune*
```

`*` = Sprint 3 / 4 / 9 additions.

---

## 5. Open questions (to confirm before / during v0.5)

1. **Regime inclusion default.** Should `vol_regime` default to an included feature (current plan) or stay behind `regime.include_as_feature: true`? → Sprint 6.
2. **Per-regime default.** Single model with `vol_regime` column, OR per-regime bundles by default? Plan keeps single-model + `groupby=false` default. → Sprint 7.
3. **Meta-label family.** Stay with logistic regression for 0.5.0 or allow XGB classifier for meta? Plan keeps logistic only (auditability). → Sprint 4.
4. **WF tuner metric default.** Currently planned `f1_macro`; confirm vs `neg_log_loss` for calibrated gating. → Sprint 8.
5. **`hmmlearn` wheels vs source builds.** Resolved for packaging as ``pip install aprilalgo[hmm]`` / ``uv sync --extra hmm``; CI matrices on bleeding-edge Python may still skip HMM until wheels exist.

---

## 6. References

- [`BACKLOG.md`](BACKLOG.md) — the 120-task task list (source of truth).
- [`CHANGELOG.md`](CHANGELOG.md) — versioned history.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design across versions.
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) — column contracts + bundle schema.
- [`docs/MODEL_ROUTING.md`](docs/MODEL_ROUTING.md) — Cursor model tier policy.
- [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — per-session context and handoff template (Sprint-scoped after v0.5).
