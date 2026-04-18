# AprilAlgo — v0.5 (ML depth) Backlog

> **Target version:** `0.5.0`  
> **Baseline:** `0.4.1` + information-bars ML integration (CHANGELOG "Unreleased")  
> **Principle:** opt-in YAML blocks; every 0.4.1 config must keep passing tests without edits.  
> **Tiers** follow [`docs/MODEL_ROUTING.md`](docs/MODEL_ROUTING.md) — **F** = Fast, **S** = Standard, **H** = Heavy.

Each task is `< 5 min` for an agent, covers **one** functional change, and has a single **DoD** verification command. Do not merge more than one task's diff into a single commit.

---

## Sprint 1 — Sample-weight plumbing (trainer signature only)

- [x] **S1-T01** [H] Add `sample_weight: np.ndarray | pd.Series | None = None` parameter to `train_xgb_classifier` in [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py). No internal use yet. **DoD:** `uv run python -c "from aprilalgo.ml.trainer import train_xgb_classifier; import inspect; assert 'sample_weight' in inspect.signature(train_xgb_classifier).parameters"`
- [x] **S1-T02** [S] Forward `sample_weight` to `clf.fit` in the **binary** branch of `train_xgb_classifier`. **DoD:** `uv run pytest tests/test_trainer.py::test_train_save_load_roundtrip_binary -q`
- [x] **S1-T03** [S] Forward `sample_weight` to `clf.fit` in the **multiclass** branch (apply in the same `LabelEncoder`-encoded space). **DoD:** `uv run pytest tests/test_trainer.py::test_train_multiclass_bundle_takeprofit_index -q`
- [x] **S1-T04** [F] Add a commented `sampling:` block (fields: `strategy`, `random_state`) to [`configs/ml/default.yaml`](configs/ml/default.yaml). **DoD:** `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('configs/ml/default.yaml').read_text())"`
- [x] **S1-T05** [S] Add `_weights_for_training(cfg, t0, t1) -> np.ndarray | None` stub in [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) that returns `None` when the block is missing. **DoD:** `uv run python -c "from aprilalgo.cli import _weights_for_training"`
- [x] **S1-T06** [S] Call `_weights_for_training` from `_train_and_save` and pass the result into `train_xgb_classifier`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_predict_roundtrip -q`
- [x] **S1-T07** [F] Persist a `sampling` stub (`{"strategy": "none"}`) into the `extra_meta` dict of `_train_and_save`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_persists_sampling_meta -q`
- [x] **S1-T08** [S] `tests/test_trainer_sample_weight.py::test_uniform_sample_weight_matches_no_weight`. **DoD:** `uv run pytest tests/test_trainer_sample_weight.py::test_uniform_sample_weight_matches_no_weight -q`
- [x] **S1-T09** [H] `tests/test_trainer_sample_weight.py::test_skewed_sample_weight_changes_feature_importance`. **DoD:** `uv run pytest tests/test_trainer_sample_weight.py::test_skewed_sample_weight_changes_feature_importance -q`
- [x] **S1-T10** [F] Regenerate [`tests.md`](tests.md) with updated header count. **DoD:** `uv run pytest tests/ --collect-only -q` (expect `120 tests collected`)

### Downstream dependencies (Sprint 1)
- [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py), [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py), [`configs/ml/default.yaml`](configs/ml/default.yaml), [`tests/test_trainer.py`](tests/test_trainer.py), new [`tests/test_trainer_sample_weight.py`](tests/test_trainer_sample_weight.py), [`tests.md`](tests.md). No strategy / predict / SHAP changes this sprint.

---

## Sprint 2 — Sequential bootstrap + uniqueness weights in training

- [x] **S2-T01** [F] Document `sampling.strategy` options (`none | uniqueness | bootstrap`) and `n_draw` / `random_state` defaults in the commented YAML block. **DoD:** `grep -n "strategy:" configs/ml/default.yaml`
- [x] **S2-T02** [H] Implement `_weights_for_training` branch `strategy == "uniqueness"` calling `uniqueness_weights(t0, t1)`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_predict_roundtrip -q`
- [x] **S2-T03** [H] Implement `_weights_for_training` branch `strategy == "bootstrap"` returning a weight vector from index multiplicities of `sequential_bootstrap_sample`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_predict_roundtrip -q`
- [x] **S2-T04** [S] Unit test: `tests/test_cli_ml.py::test_cli_train_uniqueness_persists_meta` asserts `meta.json.sampling.strategy == "uniqueness"`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_uniqueness_persists_meta -q`
- [x] **S2-T05** [S] Unit test: `tests/test_cli_ml.py::test_cli_train_bootstrap_persists_meta` asserts `meta.json.sampling.strategy == "bootstrap"` and records `random_state`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_bootstrap_persists_meta -q`
- [x] **S2-T06** [S] Unit test: `tests/test_sampling.py::test_uniqueness_weights_sum_to_n_on_overlap` (new). **DoD:** `uv run pytest tests/test_sampling.py::test_uniqueness_weights_sum_to_n_on_overlap -q`
- [x] **S2-T07** [S] Unit test: `tests/test_sampling.py::test_bootstrap_draw_reproducible_with_seed`. **DoD:** `uv run pytest tests/test_sampling.py::test_bootstrap_draw_reproducible_with_seed -q`
- [x] **S2-T08** [F] Extend [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) **§11** with the full `sampling` YAML + `meta.json.sampling` contract (strategy options, `n_draw`, etc.). **DoD:** `grep -n "## 11" docs/DATA_SCHEMA.md`
- [x] **S2-T09** [F] Add one-line CLI example for sampling to [`README.md`](README.md) Quick commands table. **DoD:** `grep -n "sampling" README.md`
- [x] **S2-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 2)
- [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py), [`src/aprilalgo/ml/sampling.py`](src/aprilalgo/ml/sampling.py) (read-only), [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py) (via weights path), [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md), [`README.md`](README.md), [`tests/test_sampling.py`](tests/test_sampling.py), [`tests/test_cli_ml.py`](tests/test_cli_ml.py).

---

## Sprint 3 — Primary OOF capture

- [x] **S3-T01** [H] New module [`src/aprilalgo/ml/oof.py`](src/aprilalgo/ml/oof.py) exposing `compute_primary_oof(X, y, t0, t1, *, factory, n_splits, embargo) -> pd.DataFrame` (columns: `row_idx`, `y`, `oof_pred`, `oof_proba_<class>`). **DoD:** `uv run python -c "from aprilalgo.ml.oof import compute_primary_oof"`
- [x] **S3-T02** [S] Wire `compute_primary_oof` to use [`PurgedKFold`](src/aprilalgo/ml/cv.py) internally with the provided `factory`. **DoD:** `uv run pytest tests/test_oof.py::test_oof_shape -q`
- [x] **S3-T03** [S] Unit test `tests/test_oof.py::test_oof_shape` covers expected columns and index. **DoD:** `uv run pytest tests/test_oof.py::test_oof_shape -q`
- [x] **S3-T04** [S] Unit test `tests/test_oof.py::test_oof_no_all_nan_on_sufficient_data`. **DoD:** `uv run pytest tests/test_oof.py::test_oof_no_all_nan_on_sufficient_data -q`
- [x] **S3-T05** [S] Add `oof` CLI subcommand that writes `oof_primary.csv` next to the bundle under `model.out_dir`. **DoD:** `uv run python -m aprilalgo.cli oof --config configs/ml/default.yaml`
- [x] **S3-T06** [S] CLI smoke `tests/test_cli_ml.py::test_cli_oof_writes_csv`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_oof_writes_csv -q`
- [x] **S3-T07** [F] When `oof` runs it records a relative `oof.path` into the sibling `meta.json`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_oof_writes_csv -q`
- [x] **S3-T08** [F] Add **§12 OOF artifacts** to [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). **DoD:** `grep -n "## 12" docs/DATA_SCHEMA.md`
- [x] **S3-T09** [S] Determinism test: fixed `random_state` produces stable OOF. **DoD:** `uv run pytest tests/test_oof.py::test_oof_deterministic_with_seed -q`
- [x] **S3-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 3)
- New [`src/aprilalgo/ml/oof.py`](src/aprilalgo/ml/oof.py); [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) subcommand wiring; [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py) meta extension; new [`tests/test_oof.py`](tests/test_oof.py). Sprint 4 meta-label consumes this artifact.

---

## Sprint 4 — Meta-label bundle (persisted)

- [ ] **S4-T01** [H] New [`src/aprilalgo/ml/meta_bundle.py`](src/aprilalgo/ml/meta_bundle.py): `save_meta_logit_bundle(out_dir, clf, feature_names)` and `load_meta_logit_bundle(out_dir) -> MetaBundle`. Serialize `coef_`, `intercept_`, `classes_`, `feature_names`. **DoD:** `uv run python -c "from aprilalgo.ml.meta_bundle import save_meta_logit_bundle, load_meta_logit_bundle"`
- [ ] **S4-T02** [S] Define `MetaBundle` dataclass with `predict_proba(X)` using stored coefficients. **DoD:** `uv run pytest tests/test_meta_bundle.py::test_predict_proba_matches_sklearn -q`
- [ ] **S4-T03** [S] Unit test save → load roundtrip equivalence on probabilities. **DoD:** `uv run pytest tests/test_meta_bundle.py::test_predict_proba_matches_sklearn -q`
- [ ] **S4-T04** [S] CLI subcommand `meta-train` reads `oof_primary.csv` + features and calls `fit_meta_logit_purged`. **DoD:** `uv run python -m aprilalgo.cli meta-train --config configs/ml/default.yaml`
- [ ] **S4-T05** [S] `meta-train` writes `meta_logit.json` and `meta_oof.csv` into the bundle dir. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_meta_train_writes_artifacts -q`
- [ ] **S4-T06** [S] Update primary `meta.json` with `meta_logit: {enabled: true, path: "meta_logit.json"}`. **DoD:** `uv run python -c "import json; print(json.load(open('models/xgboost/latest/meta.json'))['meta_logit'])"`
- [ ] **S4-T07** [S] Unit test: without `oof_primary.csv` present, `meta-train` errors with a clear message. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_meta_train_requires_oof -q`
- [ ] **S4-T08** [F] Add **§13 Meta-label bundle** to [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). **DoD:** `grep -n "## 13" docs/DATA_SCHEMA.md`
- [ ] **S4-T09** [F] CHANGELOG Unreleased: `meta-train` subcommand + bundle schema. **DoD:** `grep -n "meta-train" CHANGELOG.md`
- [ ] **S4-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 4)
- New [`src/aprilalgo/ml/meta_bundle.py`](src/aprilalgo/ml/meta_bundle.py); [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) (new subcommand); [`src/aprilalgo/ml/trainer.py`](src/aprilalgo/ml/trainer.py) (`meta_logit` key); [`src/aprilalgo/labels/meta_label.py`](src/aprilalgo/labels/meta_label.py) (call site only); new [`tests/test_meta_bundle.py`](tests/test_meta_bundle.py); touches Sprint 5 strategy gate.

---

## Sprint 5 — `ml_xgboost` meta-gate (runtime)

- [ ] **S5-T01** [H] In [`MLStrategy.init`](src/aprilalgo/strategies/ml_strategy.py), if `bundle.meta.meta_logit.enabled` → load `MetaBundle` from the same bundle dir. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_meta_bundle_loaded_when_enabled -q`
- [ ] **S5-T02** [H] Compute `p_meta = MetaBundle.predict_proba(xrow)[:,1]` in `on_bar` when loaded. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_meta_proba_shape -q`
- [ ] **S5-T03** [H] Gate entries on `p_primary >= entry_proba_threshold AND p_meta >= meta_proba_threshold`. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_meta_gate_blocks_entry_below_threshold -q`
- [ ] **S5-T04** [S] Add `meta_proba_threshold: float = 0.5` ctor arg to `MLStrategy`. **DoD:** `uv run python -c "from aprilalgo.strategies.ml_strategy import MLStrategy; import inspect; assert 'meta_proba_threshold' in inspect.signature(MLStrategy.__init__).parameters"`
- [ ] **S5-T05** [S] Log `pred_proba_meta` in the JSONL event when the gate is active. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_meta_proba_logged_in_event -q`
- [ ] **S5-T06** [S] Update `docs/DATA_SCHEMA.md` §6 logger schema with `pred_proba_meta`. **DoD:** `grep -n "pred_proba_meta" docs/DATA_SCHEMA.md`
- [ ] **S5-T07** [S] Backward-compat test: legacy bundle without `meta_logit` key → gate disabled, no error. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_legacy_bundle_no_meta_gate -q`
- [ ] **S5-T08** [S] Integration test: full backtest with meta bundle ends with `metrics` dict. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_backtest_runs_with_meta_bundle -q`
- [ ] **S5-T09** [F] README row: "ML backtest with meta gate". **DoD:** `grep -n "meta gate" README.md`
- [ ] **S5-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 5)
- [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py); [`src/aprilalgo/backtest/logger.py`](src/aprilalgo/backtest/logger.py) (schema consumers); [`tests/test_ml_strategy.py`](tests/test_ml_strategy.py); reporting Sprint 10 reads the new logger field.

---

## Sprint 6 — Regime as a feature

- [ ] **S6-T01** [F] Add commented `regime:` block to [`configs/ml/default.yaml`](configs/ml/default.yaml) (`enabled`, `window`, `n_buckets`, `use_hmm`). **DoD:** `grep -n "regime:" configs/ml/default.yaml`
- [ ] **S6-T02** [H] In `_prepare_xy`, if `regime.enabled`, call `add_vol_regime` on the raw OHLCV **before** `build_feature_matrix`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_with_regime_enabled -q`
- [ ] **S6-T03** [H] Add `vol_regime` to the feature space **explicitly** (not excluded) but `realized_vol` **excluded** by default. **DoD:** `uv run pytest tests/test_features.py::test_regime_inclusion_rules -q`
- [ ] **S6-T04** [S] Persist `regime.*` params into `meta.json`. **DoD:** `uv run python -c "import json; print(json.load(open('models/xgboost/latest/meta.json'))['regime'])"`
- [ ] **S6-T05** [S] `_cfg_for_inference` merges `meta.regime` over config for `predict` / `shap`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_predict_regime_roundtrip -q`
- [ ] **S6-T06** [H] [`MLStrategy.init`](src/aprilalgo/strategies/ml_strategy.py) applies `add_vol_regime` if `bundle.meta.regime.enabled` before `IndicatorRegistry.apply`. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_regime_applied_in_strategy -q`
- [ ] **S6-T07** [S] Unit test: with `regime.enabled=false`, pipeline is bit-identical to 0.4.1 training. **DoD:** `uv run pytest tests/test_cli_ml.py::test_regime_off_matches_legacy -q`
- [ ] **S6-T08** [F] **§14 Regime conditioning** in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). **DoD:** `grep -n "## 14" docs/DATA_SCHEMA.md`
- [ ] **S6-T09** [F] AGENTS.md rule: "regime features must be computed with the same window as training". **DoD:** `grep -n "regime features" AGENTS.md`
- [ ] **S6-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 6)
- [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py), [`src/aprilalgo/ml/features.py`](src/aprilalgo/ml/features.py) (inclusion rules), [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py), [`src/aprilalgo/meta/regime.py`](src/aprilalgo/meta/regime.py) (no API change), [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). Sprint 7 consumes the regime column.

---

## Sprint 7 — Per-regime model routing

- [ ] **S7-T01** [F] YAML flag `regime.groupby: true` (new). **DoD:** `grep -n "groupby" configs/ml/default.yaml`
- [ ] **S7-T02** [H] `_train_and_save` splits `X/y` by `vol_regime` and writes one bundle per bucket into `out_dir/regime_<k>/`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_train_groupby_regime_writes_per_bucket -q`
- [ ] **S7-T03** [S] Write `out_dir/regime_index.json` = `{"buckets": {"0": "regime_0", ...}, "default": "regime_0"}`. **DoD:** `uv run python -c "import json, pathlib; print(json.loads(pathlib.Path('models/xgboost/latest/regime_index.json').read_text()))"`
- [ ] **S7-T04** [S] Each sub-bundle records `regime.bucket` in its own `meta.json`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_per_bucket_meta_has_bucket -q`
- [ ] **S7-T05** [H] `predict` subcommand supports `regime_index.json` — routes each row to its regime bundle and concatenates predictions. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_predict_regime_index_routing -q`
- [ ] **S7-T06** [H] [`MLStrategy`](src/aprilalgo/strategies/ml_strategy.py) loads `regime_index.json` if present and picks per-bar bundle. **DoD:** `uv run pytest tests/test_ml_strategy.py::test_strategy_routes_per_regime -q`
- [ ] **S7-T07** [S] Determinism test: same seed + same regimes → identical per-regime predictions. **DoD:** `uv run pytest tests/test_cli_ml.py::test_per_regime_predict_deterministic -q`
- [ ] **S7-T08** [F] Docs **§15 Per-regime routing** in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). **DoD:** `grep -n "## 15" docs/DATA_SCHEMA.md`
- [ ] **S7-T09** [F] README row: "Per-regime train / predict". **DoD:** `grep -n "Per-regime" README.md`
- [ ] **S7-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 7)
- [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) (train + predict), [`src/aprilalgo/strategies/ml_strategy.py`](src/aprilalgo/strategies/ml_strategy.py); new artifact `regime_index.json`. Reporting (Sprint 10) surfaces per-regime metrics; SHAP per-regime (Sprint 10) reads these sub-bundles.

---

## Sprint 8 — Purged walk-forward tuner core

- [ ] **S8-T01** [H] New [`src/aprilalgo/tuner/ml_walk_forward.py`](src/aprilalgo/tuner/ml_walk_forward.py) with public `ml_walk_forward_tune(cfg, grid, n_folds, metric) -> pd.DataFrame`. **DoD:** `uv run python -c "from aprilalgo.tuner.ml_walk_forward import ml_walk_forward_tune"`
- [ ] **S8-T02** [H] For each grid point: iterate `walk_forward_splits`; inside each train window run `purged_cv_evaluate`; append one row per `(grid_id, fold)`. **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_returns_one_row_per_grid_fold -q`
- [ ] **S8-T03** [S] Helper `aggregate_grid(results_df, metric) -> pd.DataFrame` with `mean`, `std`, `n_folds`. **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_aggregate_grid_columns -q`
- [ ] **S8-T04** [S] Helper `expand_grid(spec: dict) -> list[dict]` (cartesian product). **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_expand_grid_product -q`
- [ ] **S8-T05** [S] Metric registry (accuracy, f1_macro, neg_log_loss). **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_metric_registry_keys -q`
- [ ] **S8-T06** [S] Determinism test: fixed `random_state` reproduces results. **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_deterministic_with_seed -q`
- [ ] **S8-T07** [S] Error handling: empty grid raises `ValueError`. **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_empty_grid_raises -q`
- [ ] **S8-T08** [S] Sanity test: `n_folds=1` is allowed (degenerates to train/test split). **DoD:** `uv run pytest tests/test_ml_walk_forward.py::test_single_fold_runs -q`
- [ ] **S8-T09** [F] Docstrings + types on public functions. **DoD:** `uv run python -c "import aprilalgo.tuner.ml_walk_forward as m; assert m.ml_walk_forward_tune.__doc__"`
- [ ] **S8-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 8)
- New [`src/aprilalgo/tuner/ml_walk_forward.py`](src/aprilalgo/tuner/ml_walk_forward.py); reuses [`src/aprilalgo/ml/cv.py`](src/aprilalgo/ml/cv.py), [`src/aprilalgo/tuner/walk_forward.py`](src/aprilalgo/tuner/walk_forward.py), [`src/aprilalgo/ml/evaluator.py`](src/aprilalgo/ml/evaluator.py); new tests file. Sprint 9 wires this into CLI + Streamlit.

---

## Sprint 9 — WF tuner CLI + Streamlit

- [ ] **S9-T01** [S] CLI `wf-tune` subcommand consumes `wf_tuner.grid` / `wf_tuner.metric` from YAML. **DoD:** `uv run python -m aprilalgo.cli wf-tune --config configs/ml/default.yaml`
- [ ] **S9-T02** [S] `wf-tune` writes `wf_tune_results.csv` under `model.out_dir`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_wf_tune_writes_csv -q`
- [ ] **S9-T03** [S] CLI prints top-5 grid rows by aggregated metric. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_wf_tune_prints_top5 -q`
- [ ] **S9-T04** [F] Config sample `wf_tuner:` block (grid, metric) in `configs/ml/default.yaml`. **DoD:** `grep -n "wf_tuner:" configs/ml/default.yaml`
- [ ] **S9-T05** [S] Streamlit: new sub-tab in [`src/aprilalgo/ui/pages/walk_forward_lab.py`](src/aprilalgo/ui/pages/walk_forward_lab.py) called "Tuner". **DoD:** `uv run pytest tests/test_streamlit_smoke.py::test_streamlit_pages_importable -q`
- [ ] **S9-T06** [S] Tuner tab: file uploader or auto-discover `wf_tune_results.csv`. **DoD:** `uv run pytest tests/test_streamlit_smoke.py::test_streamlit_pages_importable -q`
- [ ] **S9-T07** [S] Tuner tab: plotly box/violin of per-fold metric by `grid_id`. **DoD:** `uv run pytest tests/test_streamlit_smoke.py::test_streamlit_pages_importable -q`
- [ ] **S9-T08** [F] README row: "Walk-forward ML tuner". **DoD:** `grep -n "wf-tune" README.md`
- [ ] **S9-T09** [F] CHANGELOG Unreleased: wf-tune CLI + Streamlit Tuner tab. **DoD:** `grep -n "wf-tune" CHANGELOG.md`
- [ ] **S9-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 9)
- [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py) (new subcommand), [`src/aprilalgo/ui/pages/walk_forward_lab.py`](src/aprilalgo/ui/pages/walk_forward_lab.py), [`configs/ml/default.yaml`](configs/ml/default.yaml), [`README.md`](README.md), [`CHANGELOG.md`](CHANGELOG.md).

---

## Sprint 10 — Reporting + SHAP ties

- [ ] **S10-T01** [S] `reporting/report.py`: `render_sampling_section(...)` emits section with id `section-sampling` (uniqueness histogram when weights present). **DoD:** `uv run pytest tests/test_reporting.py::test_sampling_section_id -q`
- [ ] **S10-T02** [S] `render_meta_section(...)` — id `section-meta` (coefficients + OOF coverage). **DoD:** `uv run pytest tests/test_reporting.py::test_meta_section_id -q`
- [ ] **S10-T03** [S] `render_regime_section(...)` — id `section-regime` (bucket counts + per-regime accuracy). **DoD:** `uv run pytest tests/test_reporting.py::test_regime_section_id -q`
- [ ] **S10-T04** [S] `render_wf_tuner_section(...)` — id `section-wf-tuner` (top-N grid + per-fold dispersion). **DoD:** `uv run pytest tests/test_reporting.py::test_wf_tuner_section_id -q`
- [ ] **S10-T05** [S] New helper `shap_values_per_regime(bundle_index, X_by_regime, ...)` in [`src/aprilalgo/ml/explain.py`](src/aprilalgo/ml/explain.py). **DoD:** `uv run pytest tests/test_shap.py::test_per_regime_shape -q`
- [ ] **S10-T06** [S] CLI: `shap --per-regime` flag writes `regime_<k>_shap_values.csv` + `regime_<k>_shap_importance.csv`. **DoD:** `uv run pytest tests/test_cli_ml.py::test_cli_shap_per_regime -q`
- [ ] **S10-T07** [S] Test `tests/test_reporting.py::test_render_full_html_sections_when_all_artifacts_present`. **DoD:** `uv run pytest tests/test_reporting.py::test_render_full_html_sections_when_all_artifacts_present -q`
- [ ] **S10-T08** [F] **§16 Per-regime SHAP artifacts** in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). **DoD:** `grep -n "## 16" docs/DATA_SCHEMA.md`
- [ ] **S10-T09** [F] CHANGELOG Unreleased: new report sections + per-regime SHAP. **DoD:** `grep -n "section-wf-tuner" CHANGELOG.md`
- [ ] **S10-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 10)
- [`src/aprilalgo/reporting/report.py`](src/aprilalgo/reporting/report.py), [`src/aprilalgo/ml/explain.py`](src/aprilalgo/ml/explain.py), [`src/aprilalgo/cli.py`](src/aprilalgo/cli.py); reads Sprint 1–7 artifacts only.

---

## Sprint 11 — Documentation synchronization

- [ ] **S11-T01** [F] [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md): verify §11 Sampling (Sprint 2) complete. **DoD:** `grep -c "### 11" docs/DATA_SCHEMA.md`
- [ ] **S11-T02** [F] [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md): verify §12 OOF, §13 Meta-label, §14 Regime, §15 Per-regime, §16 Per-regime SHAP. **DoD:** `for s in 12 13 14 15 16; do grep -c "## $s" docs/DATA_SCHEMA.md; done`
- [ ] **S11-T03** [S] Replace the v0.5 placeholder in [`ARCHITECTURE.md`](ARCHITECTURE.md) §2 with the full v0.5 roadmap (sampling / meta / regime / wf tuner). **DoD:** `grep -n "v0.5" ARCHITECTURE.md`
- [ ] **S11-T04** [F] [`CHANGELOG.md`](CHANGELOG.md) Unreleased entries grouped by Added / Changed / Dependencies. **DoD:** `grep -c "Unreleased" CHANGELOG.md`
- [ ] **S11-T05** [F] [`README.md`](README.md) Quick commands table rows: `oof`, `meta-train`, `wf-tune`, `shap --per-regime`. **DoD:** `grep -c "wf-tune" README.md`
- [ ] **S11-T06** [F] [`CLAUDE.md`](CLAUDE.md) bumps ML CLI verb list + mentions regime / meta gate. **DoD:** `grep -n "meta-train" CLAUDE.md`
- [ ] **S11-T07** [F] [`AGENTS.md`](AGENTS.md) adds rule: "training must not use regime computed with a different window than the model was trained with". **DoD:** `grep -n "regime features" AGENTS.md`
- [ ] **S11-T08** [F] [`docs/MODEL_ROUTING.md`](docs/MODEL_ROUTING.md): add a v0.5 sprint routing table (F/S/H per sprint). **DoD:** `grep -n "v0.5" docs/MODEL_ROUTING.md`
- [ ] **S11-T09** [F] Drop the "v0.2" bootstrapping lines in [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) interior (below the Today block). **DoD:** `! grep -n "v0.2 with 11 indicators" docs/SESSION_HANDOFF.md`
- [ ] **S11-T10** [F] Regenerate [`tests.md`](tests.md). **DoD:** `uv run pytest tests/ --collect-only -q`

### Downstream dependencies (Sprint 11)
- Docs only: [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`README.md`](README.md), [`CHANGELOG.md`](CHANGELOG.md), [`CLAUDE.md`](CLAUDE.md), [`AGENTS.md`](AGENTS.md), [`docs/MODEL_ROUTING.md`](docs/MODEL_ROUTING.md), [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md).

---

## Sprint 12 — Release prep + hygiene

- [ ] **S12-T01** [F] Run full pytest and capture baseline count. **DoD:** `uv run pytest tests/ -q | tail -n 3`
- [ ] **S12-T02** [F] Bump [`pyproject.toml`](pyproject.toml) `version = "0.5.0"`. **DoD:** `grep -n "^version" pyproject.toml`
- [ ] **S12-T03** [F] Bump `__version__` in [`src/aprilalgo/__init__.py`](src/aprilalgo/__init__.py). **DoD:** `grep -n "__version__" src/aprilalgo/__init__.py`
- [ ] **S12-T04** [F] Promote [`CHANGELOG.md`](CHANGELOG.md) Unreleased body into a new `## [0.5.0] - <date>` block; leave `## [Unreleased]` empty. **DoD:** `grep -n "## \[0.5.0\]" CHANGELOG.md`
- [ ] **S12-T05** [F] Update [`CLAUDE.md`](CLAUDE.md) Current version to `0.5.0`. **DoD:** `grep -n "Current version" CLAUDE.md`
- [ ] **S12-T06** [F] Final regeneration of [`tests.md`](tests.md) with new count. **DoD:** `head -n 5 tests.md`
- [ ] **S12-T07** [F] Update `Today's Handoff` date in [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md); copy Sprint 12 handoff block from the template. **DoD:** `grep -n "Today's Handoff" docs/SESSION_HANDOFF.md`
- [ ] **S12-T08** [F] Mark all 120 BACKLOG boxes completed; move completed sprints to a `## Done (v0.5.0)` appendix section. **DoD:** `! grep -n "- \[ \] S" BACKLOG.md`
- [ ] **S12-T09** [F] Clean stale "v0.2" references from [`docs/REPO_ANALYSIS.md`](docs/REPO_ANALYSIS.md) or append dated "historical" note. **DoD:** `grep -n "historical" docs/REPO_ANALYSIS.md`
- [ ] **S12-T10** [F] Final full test run. **DoD:** `uv run pytest tests/ -q`

### Downstream dependencies (Sprint 12)
- [`pyproject.toml`](pyproject.toml), [`src/aprilalgo/__init__.py`](src/aprilalgo/__init__.py), [`CHANGELOG.md`](CHANGELOG.md), [`CLAUDE.md`](CLAUDE.md), [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md), [`BACKLOG.md`](BACKLOG.md), [`tests.md`](tests.md).

---

## Done (v0.5.0)

_Completed sprints will be moved here by Sprint 12-T08._
