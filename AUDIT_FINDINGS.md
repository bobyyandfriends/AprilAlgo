# AprilAlgo — Principal-Staff Audit Findings

Date: 2026-04-18
Scope: full `src/aprilalgo/**/*.py` static analysis (78 modules). No runtime profiling or
stress testing was performed. Findings were produced by three parallel read-only audit
passes (backtest/strategies/confluence; indicators/data/UI; ML/labels/meta/tuner/reporting)
and cross-checked against the current test suite (183 passed, 1 skipped after the fixes
below).

This document captures (A) the bugs fixed in this audit, and (B) the architectural risks
and design concerns that were intentionally *not* fixed because they require product or
architecture decisions.

---

## A. Bugs fixed in this pass

Every item below was a definitive correctness defect. Each has been repaired in place; the
full pytest suite continues to pass.

### A1. `backtest/portfolio.py::record_equity` — short-position equity double-count (CRITICAL)

Prior code credited shorts with the full `entry_price * quantity` at open and never
subtracted that liability when snapshotting equity, so every open short inflated reported
equity by its notional value for the lifetime of the position. Rewritten the equity
snapshot to subtract the short liability and add the long entry cost back, so the result
equals `cash + MTM` for both sides consistently.

### A2. `backtest/portfolio.py::close_trade` — exit commission & slippage missing from `Trade.realized_pnl` (HIGH)

Exit-side `slip`/`comm` were applied to `Portfolio.cash` but never accumulated onto the
`Trade` before `Trade.close` computed `realized_pnl`. `metrics.py` derives every summary
stat from `trade.realized_pnl`, so reports were systematically better than true cash
balance. Now both exit-side costs are added to `trade.commission` / `trade.slippage`
before `Trade.close` is invoked. Also switched `open_positions.remove(trade)` to an
identity-match loop to avoid `dataclass.__eq__` collisions.

### A3. `indicators/pv_sequences.py` + `indicators/tmi.py` — empty-DataFrame crash (CRITICAL)

Both indicators called `price_up[0] = False` / `prev[0] = 0.0` on a zero-length array.
Any caller passing an empty frame (e.g. a symbol with zero rows after `dropna`, the Signal
Feed page during warmup, or an unavailable fixture) would crash with `IndexError`. Added
early-return guards that return the empty DataFrame with the expected indicator columns
correctly-typed.

### A4. `indicators/demark.py` — signal re-fire + countdown reset bugs (HIGH, two related)

Two coupled defects:

1. `if buy_setup[i] == 9:` unconditionally reset `buy_cd_count = 0` **every bar** the
   setup stayed at 9, so the countdown could never reach 13 unless the setup broke on
   exactly the tenth bar.
2. The "weaker completed-setup" signal `td_bull[i] = True` fired on every bar the setup
   remained at 9, not only on the completion transition, so downstream confluence /
   strategy consumers saw a standing flag instead of a discrete event.

Both are now gated on the transition `buy_setup[i-1] < 9 and buy_setup[i] == 9`, so
countdown activation is idempotent and the weak signal fires exactly once per completion.

### A5. `ml/oof.py::compute_primary_oof` — multiclass OOF column misalignment (CRITICAL, ML correctness)

The OOF probability matrix was sized from the **first** fold's `est.classes_`. For
`task="multiclass"` with `y ∈ {-1, 0, 1}`, a fold whose training block excluded a class
either crashes with a broadcast error or — worse — silently shifts the column → class
mapping, so downstream meta-labeling and sizing saw wrong probabilities.

Replaced with a global class axis computed from `np.unique(y)` up-front; each fold's
`est.classes_` is now mapped onto the global axis, and classes missing from a fold's
training set stay NaN for that fold's test rows.

### A6. `labels/meta_label.py::build_meta_labels` + `fit_meta_logit_purged` — NaN-pred rows bias the meta dataset (CRITICAL, ML correctness)

`build_meta_labels` used `oof_primary_pred == y_true`, and `NaN == anything` evaluates to
`False`. Every OOF row with a NaN primary prediction (legitimately produced when a
PurgedKFold train block was emptied by purging) was therefore labeled `0 = "primary
incorrect"`, biasing the meta model toward the positive class on every uncertain bar.
Additionally, `LogisticRegression.fit(X_meta, z)` in `fit_meta_logit_purged` would also
have raised on rows whose `primary_pred` column was NaN.

Refactored `build_meta_labels` to return `np.float64` with NaN for undefined rows.
`fit_meta_logit_purged` now restricts both the inner CV and the final fit to the
valid-mask, and re-expands `meta_oof` / `z` to the full length with NaN for the excluded
rows.

### A7. `ml/explain.py::_shap_matrix` — signed multiclass aggregation (HIGH)

The older-API list branch did `np.stack([...]).mean(axis=0)` **before** the absolute
value, so a feature with `+0.3` for class 1 and `-0.3` for class -1 cancelled to `0` and
was ranked as unimportant by `shap_importance_table`. The `ndim == 3` branch was already
correct. Moved `np.abs` inside the list-branch aggregation.

### A8. `tuner/ml_walk_forward.py::ml_walk_forward_tune` — `json.loads(json.dumps(cfg))` crash on non-JSON types (HIGH)

`cfg` is yaml-loaded and may contain `datetime`, `Path`, `tuple`, or numpy scalars;
`json.dumps` raises `TypeError` for those. Replaced with `copy.deepcopy(cfg)`.

### A9. `tuner/analyzer.py::_check_robustness` — degradation formula inverted for negative metrics (HIGH)

`degradation = 1 - neighbor_mean / best_metric` flips sign when `best_metric` is
negative: a better-performing neighbor of a negative-Sharpe "best" was reported as a
positive degradation. Changed to `(best - neighbor) / abs(best)` so robustness semantics
are preserved across the sign of the metric.

### A10. `ml/evaluator.py` — bare `except` hides real errors; `predict_proba[:, 1]` crashes on single-class train (HIGH)

The binary path wrapped `est.predict_proba(X_te)[:, 1]` plus the ROC/log-loss calls in
`except Exception: ...`, which swallowed every failure mode uniformly — including real
bugs in downstream metric code — and still crashed on `IndexError` when a fold's training
block held only one class. Replaced with an explicit `len(np.unique(y_tr)) < 2` guard and
a narrow `except (ValueError, IndexError):` so genuine bugs surface instead of silently
becoming `None` metrics.

### A11. `ui/helpers.py::format_metric` — crash on `None` / `NaN` values (HIGH impact UI)

`f"{None:.2f}"` and `int("N/A")` raised `TypeError` / `ValueError` respectively. Tuner
and Dashboard pages pass `best[k]` where `k` can be NaN for errored combos. Added a
leading None / NaN guard that returns `"—"`, and wrapped the numeric format fallbacks in
a `(TypeError, ValueError)` handler so we never surface a raw traceback in the UI.

### A12. `ui/pages/signals.py` — `int(NaN)` crash on warm-up rows (HIGH)

`int(row.get('bull_count', 0))` only handles missing *keys*; NaN values from indicator
warm-ups raised `ValueError`. Added a `_safe_int` helper and a `pd.isna` guard on
`datetime` / `close` / counts; if a bar is un-renderable the loop skips it cleanly.

### A13. `ui/pages/model_metrics.py` — double `clf.fit` + unguarded importance CSV (HIGH perf / MEDIUM crash)

The binary branch fit the XGBClassifier twice (once for ROC, once for proxy equity) on
the same `(X, y)`. Consolidated into one fit, added an explicit "in-sample, not
out-of-sample" caption to prevent misreading, and wrapped the `importance_gain.csv`
bar-chart path in a column-schema check so a malformed CSV doesn't crash the page.

### A14. `reporting/report.py::render_backtest_html` — missing Jinja autoescape + shadowed `html` import (MEDIUM)

`Template(_HTML)` does not autoescape by default; any `<` in a column name, note, or
metric name would break the rendered document. Additionally, the local variable `html =
Template(...).render(...)` shadowed the stdlib `html` module imported at the top of the
file, leaving a landmine for any future edit that calls `html.escape`. Switched to
`Template(..., autoescape=select_autoescape(["html","xml"]))` and renamed the local to
`rendered`.

### Test verification

After the fixes: **183 passed, 1 skipped**. No new lint errors were introduced on the
edited files.

---

## B. Architectural risks & design concerns — STATUS

> **Status update (2026-04-18, session 2):** All 23 items below (B1–B23) have been addressed. See `CHANGELOG.md` `[Unreleased]` Fixed block and `PROJECT_STATE.md` §2 handoff for the per-item diff. This section is retained for historical reference; treat it as "done" rather than "backlog".

### Resolved (session 2, 2026-04-18)

Each item below is a legitimate concern that was identified but **intentionally not
altered** because it requires a product decision, an API contract change, or a
cross-cutting refactor that should not land silently inside a bug-hunt audit. Issues
are grouped by theme and ranked by impact.

### B1. Short-sale modeling (backtest/portfolio.py) — impact: HIGH

Even after fixing the equity-snapshot bug (§A1), the short model is incomplete:

* No margin requirement — a strategy can open unbounded short notional with zero equity.
* No borrow fee / stock-loan cost.
* Symmetric `price * quantity` proceeds / cover on a naked short. Real brokers hold
  proceeds as collateral and charge interest on the borrow.

**Recommendation:** add optional `margin_ratio` and `borrow_rate_bps_per_day` parameters
to `Portfolio`. Until then, short-heavy backtests should carry an explicit disclaimer.

### B2. `initial_capital` as denominator (backtest/metrics.py) — impact: HIGH

`trade_returns = pnl / initial_capital` ignores compounding — returns computed on the
realized-P&L basis mis-state both Sharpe (denominator volatility) and simple return
accuracy on long backtests. Additionally, per-trade returns are annualized with
`sqrt(252)` regardless of trade frequency, which can overstate Sharpe by an order of
magnitude for low-turnover strategies.

**Recommendation:** compute returns from the daily (or bar-level) equity curve and
annualize against the actual sample frequency. Break this into a separate `metrics_v2`
to avoid silently shifting existing reports.

### B3. Strategy/engine loop-frame contract is unenforced (backtest/engine.py) — impact: HIGH

Only `MLStrategy` sets `_backtest_bars_df`; `ConfigurableStrategy`, `DeMarkConfluenceStrategy`,
and `RsiSmaStrategy` implicitly trust that the indicator pipeline preserves row count.
Any future indicator that drops warm-up rows breaks these strategies silently, with
`IndexError` deep inside `on_bar`.

**Recommendation:** standardize a `strategy._backtest_bars_df` contract in
`BaseStrategy.init`, and assert length equality against `strategy._data` (or the
enriched frame) before entering the loop in `engine.run_backtest`.

### B4. Sample-weight plumbing is inconsistent (ml/**) — impact: HIGH

* `compute_primary_oof` accepts `sample_weight` and forwards it to `est.fit`.
* `purged_cv_evaluate` has no `sample_weight` parameter.
* `ml_walk_forward_tune` calls `purged_cv_evaluate` — so tuner hyperparameter selection
  runs unweighted even when the production model is weighted.

**Consequence:** the tuner selects hyperparameters against one distribution and
production trains against another. Silent reproducibility gap.

**Recommendation:** thread `sample_weight` through `purged_cv_evaluate` and through
`ml_walk_forward_tune`.

### B5. Purged-CV embargo is one-sided (ml/cv.py) — impact: MEDIUM, needs decision

`_embargo_train` only drops training samples whose `t0` lies just after the test block's
`max_t1`. Many implementations of AFML §7 apply a symmetric embargo (also drop training
rows whose `t1` sits just before `min(t0[test_idx])`) to cover residual feature-level
serial correlation. This is an intentional design choice — document it or add it.

### B6. `PurgedKFold._purge_train` + `fold_train_test_interval_disjoint` are O(|train|·|test|) (ml/cv.py, ml/evaluator.py) — impact: MEDIUM perf

Both inner loops are pure-Python and quadratic. For `n=10k`, one fold iterates ~16M
times. Not a correctness issue but demonstrably slow under tuner loops. Vectorize with
broadcasting similar to `ml/sampling.py::overlap_count_matrix`.

### B7. `ModelBundle.predict_proba` binary path assumes class-0/class-1 column order (ml/trainer.py) — impact: MEDIUM, latent

`np.column_stack([1-p, p])` implicitly assumes `bundle.classes_ == [0., 1.]`. Today
`save_model_bundle` writes sorted classes, so this holds, but the invariant is
undocumented and a future `meta.json` with `[1.0, 0.0]` would silently return swapped
probabilities. Anchor the ordering on `classes_`.

### B8. `save_model_bundle` multiclass fallback via raw `clf.classes_` (ml/trainer.py) — impact: MEDIUM

For `multi:softprob`, a pre-fit classifier not produced by `train_xgb_classifier` has
`clf.classes_ == [0..k-1]` (xgboost ≥ 2.0), which is the *encoded* label space, not the
original `{-1, 0, 1}`. The fallback would then silently produce wrong take-profit
probabilities downstream. Either assert the labels or raise in that branch.

### B9. `walk_forward_splits` does not honor `n_folds` (tuner/walk_forward.py) — impact: MEDIUM, API ambiguity

`test_size` is derived from `n_folds` and then the loop emits as many splits as fit in
the remaining range. For many inputs this produces `n_folds + 1` splits. Either break
after emitting exactly `n_folds` or rename the parameter.

### B10. Circular-ish `cli.py ⇔ tuner/ml_walk_forward.py` (tuner/ml_walk_forward.py) — impact: MEDIUM

`ml_walk_forward.py` lazy-imports `_prepare_xy` and `_xgb_estimator_factory` from
`aprilalgo.cli`. Leading underscore is intentional private API, and any unrelated import
failure in `cli.py` (new handler pulling in a heavy optional dep) breaks the tuner.

**Recommendation:** relocate `_prepare_xy` and `_xgb_estimator_factory` to a new neutral
module (e.g. `aprilalgo/ml/pipeline.py`) and have both `cli.py` and the tuner import
from there.

### B11. Binary labeling conflates "stop-loss hit" with "vertical timeout" (labels/targets.py) — impact: MEDIUM, product decision

`label_binary` maps `LABEL_TAKE_PROFIT → 1`, everything else (stop-loss, timeout) → 0.
A strategy thresholding `P(class=1)` to decide "enter long" cannot distinguish a
"neutral, uncertain" market from "active downside". Consider leaving the primary as
multiclass (`{-1, 0, 1}`) and letting the strategy consume the TP-class probability only.

### B12. `confluence/scorer.py` auto-detection breaks on string signal columns (MEDIUM)

`out[bull_cols].astype(float)` raises `ValueError` if an indicator author names a string
column like `signal_bull`. Either filter by `dtype in {bool, numeric}` or use
`pd.to_numeric(..., errors="coerce")`.

### B13. `confluence/timeframe_aligner.py` assumes unique, sorted base index (MEDIUM)

`out.join(hdf, how="left")` on non-unique timestamps produces a cartesian blow-up, and
the forward-fill after join only works if `out.index` is sorted. Neither is enforced.

### B14. `meta/regime.py` — realized-vol + HMM edge cases (MEDIUM)

* `realized_vol(..., min_periods=1)` produces meaningless std for the first 1–2 rows.
* `np.log(close).diff().fillna(0.0)` lets `-inf` from `log(0)` survive; `GaussianHMM.fit`
  errors opaquely on bad input.
* `pd.qcut(..., duplicates="drop")` can yield fewer buckets than requested; downstream
  code that looks up `bundles[str(i)] for i in range(n_buckets)` would KeyError.

### B15. RSI uses SMA, not Wilder's EMA (indicators/rsi.py) — impact: MEDIUM, needs confirmation

The implementation uses `.rolling(window).mean()` rather than Wilder's
`ewm(alpha=1/period, adjust=False)`. This is a known variant but produces meaningfully
different values. Confirm strategy docstrings and tuner expectations align with the SMA
form — otherwise switch to Wilder.

### B16. `indicators/hurst.py` — `hurst_bear` docstring vs implementation mismatch (MEDIUM)

Docstring: "True when majority of windows show mean-reverting (H < threshold)".
Code: `hurst_bear = trending > majority & ~price_rising`. The confluence scorer counts
this as bearish, but the semantics differ from the docstring. Decide which is correct
and align the other.

### B17. Ehlers recursive IIR filters propagate NaN forever (indicators/ehlers.py) — impact: MEDIUM

`super_smoother`, `roofing_filter`, `decycler` reference `result[i-1]` — any NaN in the
source column (e.g. after chaining a second indicator upstream, or after resampling)
permanently poisons the tail. Forward-fill the source NaNs before the loop or seed
`result[i] = src[i]` on NaN hits.

### B18. `ml/features.py::align_features_and_labels` silently truncates long warm-ups (MEDIUM diagnostic)

`combined.dropna()` can drop tens of percent of rows with no log line. For a dataset
with a 200-bar indicator and 500 bars total, 40% of data vanishes without diagnostics.
Add a row-count delta log.

### B19. `reporting/report.py` renders the first 500 equity points without a visible truncation note (LOW)

`equity.head(500).iterrows()` silently truncates. Add a small footer showing total rows
and the truncation window.

### B20. `ui/pages/tuner.py` — hover_data passes metric names that may not be columns (MEDIUM)

`hover_data=list(METRIC_DISPLAY_NAMES.keys())` passes all metric keys to Plotly even if
some aren't in `results_df`; Plotly raises. Intersect with `results_df.columns` first.
`_check_robustness`'s fix in §A9 does not cover this page's hover-data issue.

### B21. `ui/pages/walk_forward_lab.py` swallows all exceptions (MEDIUM)

`except Exception: pass` hides schema drift, missing columns, and pandas errors
indistinguishably. Narrow to `json.JSONDecodeError` and surface the rest via
`st.warning(str(e))`.

### B22. `ui/pages/model_lab.py` / `model_trainer.py` — unbounded `subprocess.run` (MEDIUM ops)

`capture_output=True` buffers all stdout/stderr in memory, and there's no timeout. A
long or hung CLI hangs the Streamlit worker and can OOM it.

### B23. `logger.hash_features_row` and `_prepare_xy` import leaks across layers (LOW)

* `backtest/logger.py::hash_features_row` does not guard against empty frames or mixed
  dtypes — both cause cryptic errors before reaching the hash.
* `ui/pages/model_metrics.py` and `tuner/ml_walk_forward.py` both import the private
  `_prepare_xy` from `cli.py`. See §B10.

---

## C. Severity roll-up

|                 | Fixed (§A) | Unfixed (§B) |
|-----------------|-----------:|-------------:|
| CRITICAL        | 4          | 0            |
| HIGH            | 9          | 4            |
| MEDIUM          | 1          | 15           |
| LOW             | 0          | 4            |
| **Total**       | **14**     | **23**       |

Plus ~50 lower-severity items captured in the raw audit transcripts that were rolled
into the categories above.

---

## D. Recommended follow-up order

If these unfixed items are tackled later, the highest-leverage order is:

1. **B3 (strategy loop-frame contract)** — eliminates a whole class of `IndexError` /
   misalignment bugs latently lurking in three strategies.
2. **B4 (sample-weight plumbing)** — silently skews hyperparameter selection today.
3. **B10 (relocate `_prepare_xy` out of `cli`)** — trivial refactor, unblocks the ML
   pipeline from CLI coupling.
4. **B2 (compound-equity returns)** — rewrites the performance story in `metrics_v2`.
5. **B1 (short-sale margin / borrow fee)** — necessary before trusting any short-heavy
   report.
6. **B15 / B16 (RSI & Hurst semantics)** — resolve the docstring-vs-code and Wilder-vs-SMA
   questions so strategies inherit predictable indicator behavior.
7. The remaining MEDIUM items can be batched into a single cleanup PR.
