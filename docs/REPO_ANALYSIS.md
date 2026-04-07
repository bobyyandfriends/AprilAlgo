# External Repository Analysis for AprilAlgo

> **Generated:** March 29, 2026  
> **Purpose:** Catalog which GitHub repos have useful components for AprilAlgo, what to use, and when to use it in the roadmap.

---

## Quick Reference Table

| Repo | License | Language | Usable? | When to Use | Priority |
|------|---------|----------|---------|-------------|----------|
| [massive-com/client-python](https://github.com/massive-com/client-python) | MIT | Python | **YES — direct dependency** | v0.2 (data fetcher update) | **HIGH** |
| [shap/shap](https://github.com/shap/shap) | MIT | Python | **YES — direct dependency** | v0.3 (ML + explainability) | **HIGH** |
| [aticio/legitindicators](https://github.com/aticio/legitindicators) | MIT | Python | **Reference only** — port formulas, don't depend | v0.2 (Hurst, Ehlers) | **HIGH** |
| [Rachnog/Deep-Trading](https://github.com/Rachnog/Deep-Trading) | None listed | Python | **Learn from** — educational notebooks | v0.3 (ML pipeline) | **MEDIUM** |
| [bobyyandfriends/StockTradingAI](https://github.com/bobyyandfriends/StockTradingAI) | None listed | Python | **Learn from** — DQN concepts | v0.4+ (RL exploration) | **LOW** |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | **All Rights Reserved** | Python | **CANNOT use code** — concepts only | v0.3 (labeling, CV) | **MEDIUM** |
| [StockSharp/StockSharp](https://github.com/StockSharp/StockSharp) | Apache-2.0 | **C#** | **Architecture reference only** | Architecture design | **LOW** |
| [joelowj/awesome-algorithmic-trading](https://github.com/joelowj/awesome-algorithmic-trading) | CC-BY-4.0 | N/A | **Learning resources list** | Ongoing education | **LOW** |

---

## 1. massive-com/client-python (formerly Polygon.io)

**What it is:** Official Python client for the Massive.com REST and WebSocket API (OHLCV bars, trades, quotes for stocks, options, crypto).

**Install:** `pip install -U massive` (Python 3.9+)

**Key API for AprilAlgo:**
```python
from massive import RESTClient

client = RESTClient(api_key="YOUR_KEY")  # or env var MASSIVE_API_KEY
bars = []
for bar in client.list_aggs("AAPL", 1, "day", "2024-01-01", "2024-12-31", limit=50000):
    bars.append(bar)
```

**Timeframe patterns:**
- 1-min: `multiplier=1, timespan="minute"`
- 5-min: `multiplier=5, timespan="minute"`
- Daily: `multiplier=1, timespan="day"`
- Weekly: `multiplier=1, timespan="week"`

**Important notes:**
- `from_` parameter uses trailing underscore (Python reserved word workaround)
- Pagination is automatic by default (`pagination=True`); `limit` controls page size, not total
- Use max supported `limit` (50000 for aggs) to reduce API calls
- Free tier has rate limits — check [massive.com/pricing](https://massive.com/pricing)
- `api.polygon.io` still works during transition period
- Debug mode: `RESTClient(trace=True, verbose=True)`

**When to use:** v0.2 — replace the old Polygon.io fetcher script with `massive` package

---

## 2. shap/shap — SHAP Explainability

**What it is:** Game-theoretic approach to explain ML model outputs. 25.2k stars, MIT license.

**Key components for AprilAlgo:**

| Component | Use Case |
|-----------|----------|
| `TreeExplainer` (via `shap.Explainer`) | Fast exact SHAP for XGBoost — primary explainer |
| Waterfall plots | Single trade explanation: "DeMark drove +0.4" |
| Force plots | Alternative visual for individual signal explanations |
| Beeswarm plots | Dataset-level: which features matter overall |
| `shap.plots.bar()` | Global feature importance ranking |
| `shap_interaction_values` | Optional: pairwise feature interactions |

**Integration pattern:**
```python
import xgboost, shap

model = xgboost.XGBClassifier().fit(X_train, y_train)
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Single trade narrative
shap.plots.waterfall(shap_values[0])

# For text output: sort by abs SHAP, format top-k with signs
```

**Gotchas for trading:**
- SHAP explains the *model*, not *causality* — "DeMark drove the score" means the model used it, not that DeMark causes returns
- Background/reference distribution matters — align with training window
- Correlated features split credit (Shapley-style)
- Standard tabular SHAP treats rows as independent — your *features* should encode time structure
- Explaining every bar is expensive — batch or only explain signals

**When to use:** v0.3 — alongside XGBoost signal classifier

---

## 3. aticio/legitindicators — Technical Indicators Reference

**What it is:** Collection of 33+ trading indicators in a single Python file. MIT license, `pip install legitindicators`.

**Indicators directly useful for AprilAlgo (not already implemented):**

### HIGH PRIORITY (Hurst + Ehlers cycle stack):
1. **`hurst_coefficient`** — fractal dimension + super_smoother; validate vs textbook R/S or DFA Hurst
2. **`super_smoother`** — foundation for many other Ehlers indicators
3. **`roofing_filter`** — cycle/trend separation
4. **`decycler` / `decycler_oscillator`** — trend extraction
5. **`high_pass_filter`** — cycle isolation
6. **`ebsw`** (Even Better Sine Wave) — phase/cycle signals
7. **`voss`** — cycle detection
8. **`trendflex` / `custom_trendflex`** — trend-following oscillator

### MEDIUM PRIORITY (regime/trend context):
9. **`kaufman_er`** — efficiency ratio (trend quality)
10. **`kama`** — Kaufman Adaptive Moving Average
11. **`damiani_volatmeter`** — volatility regime
12. **`atr` / `smoothed_atr`** — volatility measurement
13. **`szladx`** — low-lag ADX upgrade
14. **`linreg_curve` / `linreg_slope`** — linear regression indicators
15. **`volume_heat`** — binary volume spike flag

### ALREADY COVERED: `sma`, `ema`, `bollinger_bands_pb` (overlap with existing AprilAlgo indicators)

**Quality assessment: PROTOTYPE — do NOT depend directly. Issues found:**
- `vwap()` has stray `print()` side effect
- `linreg_slope()` mutates caller's data via `.pop()` and `.insert()`
- `ema()` seeds with `1` (non-standard warmup)
- Pure Python loops over lists (slow for large datasets)
- `scipy` may be undeclared runtime dependency

**Recommendation:** Use as *reference implementations* to sanity-check our own NumPy/Pandas vectorized versions. Port the math, not the code.

**When to use:** v0.2 — reference for Hurst, Ehlers indicators

---

## 4. Rachnog/Deep-Trading — DL Trading Experiments

**What it is:** Collection of deep learning experiments for trading. 1.5k stars.

**Most relevant folders for AprilAlgo:**

| Priority | Folder | Value |
|----------|--------|-------|
| **HIGH** | `top-ten-mistakes/` | **Must read** — illustrates fake strategies, leakage, backtesting pitfalls |
| **HIGH** | `backtesting/` | Strategy/Portfolio pattern, PnL loop — conceptual overlap with our engine |
| **HIGH** | `multivariate/` | Direction classification (up/down) from windowed OHLCV with 1D CNN — closest to signal classification |
| **HIGH** | `hyperparameters/` | Hyperopt (TPE) for window size, depth, activations — useful tuning workflow idea |
| **MEDIUM** | `strategy/` (`skew.py`) | RSI, MACD, Williams %R, rolling skew/kurt fed into dense network — feature engineering reference |
| **MEDIUM** | `volatility/` | Conv1D for future realized vol — volatility as learned indicator |
| **MEDIUM** | `bayesian/` | Pyro notebooks for Bayesian NN — uncertainty/probability around predictions |
| **LOWER** | `multimodal/`, `time_series_plus_text/` | DJIA + news — out of current scope |

**Quality: Educational / research notebooks, NOT production code.**
- Legacy Keras 1.x API, Python 2 prints in places
- No packaging, no tests, no reproducible env
- Treat as "starred reference + blog companion"

**Key takeaways to apply:**
1. `top-ten-mistakes/` — study before building ML pipeline
2. Classification framing (up/down) is more natural for indicator-probability than regression
3. Hyperopt/TPE for systematic tuning
4. Author's later work (Advanced-Deep-Trading) goes deeper on labeling

**When to use:** v0.3 — mental models for ML pipeline design

---

## 5. bobyyandfriends/StockTradingAI — Deep Q-Learning

**What it is:** DQN stock trader using TensorFlow/Keras/OpenAI Gym on S&P 500 data. This is your own fork.

**Relevant concepts:**
- Time-based train/val/test split (2006-2016 / 2016-2018 / 2018-2021) — good leakage discipline
- Universe construction from S&P 500 (Wikipedia scraping, filtering)
- Bulk `yfinance` download pattern
- Normalizer + `.npy` + `.pkl` preprocessing workflow

**What does NOT transfer:**
- Single Colab notebook — not modular
- OpenAI Gym env is brittle to port
- DQN optimizes cumulative reward — different from supervised signal classification
- RL policies are hard to explain (conflicts with interpretability goal)
- Old TF/Keras assumptions

**Recommendation:** Learn from the train/test discipline and universe construction. Do not try to port the DQN. If RL interests you later (v0.4+), use a modern framework (Stable-Baselines3, RLlib).

**When to use:** v0.4+ — only if exploring RL as an alternative to supervised ML

---

## 6. hudson-and-thames/mlfinlab — ML Financial Lab

**What it is:** Professional ML toolkit for quantitative finance. 4.6k stars.

### LICENSE WARNING: ALL RIGHTS RESERVED
The public repo exists for issue tracking only. Code requires a paid Business or Enterprise license. **Do NOT copy code into AprilAlgo.**

**Concepts to implement ourselves (from the book "Advances in Financial Machine Learning"):**

| Concept | AprilAlgo Module | Priority |
|---------|-----------------|----------|
| Triple-barrier labeling | `src/aprilalgo/labels/` | v0.3 |
| Meta-labeling | `src/aprilalgo/labels/` | v0.3 |
| Purged k-fold CV + embargo | `src/aprilalgo/ml/cv.py` | v0.3 |
| Sequential bootstrap sampling | `src/aprilalgo/ml/sampling.py` | v0.3 |
| Fractional Kelly position sizing | `src/aprilalgo/backtest/portfolio.py` | v0.2-v0.3 |
| Feature importance (MDI, MDA) | `src/aprilalgo/ml/importance.py` | v0.3 |
| Information-driven bars (tick/volume/dollar) | `src/aprilalgo/data/bars.py` | v0.4+ |

**Open-source alternatives for the same ideas:**
- Community AFML exercise repos (BSD/MIT — verify each)
- RiskLabAI.py (BSD 3-Clause — verify)
- Roll your own with NumPy/Pandas + sklearn custom splitters
- The algorithms are in the published literature, not proprietary

**Recommendation:** Use mlfinlab's feature list as a *checklist*. Implement from the book + papers under Apache 2.0 in AprilAlgo.

**When to use:** v0.3 — labeling, CV, sampling concepts

---

## 7. StockSharp/StockSharp — Trading Platform (C#)

**What it is:** Full algorithmic trading platform in C#. 9.6k stars, Apache-2.0.

**No direct code reuse** (C# → Python gap too large).

**Architectural patterns to borrow:**

| Pattern | AprilAlgo Application |
|---------|----------------------|
| Connector abstraction (normalize many data sources) | Data source adapter: CSV, Massive API, Parquet → same internal format |
| Ingestion / Execution / UI separation | Data pipeline, backtest engine, and Streamlit UI as clear modules |
| Hydra-style compressed storage | Future: Parquet files, chunked storage, registry API |
| Event/message-driven strategies | Strategy `on_bar()` already follows this pattern |
| Multi-timeframe candle types | Resampling + look-ahead-safe joins |

**Recommendation:** Browse their architecture docs when designing our folder structure. Don't import code.

**When to use:** Now — architecture design reference

---

## 8. joelowj/awesome-algorithmic-trading — Resource List

**What it is:** Curated list of algo trading resources (tutorials, papers, books, communities).

**Top resources for AprilAlgo's goals:**

### Tutorials:
1. [ML for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501) (Udacity) — ML + market data
2. [AI for Trading](https://www.udacity.com/course/ai-for-trading--nd880) (Udacity) — signal/alpha workflow
3. [ML & RL in Finance](https://www.coursera.org/specializations/machine-learning-reinforcement-finance) (Coursera)
4. [MIT 18.S096 Topics in Math for Finance](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/) — stochastic processes, time series
5. [Financial Engineering & Risk Management](https://www.coursera.org/learn/financial-engineering-1/) (Coursera)

### Research Papers:
1. "The 101 Ways to Measure Portfolio Performance" — metrics discipline
2. "Momentum" — classic signal/factor literature
3. "Pairs Trading: Relative Value Arbitrage" — statistical strategies

### Books:
1. "Building Winning Algorithmic Trading Systems" — data mining → Monte Carlo → live
2. "Hands-On ML with Scikit-Learn and TensorFlow" — ML depth
3. "Finding Alphas" — systematic alpha research process

### Communities:
1. [QuantConnect](https://www.quantconnect.com/) — backtesting ecosystem, Python-friendly
2. [QuantStart](https://www.quantstart.com/) — backtesting articles
3. [The Python Quants Group](http://tpq.io/) — Python-first quant community

### Backtesting Articles (most directly relevant):
- [Successful Backtesting Part I](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-I)
- [Successful Backtesting Part II](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II)

**NOTE:** This list has NO resources specifically about DeMark indicators, Hurst exponents, or multi-timeframe confluence. For those topics, seek dedicated sources outside this repo.

**When to use:** Ongoing — self-education alongside development

---

## Dependency Decision Summary

### Direct Dependencies (add to pyproject.toml):
- `massive` — data fetching (v0.2)
- `shap` — explainability (v0.3)
- `xgboost` — ML signals (v0.3)
- `streamlit` — UI (v0.3)

### Reference Only (port formulas, don't import):
- `legitindicators` — Hurst, Ehlers indicator math
- Deep-Trading notebooks — ML pipeline patterns
- StockTradingAI — train/test split discipline

### Cannot Use (license restrictions):
- `mlfinlab` — implement concepts from the book instead

### Architecture Inspiration (design, not code):
- StockSharp — module separation patterns
- awesome-algorithmic-trading — learning resources
