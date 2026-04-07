# AGENTS.md — Rules for AI Coding Agents

Non-negotiable rules for any AI agent working on this project.

---

## Project Identity

- **Name:** AprilAlgo
- **Type:** Modular multi-timeframe trading backtester
- **Language:** Python 3.11+
- **Package manager:** `uv`
- **License:** Apache 2.0
- **Owner:** Joshua

---

## Non-Negotiable Rules

1. **No look-ahead bias** — indicators, strategies, and features must use only past and current data
2. **No hardcoded API keys** — use environment variables or config files
3. **No modifying data/ contents** — CSV data is gitignored and managed separately
4. **Every indicator must emit parameterized bull/bear columns** — `{name}_{params}_bull` and `{name}_{params}_bear` (e.g., `rsi_14_bull`, `sma_20_bear`)
5. **Pure functions for indicators** — `DataFrame → DataFrame`, no side effects
6. **Strategies are event handlers** — implement `on_bar()`, never peek at future bars
7. **Type hints required** on all function signatures
8. **PEP 8 style**, lines under 100 characters
9. **Use `pathlib.Path`** for all file paths
10. **Register new indicators in `descriptor.py`** — the catalog is the single source of truth

---

## Adding Dependencies

```bash
uv add <package>          # adds to pyproject.toml and updates uv.lock
uv sync                   # install everything from lock file
```

Never edit `pyproject.toml` dependencies by hand. Always use `uv add`.

---

## File Structure Rules

| Module | Purpose | Key Pattern |
|--------|---------|-------------|
| `data/` | Load, fetch, store, resample OHLCV | `loader.py` returns clean DataFrames |
| `indicators/` | Technical indicators | Pure function: `df → df` with parameterized `_bull`/`_bear` columns |
| `indicators/descriptor.py` | Indicator catalog | `IndicatorSpec` + `ParamSpec` — single source of truth |
| `confluence/` | Multi-timeframe scoring | Align timeframes, tally signals, compute probability |
| `tuner/` | Parameter optimization | Grid search + robustness analysis |
| `backtest/` | Simulation engine | Bar-by-bar loop, trade management, metrics |
| `strategies/` | Trading logic | Subclass `BaseStrategy` or use `ConfigurableStrategy` |
| `ui/` | Streamlit dashboard | Pages pull from descriptor catalog, no hardcoded indicator lists |
| `tests/` | Persistent test suite | `uv run pytest tests/ -v` |

---

## Code Style

- Docstrings: Google style, explain what columns are added
- Imports: `from __future__ import annotations` at top of every module
- Constants: UPPER_SNAKE_CASE at module level
- Classes: PascalCase
- Functions/variables: snake_case
- No comments that just narrate what the code does

---

## Changelog

When making changes, update `CHANGELOG.md` under `[Unreleased]`:
- `Added` for new features
- `Changed` for changes in existing functionality
- `Fixed` for bug fixes
- `Removed` for removed features

---

## Running

```bash
uv run python main.py                                    # default backtest
uv run python main.py --strategy demark_confluence        # DeMark strategy
uv run streamlit run src/aprilalgo/ui/app.py              # Streamlit UI
uv run pytest tests/ -v                                   # test suite (39 tests)
uv run python scripts/fetch_data.py --symbols AAPL,NVDA   # fetch data
```
