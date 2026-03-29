# CLAUDE.md — Context for Claude AI

This file provides Claude with everything it needs to understand and work effectively in the AprilAlgo project.

---

## Project Overview

**AprilAlgo** is a Python data science and analysis project. The goal is to load, clean, explore, and visualize data using standard Python data science libraries.

- **Language:** Python 3.11+
- **Package manager:** `uv` (replaces pip + venv)
- **License:** Apache 2.0
- **Owner:** Joshua

---

## Project Structure

```
AprilAlgo/
├── .gitignore
├── .python-version         # Pins Python version for uv
├── .venv/                  # Virtual environment (do NOT edit manually)
├── AGENTS.md               # Rules for AI coding agents
├── CHANGELOG.md            # Version history
├── CLAUDE.md               # This file
├── LICENSE                 # Apache 2.0
├── README.md               # User-facing documentation
├── main.py                 # Entry point script
├── pyproject.toml          # Project config and dependencies
├── uv.lock                 # Locked dependency versions (commit this)
└── src/
    └── aprilalgo/
        └── __init__.py     # Package entry point and version
```

---

## Key Commands

| Task                        | Command                          |
|-----------------------------|----------------------------------|
| Install all dependencies    | `uv sync`                        |
| Run the main script         | `uv run python main.py`          |
| Start Jupyter Notebook      | `uv run jupyter notebook`        |
| Add a new package           | `uv add <package>`               |
| Remove a package            | `uv remove <package>`            |
| Run a Python file           | `uv run python <file.py>`        |

---

## Dependencies

Core data science stack:
- `pandas` — DataFrames, data cleaning, CSV/Excel I/O
- `numpy` — Arrays, math, linear algebra
- `matplotlib` — Plots, charts, figures
- `jupyter` — Interactive notebook environment

---

## Coding Conventions

- Use `uv add` to add dependencies — never edit `pyproject.toml` dependencies by hand.
- Always run scripts with `uv run` so the virtual environment is activated automatically.
- Place reusable code in `src/aprilalgo/` as modules.
- Keep analysis notebooks in a `notebooks/` folder (create it when needed).
- Keep raw data in `data/raw/` and cleaned data in `data/processed/` (create when needed).
- Follow PEP 8 style. Keep lines under 100 characters.
- Commit `uv.lock` to version control.

---

## When Helping with This Project

- Prefer `uv run` over activating the venv manually.
- When adding new files, follow the existing structure.
- When changing dependencies, always use `uv add` / `uv remove`.
- Do not delete or regenerate `uv.lock` — it is managed by uv automatically.
- Check `CHANGELOG.md` before suggesting version bumps.
