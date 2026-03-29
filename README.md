# AprilAlgo

A Python data science and analysis project. Built for exploring, cleaning, and visualizing data using pandas, numpy, matplotlib, and Jupyter.

## Features

- Data loading and cleaning with **pandas**
- Numerical computation with **numpy**
- Charts and plots with **matplotlib**
- Interactive analysis with **Jupyter Notebooks**

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone or download this project.
2. Install dependencies using `uv`:

```bash
uv sync
```

This will automatically create a virtual environment (`.venv`) and install all required packages.

## Usage

### Start a Jupyter Notebook

```bash
uv run jupyter notebook
```

### Run the main script

```bash
uv run python main.py
```

### Import the package in your code

```python
import aprilalgo

print(aprilalgo.__version__)
```

## Project Structure

```
AprilAlgo/
├── .gitignore
├── .python-version
├── .venv/                  # Virtual environment (auto-created by uv)
├── AGENTS.md
├── CHANGELOG.md
├── CLAUDE.md
├── LICENSE
├── README.md               # This file
├── main.py                 # Entry point script
├── pyproject.toml          # Project config and dependencies
├── uv.lock                 # Locked dependency versions
└── src/
    └── aprilalgo/
        └── __init__.py     # Package entry point
```

## Adding New Dependencies

```bash
uv add <package-name>
```

Example:
```bash
uv add seaborn scikit-learn
```

## License

This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.
