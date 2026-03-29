# How Your Project Works — A Complete Beginner's Guide

_Everything we built, explained with diagrams and analogies._

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Git — Your Time Machine](#2-git--your-time-machine)
3. [GitHub — Your Cloud Backup](#3-github--your-cloud-backup)
4. [The Git Workflow — Three Zones](#4-the-git-workflow--three-zones)
5. [Branches — Parallel Universes](#5-branches--parallel-universes)
6. [Python and Packages — The Kitchen Analogy](#6-python-and-packages--the-kitchen-analogy)
7. [Virtual Environments — Sealed Containers](#7-virtual-environments--sealed-containers)
8. [uv — The Master Chef's Toolbox](#8-uv--the-master-chefs-toolbox)
9. [pyproject.toml and uv.lock — Recipe vs Receipt](#9-pyprojecttoml-and-uvlock--recipe-vs-receipt)
10. [Project Structure — The Building Blueprint](#10-project-structure--the-building-blueprint)
11. [How Everything Connects](#11-how-everything-connects)
12. [The Full Lifecycle of a Change](#12-the-full-lifecycle-of-a-change)

---

## 1. The Big Picture

Here's everything we installed and how it all connects:

```
YOUR COMPUTER                                        THE INTERNET
+--------------------------------------------------+     +------------------+
|                                                  |     |                  |
|  +--------+   +-----+   +----+   +-----------+  |     |  +------------+  |
|  | Rust   |-->| uv  |-->| .venv |-->| Python  |  |     |  |  GitHub    |  |
|  | (engine)|  |(tool)|  |(bubble)| | packages |  |     |  |  (backup)  |  |
|  +--------+   +-----+   +----+   +-----------+  |     |  +------^-----+  |
|                                                  |     |         |        |
|  +-----+          +---------------------------+  |     |         |        |
|  | Git  |--------->|     AprilAlgo project     |--+-----+--push--+        |
|  |(tracker)|       |                           |  |     |                 |
|  +-----+          +---------------------------+  |     +-----------------+
|                                                  |
+--------------------------------------------------+

  Rust     = Engine that powers uv (you never touch it directly)
  uv       = Installs Python, creates .venv, manages packages
  .venv    = Isolated bubble for your project's packages
  Git      = Tracks every change you make (local time machine)
  GitHub   = Cloud copy of your Git history (online backup)
```

**Analogy:** Think of your project like a restaurant.
- **Rust** is the factory that built your kitchen equipment
- **uv** is the restaurant manager who orders ingredients, hires cooks
- **.venv** is your restaurant's private kitchen (separate from every other restaurant)
- **Git** is the security camera that records everything that happens
- **GitHub** is the off-site backup of all your security footage

---

## 2. Git — Your Time Machine

Git takes "snapshots" of your entire project at any moment. Each snapshot is called a **commit**.

```
  Time --->

  Commit 1          Commit 2          Commit 3          Commit 4
  +----------+      +----------+      +----------+      +----------+
  | main.py  |      | main.py  |      | main.py  |      | main.py  |
  | README   |      | README   |      | README   |      | README   |
  | init.py  |      | init.py  |      | init.py  |      | init.py  |
  |          |      | data.py  |      | data.py  |      | data.py  |
  |          |      |          |      | chart.py |      | chart.py |
  +----------+      +----------+      +----------+      +----------+
       |                 |                 |                  |
  "Initial          "Added data      "Added chart      "Fixed bug
   commit"           loading"         module"           in chart"
```

**Key insight:** You can JUMP to any snapshot at any time. Broke everything in Commit 4? Jump back to Commit 3. It's like unlimited undo, but for your entire project.

**Analogy:** Imagine writing a book. Without Git, you have one copy. If you accidentally delete Chapter 3, it's gone. With Git, you saved a copy after every writing session. You can open "Tuesday's version" and get Chapter 3 back.

### What a commit actually contains

```
+---------------------------+
|  Commit: a507c5d          |  <-- Unique ID (like a fingerprint)
|  Author: Joshua           |
|  Date: Mar 29, 2026       |
|  Message: "Initial commit"|
|                           |
|  Files changed:           |
|    + .gitignore   (new)   |  <-- "+" means added
|    + README.md    (new)   |
|    + main.py      (new)   |
|    + pyproject.toml (new) |
|    ... 8 more files       |
+---------------------------+
```

Every commit stores:
- **WHO** made the change (your name + email)
- **WHEN** it happened (timestamp)
- **WHAT** changed (the exact lines added/removed)
- **WHY** it changed (your commit message)

---

## 3. GitHub — Your Cloud Backup

Git lives on YOUR computer. GitHub lives on THE INTERNET. They sync with `push` and `pull`.

```
  YOUR LAPTOP                           GITHUB.COM
  +--------------------+                +--------------------+
  |  AprilAlgo/.git    |    git push    |  AprilAlgo repo    |
  |                    | =============> |                    |
  |  Commit 1          |                |  Commit 1          |
  |  Commit 2          |    git pull    |  Commit 2          |
  |  Commit 3          | <============= |  Commit 3          |
  |                    |                |                    |
  +--------------------+                +--------------------+
    (your time machine)                  (cloud copy of it)
```

**Why both?**
- **Git** (local) = Fast, works offline, private
- **GitHub** (cloud) = Backup, sharing, portfolio, collaboration

**Analogy:** Git is your personal diary. GitHub is a copy of that diary stored in a bank safe-deposit box. Even if your house burns down, the bank has your diary.

### What lives where

```
  YOUR COMPUTER                    GITHUB
  +--------------------------+     +-------------------------+
  | .git/         (hidden)   |     | Repository page         |
  | .venv/        (ignored)  |     |   README.md (displayed) |
  | .gitignore               |     |   All committed files   |
  | README.md                |     |   Commit history        |
  | main.py                  |     |   Issues / PRs          |
  | pyproject.toml           |     |                         |
  | uv.lock                  |     | Does NOT have:          |
  | src/aprilalgo/           |     |   .venv (too big)       |
  |   __init__.py            |     |   __pycache__ (temp)    |
  +--------------------------+     +-------------------------+

  .gitignore decides what NEVER goes to GitHub
```

---

## 4. The Git Workflow — Three Zones

This is the most important concept in Git. Your files exist in three zones:

```
  +------------------+    git add     +------------------+   git commit   +------------------+
  |                  | =============> |                  | =============> |                  |
  |  WORKING         |                |  STAGING AREA    |                |  REPOSITORY      |
  |  DIRECTORY       |                |  (the "cart")    |                |  (the "vault")   |
  |                  | <--- edit ---> |                  |                |                  |
  |  Where you edit  |                |  Files selected  |                |  Permanent       |
  |  your code live  |                |  for next commit |                |  snapshots       |
  |                  |                |                  |                |                  |
  +------------------+                +------------------+                +------------------+
                                                                                  |
                                                                                  | git push
                                                                                  v
                                                                         +------------------+
                                                                         |  GITHUB          |
                                                                         |  (cloud vault)   |
                                                                         +------------------+
```

**Analogy — Online Shopping:**
1. **Working Directory** = You're browsing the store, trying things on
2. **Staging Area** = You put items in your shopping cart
3. **Commit** = You click "Purchase" — it's now in your order history
4. **Push** = The store ships it to the warehouse (GitHub)

**In practice:**

```
  You edit main.py              -->  Working Directory (changed)
  You run: git add main.py     -->  Staging Area (selected for commit)
  You run: git commit -m "..."  -->  Repository (saved forever)
  You run: git push             -->  GitHub (uploaded to cloud)
```

### Why have a staging area at all?

Because sometimes you change 5 files but only want to save 2 of them right now:

```
  Changed files:        Staging Area:        Commit:
  +--------------+      +--------------+     +--------------+
  | main.py      |  add | main.py      | --> | main.py      |
  | data.py      | ---> | data.py      |     | data.py      |
  | scratch.py   |      |              |     |              |
  | notes.txt    |      |              |     | "Fixed data  |
  | test.py      |      |              |     |  loading"    |
  +--------------+      +--------------+     +--------------+

  scratch.py, notes.txt, test.py stay behind — not ready yet
```

---

## 5. Branches — Parallel Universes

You don't have branches yet, but you will soon. Here's the concept:

```
                          main (stable, working code)
  Commit 1 --- Commit 2 --- Commit 3 --- Commit 4
                    \                        /
                     \--- Commit A --- Commit B
                          feature-branch (experimental code)
```

**Analogy:** You're writing a book (main branch). You want to try rewriting Chapter 5 but aren't sure it'll be good. So you make a COPY of the book (feature branch), rewrite Chapter 5 there. If it's good, you paste it back into the original. If it's bad, you throw away the copy. The original was never at risk.

```
  BEFORE BRANCHING:              AFTER MERGING BACK:

  main: Ch1, Ch2, Ch3, Ch4      main: Ch1, Ch2, Ch3, Ch4, Ch5(new!)
            \                               /
             Ch5-draft-1, Ch5-draft-2 -----+
             (experimental branch)     (merge: the experiment worked!)
```

**When to use branches (later):**
- `main` = always works, always stable
- `feature/add-charts` = you're building a new chart module
- `fix/date-bug` = you're fixing a specific bug
- When the feature/fix is done, you **merge** it back into main

---

## 6. Python and Packages — The Kitchen Analogy

Python by itself is like an empty kitchen. It has a stove, a sink, and a counter, but no ingredients. **Packages** are the ingredients.

```
  Python (empty kitchen)          Python + Packages (stocked kitchen)
  +-------------------+           +-------------------+
  |  print()          |           |  print()          |
  |  len()            |           |  len()            |
  |  open()           |           |  open()           |
  |  if/else/for      |           |  if/else/for      |
  |                   |           |                   |
  |  (basic stuff)    |           |  pandas           |  <-- read CSVs, tables
  |                   |           |  numpy            |  <-- fast math
  |                   |           |  matplotlib       |  <-- draw charts
  |                   |           |  jupyter           |  <-- interactive notebooks
  +-------------------+           +-------------------+
```

**How packages connect to each other:**

When you installed 4 packages, uv actually installed 106. Why? Because packages depend on other packages:

```
  You asked for:          Which need:              Which need:
  +------------+          +------------------+     +------------------+
  | pandas     | -------> | numpy            | --> | (math libraries) |
  |            | -------> | python-dateutil  | --> | six              |
  |            | -------> | pytz             |     +------------------+
  +------------+          +------------------+

  +------------+          +------------------+     +------------------+
  | matplotlib | -------> | numpy            |     | (already there!) |
  |            | -------> | pillow           | --> | (image support)  |
  |            | -------> | kiwisolver       |     +------------------+
  |            | -------> | cycler           |
  +------------+          +------------------+

  +------------+          +------------------+
  | jupyter    | -------> | ipython          |     This is why 4 packages
  |            | -------> | notebook         |     turned into 106!
  |            | -------> | jupyterlab       |     Each one brings along
  |            | -------> | ipykernel        |     everything IT needs.
  +------------+          +------------------+
```

**Analogy:** You ordered a pizza (pandas). The pizza needs dough, sauce, cheese, toppings. The dough needs flour, water, yeast. The sauce needs tomatoes, garlic, salt. One "pizza" order triggered a chain of 20+ ingredients. Same thing with Python packages.

---

## 7. Virtual Environments — Sealed Containers

The `.venv` folder is a **virtual environment**. It's the most important concept for keeping your projects healthy.

**The problem without virtual environments:**

```
  YOUR WHOLE COMPUTER (one shared kitchen)
  +--------------------------------------------------+
  |                                                  |
  |  Project A needs pandas 1.5                      |
  |  Project B needs pandas 2.0     CONFLICT!        |
  |  Project C needs pandas 1.8                      |
  |                                                  |
  |  They all share the same Python installation     |
  |  Only ONE version of pandas can exist            |
  |  Updating for Project B BREAKS Project A         |
  |                                                  |
  +--------------------------------------------------+
```

**The solution with virtual environments:**

```
  Project A                Project B               Project C
  +----------------+       +----------------+      +----------------+
  | .venv/          |       | .venv/          |      | .venv/          |
  |   pandas 1.5   |       |   pandas 2.0   |      |   pandas 1.8   |
  |   numpy 1.21   |       |   numpy 1.24   |      |   numpy 1.23   |
  |   (own Python) |       |   (own Python) |      |   (own Python) |
  +----------------+       +----------------+      +----------------+
       ISOLATED                 ISOLATED                ISOLATED
```

**Analogy:** Without virtual environments, your projects are roommates sharing one fridge — they fight over space and food. With virtual environments, each project gets its own apartment with its own fridge. What happens in one apartment doesn't affect the others.

### What's literally inside .venv?

```
  .venv/
  +-------------------------------------------+
  |  Scripts/                                  |
  |    python.exe    <-- A copy of Python      |
  |    pip.exe       <-- Package installer     |
  |    jupyter.exe   <-- Jupyter launcher      |
  |                                            |
  |  Lib/                                      |
  |    site-packages/                          |
  |      pandas/     <-- Actual package code   |
  |      numpy/                                |
  |      matplotlib/                           |
  |      ... (106 packages)                    |
  |                                            |
  +-------------------------------------------+
  Size: ~500MB+ (this is why we DON'T put it on GitHub)
```

---

## 8. uv — The Master Chef's Toolbox

Before `uv`, you needed FIVE separate tools. `uv` replaces them all:

```
  THE OLD WAY (confusing)              THE NEW WAY (uv does it all)
  +---------------------------+        +---------------------------+
  |                           |        |                           |
  |  pyenv   - pick Python    |        |                           |
  |  venv    - create .venv   |        |         uv                |
  |  pip     - install pkgs   |  --->  |                           |
  |  pip-tools - lock versions|        |    One tool. Done.        |
  |  poetry  - manage project |        |                           |
  |                           |        |                           |
  +---------------------------+        +---------------------------+
```

**What each uv command does:**

```
  +-------------------+--------------------------------------------+
  | Command           | What it does                               |
  +-------------------+--------------------------------------------+
  | uv init           | Creates pyproject.toml, .python-version    |
  |                   |   Like filling out a birth certificate     |
  +-------------------+--------------------------------------------+
  | uv add pandas     | Installs pandas into .venv                 |
  |                   |   Updates pyproject.toml + uv.lock         |
  |                   |   Like ordering a new ingredient           |
  +-------------------+--------------------------------------------+
  | uv sync           | Reads uv.lock, installs everything         |
  |                   |   Like restocking from a shopping list     |
  +-------------------+--------------------------------------------+
  | uv run python x   | Runs x.py using the .venv Python           |
  |                   |   Like cooking with YOUR kitchen's tools   |
  +-------------------+--------------------------------------------+
  | uv remove pandas  | Uninstalls pandas from .venv               |
  |                   |   Updates pyproject.toml + uv.lock         |
  |                   |   Like returning an ingredient             |
  +-------------------+--------------------------------------------+
```

---

## 9. pyproject.toml and uv.lock — Recipe vs Receipt

These two files work as a pair but do very different things:

```
  pyproject.toml (THE RECIPE)          uv.lock (THE RECEIPT)
  +----------------------------+       +----------------------------+
  | "I need pandas >= 3.0"     |       | "pandas 3.0.1"             |
  | "I need numpy >= 2.4"      |       | "numpy 2.4.4"              |
  | "I need matplotlib >= 3.10"|       | "matplotlib 3.10.8"        |
  | "I need jupyter >= 1.1"    |       | "pillow 12.1.1"            |
  |                            |       | "kiwisolver 1.5.0"         |
  | (4 items — what you want)  |       | "cycler 0.12.1"            |
  |                            |       | ... (106 exact items)      |
  +----------------------------+       +----------------------------+
   YOU write this (via uv add)          UV writes this automatically
   Says "at least this version"         Says "EXACTLY this version"
   Human-readable wishlist              Machine-readable master list
```

**Analogy:**
- `pyproject.toml` = "I want pizza, salad, and drinks for the party"
- `uv.lock` = "Domino's large pepperoni ($14.99, order #38291), Caesar salad from Costco (SKU 847261, $8.49), 24-pack Coca-Cola from Walmart ($12.98)"

The recipe says WHAT. The receipt says EXACTLY WHAT, from WHERE, at WHAT PRICE.

**Why you need both:**
- `pyproject.toml` is for humans — quick to read, easy to update
- `uv.lock` is for machines — ensures everyone gets the EXACT same packages

---

## 10. Project Structure — The Building Blueprint

```
  AprilAlgo/
  |
  |-- .git/                  HIDDEN: Git's database (don't touch)
  |-- .venv/                 HIDDEN: Virtual environment (don't touch)
  |
  |-- .gitignore             GATEKEEPER: Decides what Git ignores
  |-- .python-version        CONFIG: Which Python version to use
  |
  |-- pyproject.toml         IDENTITY: Project name, version, dependencies
  |-- uv.lock                LOCK: Exact versions of all 106 packages
  |
  |-- main.py                CODE: Entry point (where your program starts)
  |
  |-- src/                   CODE: Your reusable library
  |   +-- aprilalgo/
  |       +-- __init__.py         Makes this folder a "package"
  |
  |-- README.md              DOCS: For humans visiting your project
  |-- LICENSE                LEGAL: Apache 2.0 (how others can use your code)
  |-- CHANGELOG.md           DOCS: History of every version
  |-- CLAUDE.md              AI: Instructions for Claude
  |-- AGENTS.md              AI: Instructions for Cursor/AI agents
  |-- HANDOFF.md             DOCS: Plain-English guide for you
  +-- LEARNING.md            DOCS: This file
```

**Think of it as layers:**

```
  +----------------------------------------------------+
  |                  DOCUMENTATION LAYER                |
  |  README, LICENSE, CHANGELOG, HANDOFF, LEARNING     |
  |  (Explains the project to humans and AI)           |
  +----------------------------------------------------+
  |                  CONFIGURATION LAYER                |
  |  pyproject.toml, uv.lock, .gitignore, .python-ver  |
  |  (Tells tools how to set up the project)           |
  +----------------------------------------------------+
  |                  CODE LAYER                         |
  |  main.py, src/aprilalgo/__init__.py                |
  |  (The actual program — where the work happens)     |
  +----------------------------------------------------+
  |                  INFRASTRUCTURE LAYER               |
  |  .git/, .venv/                                     |
  |  (Hidden machinery — Git history + packages)       |
  +----------------------------------------------------+
```

---

## 11. How Everything Connects

Here's the complete flow from "I want to start coding" to "my code is on GitHub":

```
  FIRST TIME SETUP (you did this today):

  [Install Rust] --> [Install uv] --> [uv init] --> [uv add packages]
                                          |               |
                                          v               v
                                    pyproject.toml    .venv/ created
                                                      uv.lock created
                                          |
                                          v
                              [Create all project files]
                                          |
                                          v
                    [Install Git] --> [git init] --> [git add .] --> [git commit]
                                                                        |
                    [Install gh] --> [gh auth login]                     v
                                          |                      [gh repo create]
                                          v                             |
                                    [Authenticated]                     v
                                                               [Code on GitHub!]
```

```
  DAILY WORKFLOW (what you'll do going forward):

  +--Write code--+     +--Save to Git--+     +--Upload--+
  |              |     |               |     |          |
  |  Edit files  | --> |  git add .    | --> | git push |
  |  in Cursor   |     |  git commit   |     |          |
  |              |     |  -m "message" |     |          |
  +--------------+     +---------------+     +----------+

  +--Add packages--+
  |                |
  |  uv add scipy  |  (uv updates pyproject.toml + uv.lock + .venv)
  |                |
  +----------------+
```

---

## 12. The Full Lifecycle of a Change

Let's trace one change through the ENTIRE system. Say you add a new file called `analysis.py`:

```
  STEP 1: You create analysis.py in Cursor
  +---------------------------------------------------------+
  |  # analysis.py                                          |
  |  import pandas as pd                                    |
  |                                                         |
  |  data = pd.read_csv("sales.csv")                       |
  |  print(data.head())                                     |
  +---------------------------------------------------------+
          |
          v
  STEP 2: Git notices the new file (Working Directory)
  +---------------------------------------------------------+
  |  $ git status                                           |
  |  Untracked files:                                       |
  |    analysis.py      <-- Git sees it but isn't tracking  |
  +---------------------------------------------------------+
          |
          v
  STEP 3: You stage it (Staging Area)
  +---------------------------------------------------------+
  |  $ git add analysis.py                                  |
  |  Changes to be committed:                               |
  |    new file: analysis.py    <-- Now in the "cart"       |
  +---------------------------------------------------------+
          |
          v
  STEP 4: You commit it (Local Repository)
  +---------------------------------------------------------+
  |  $ git commit -m "Add sales data analysis script"       |
  |  [main b3f8a2c] Add sales data analysis script          |
  |   1 file changed, 5 insertions(+)                       |
  |   create mode 100644 analysis.py                        |
  +---------------------------------------------------------+
          |
          v
  STEP 5: You push to GitHub (Remote Repository)
  +---------------------------------------------------------+
  |  $ git push                                             |
  |  To https://github.com/bobyyandfriends/AprilAlgo.git   |
  |     a507c5d..b3f8a2c  main -> main                     |
  +---------------------------------------------------------+
          |
          v
  STEP 6: Visible on GitHub!
  +---------------------------------------------------------+
  |  github.com/bobyyandfriends/AprilAlgo                   |
  |                                                         |
  |  Commits:                                               |
  |    b3f8a2c  Add sales data analysis script    (2 min)   |
  |    a507c5d  Initial commit: project scaffolding (today) |
  +---------------------------------------------------------+
```

---

## Brainstorming: Where Can You Go From Here?

Now that the foundation is built, here are paths you could explore:

```
  YOUR CURRENT STATE: Empty project, fully wired up
  |
  +---> PATH A: Data Analysis (recommended first)
  |     Load a CSV, explore it, make charts
  |     Skills: pandas, matplotlib
  |     Time: 1-2 hours
  |
  +---> PATH B: Jupyter Notebooks
  |     Interactive coding in a browser
  |     Great for experimenting and learning
  |     Run: uv run jupyter notebook
  |
  +---> PATH C: Build a CLI Tool
  |     A program you run from the terminal
  |     "python main.py --analyze sales.csv"
  |     Skills: argparse, file I/O
  |
  +---> PATH D: Web Dashboard (later)
  |     Visualize data in a browser with Streamlit
  |     uv add streamlit
  |     Run: uv run streamlit run app.py
  |
  +---> PATH E: Machine Learning (later)
  |     Train models to make predictions
  |     uv add scikit-learn
  |     Skills: data splitting, model training
  |
  +---> PATH F: API / Automation (later)
        Pull data from the internet automatically
        uv add requests
        Skills: HTTP requests, JSON parsing
```

**My recommendation for your first real task:**

```
  1. Find a CSV file that interests you
     (Kaggle.com has thousands of free datasets)

  2. Start a Jupyter notebook
     $ uv run jupyter notebook

  3. Load and explore:
     import pandas as pd
     data = pd.read_csv("your_file.csv")
     data.head()          # see first 5 rows
     data.describe()      # see stats
     data.info()          # see column types

  4. Make a chart:
     import matplotlib.pyplot as plt
     data["column_name"].plot(kind="bar")
     plt.show()

  5. Save your work:
     $ git add .
     $ git commit -m "First data exploration"
     $ git push
```

---

## Quick Reference Card

```
  +------------------+-----------------------------+---------------------------+
  | I want to...     | Command                     | What it does              |
  +------------------+-----------------------------+---------------------------+
  | Start Jupyter    | uv run jupyter notebook     | Opens browser notebook    |
  | Run my script    | uv run python main.py       | Runs main.py              |
  | Add a package    | uv add <name>               | Installs + updates config |
  | Remove a package | uv remove <name>            | Uninstalls + updates      |
  | See what changed | git status                  | Shows modified files      |
  | Save my work     | git add . && git commit     | Takes a snapshot          |
  | Upload to GitHub | git push                    | Syncs to the cloud        |
  | Download updates | git pull                    | Gets changes from GitHub  |
  | See history      | git log --oneline           | Lists all commits         |
  | Undo last change | git checkout -- <file>      | Reverts a file            |
  +------------------+-----------------------------+---------------------------+
```
