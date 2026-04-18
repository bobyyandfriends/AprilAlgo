# CLAUDE.md — Claude-specific context for AprilAlgo

> **This file is intentionally thin.** The canonical rules, command reference, testing guide, model routing, and session protocol all live in [`AGENTS.md`](AGENTS.md). Read that first. Everything below is additive, Claude-only preference or context — if anything conflicts with `AGENTS.md`, `AGENTS.md` wins.

---

## Single source of truth

| Topic | Canonical file |
|---|---|
| Project rules, coding conventions, commands, tests | [`AGENTS.md`](AGENTS.md) |
| System design + diagrams | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Column contracts between layers | [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) |
| Triple-barrier labeling math | [`docs/TRIPLE_BARRIER_MATH.md`](docs/TRIPLE_BARRIER_MATH.md) |
| Today's state + next sprint | [`PROJECT_STATE.md`](PROJECT_STATE.md) |
| Release history | [`CHANGELOG.md`](CHANGELOG.md) |
| Project-bootstrap narrative | [`docs/HANDOFF.md`](docs/HANDOFF.md) |
| Archived onboarding + external research | [`docs/archive/LEARNING.md`](docs/archive/LEARNING.md), [`docs/archive/REPO_ANALYSIS.md`](docs/archive/REPO_ANALYSIS.md) |

Always consult `PROJECT_STATE.md` §4 "Warning zone" before making changes — it enumerates the unfixed technical debt the owner already knows about.

---

## Claude-specific preferences

- **Edit over rewrite.** Prefer small targeted edits to existing files. Only write new files when explicitly asked.
- **Surface todos as checklists** in `PROJECT_STATE.md` §3 "Next sprint" rather than in ad-hoc comments.
- **ASCII for diagrams** in `ARCHITECTURE.md` (no mermaid) — keeps them readable in terminal viewers and in diff output.
- **Diff summaries at the end** of multi-file refactors: list the files touched and a one-line rationale per file.
- **Cite code by file + line range**, not by pasting whole blocks, when explaining existing behavior.
- **When running tests, prefer `uv run pytest tests/ -q`** for fast green/red and `-v` only when diagnosing a failure.
- **Never run destructive git commands** (`push --force`, `reset --hard`, hook-skipping flags) unless the owner explicitly requests them.

---

## Non-negotiables (mirror from `AGENTS.md` §2)

These are repeated here so Claude can't miss them if it skips straight to `CLAUDE.md`. Read [`AGENTS.md`](AGENTS.md) §2 for the authoritative numbered list.

- No look-ahead bias. No hardcoded API keys. No modifying `data/` contents.
- Every indicator emits parameterized `{name}_{params}_(bull|bear)` columns. Register in `indicators/descriptor.py`.
- Pure functions for indicators (`DataFrame → DataFrame`). Type hints everywhere. `pathlib.Path` for file paths.
- `uv add`, never hand-edit `pyproject.toml`. `uv run` for every script.
- If `meta.json` keys change, `docs/DATA_SCHEMA.md` must be updated in the same session.

---

## Things Claude must not do

- Do not create `.md` files proactively.
- Do not commit unless explicitly asked.
- Do not update `git config`.
- Do not pass `--no-verify` / `--no-gpg-sign` flags.
- Do not modify the contents of `data/` or the `tests/fixtures/` CSVs.
- Do not add comments that merely narrate what the code does.
