# AprilAlgo — Documentation Consolidation Plan (revised)

Date: 2026-04-18
Status: **REVISED after owner feedback — awaiting final approval to execute**.
No documentation files have been deleted, renamed, or merged yet.

---

## 0. Revisions from the first draft (owner feedback)

| # | First draft said | Revised decision |
|---|------------------|------------------|
| 1 | Delete `docs/HANDOFF.md` | **Keep** — owner values it as a "beginning of any project" bootstrap reference |
| 2 | Keep `docs/LEARNING.md` in place | **Move to `docs/archive/LEARNING.md`** |
| 3 | Keep `docs/REPO_ANALYSIS.md` in place | **Move to `docs/archive/REPO_ANALYSIS.md`** — retained for future reference but out of the active doc tree |
| 4 | `PROJECT_STATE.md` becomes the big living tracker (absorb SESSION_HANDOFF, BACKLOG Next, DEBUG_LOG) | **Rewrite `PROJECT_STATE.md` lean** — v0.5 work is complete, so purge the historical milestone log and start a fresh post-v0.5 tracker |
| 5 | Delete `tests.md`, rely on `pytest --collect-only` | **Delete** `tests.md`, **and** add a new "Testing" section to `AGENTS.md` that enumerates every useful pytest invocation the owner can run on demand |
| 6 | Delete `CLAUDE.md`, merge everything into `AGENTS.md` | **Keep `CLAUDE.md` as a thin pointer** — owner wants a dedicated Claude context file. Slim it to: "read AGENTS.md for canonical rules + a short Claude-specific preferences block." This prevents drift. |
| 7 | Delete `BACKLOG.md` | **Delete** (unchanged; v0.5 history lives in git + CHANGELOG `[0.5.0]`) |

---

## 1. Final target structure

**11 markdown files remain** (down from 17); **2 of them** are touched in a
normal coding session.

### 1.1 Core live files — edited per session

| File | Role |
|------|------|
| `PROJECT_STATE.md` | Today's handoff, next sprint, warning zone, CLI edge cases |
| `CHANGELOG.md` | Semver history + `[Unreleased]` |

### 1.2 Release-cadence files — edited when shipped surface changes

| File | Role |
|------|------|
| `README.md` | Front door: install, quick-start, doc links |
| `AGENTS.md` | **Unified** AI guide: rules + model routing + command reference + **testing reference** |
| `CLAUDE.md` | **Thin pointer** → AGENTS.md + short Claude-specific preferences |
| `ARCHITECTURE.md` | System design + forward roadmap (shipped history removed) |

### 1.3 Stable reference files — rarely touched

| File | Role |
|------|------|
| `docs/HANDOFF.md` | Bootstrap / project-start narrative (kept per owner request) |
| `docs/DATA_SCHEMA.md` | Column contracts between layers |
| `docs/TRIPLE_BARRIER_MATH.md` | Triple-barrier labeling math reference |

### 1.4 Archived — kept in the repo but out of the active doc tree

| File | Role |
|------|------|
| `docs/archive/LEARNING.md` | Git / uv / packaging tutorial |
| `docs/archive/REPO_ANALYSIS.md` | External-repo research snapshot (March 2026) |

---

## 2. Final per-file disposition — all 17 files

| # | File | Decision |
|---|------|----------|
| 1 | `README.md` | **KEEP (TRIM)** — fix doc-link list to match the new tree |
| 2 | `ARCHITECTURE.md` | **TRIM** — delete §2 "Versioned Roadmap" shipped content (CHANGELOG covers it), keep forward-only roadmap |
| 3 | `PROJECT_STATE.md` | **REWRITE LEAN** — purge v0.5 milestones + deep log; start fresh with the outline in §4 |
| 4 | `BACKLOG.md` | **DELETE** |
| 5 | `CHANGELOG.md` | **KEEP** |
| 6 | `AGENTS.md` | **KEEP (EXPAND)** — absorb `CLAUDE.md` project-structure content, `COMMAND_CHEAT_SHEET.md`, `docs/MODEL_ROUTING.md`; add new §"Testing" |
| 7 | `CLAUDE.md` | **SLIM** — rewrite as thin pointer + Claude preferences (see §5) |
| 8 | `COMMAND_CHEAT_SHEET.md` | **MERGE → AGENTS.md** then delete |
| 9 | `DEBUG_LOG.md` | **MERGE → PROJECT_STATE.md** (appendix "CLI edge cases") then delete |
| 10 | `tests.md` | **DELETE** (replaced by AGENTS.md §"Testing") |
| 11 | `docs/SESSION_HANDOFF.md` | **MERGE → PROJECT_STATE.md + AGENTS.md** then delete. Sprint log collapses to one line per sprint |
| 12 | `docs/HANDOFF.md` | **KEEP** (owner request) |
| 13 | `docs/MODEL_ROUTING.md` | **MERGE → AGENTS.md** then delete |
| 14 | `docs/DATA_SCHEMA.md` | **KEEP** |
| 15 | `docs/TRIPLE_BARRIER_MATH.md` | **KEEP** |
| 16 | `docs/LEARNING.md` | **MOVE → `docs/archive/LEARNING.md`** |
| 17 | `docs/REPO_ANALYSIS.md` | **MOVE → `docs/archive/REPO_ANALYSIS.md`** |

Deletions: 6 files. Merges: 4 files folded into others. Moves: 2 files to
archive. Rewrites: 2 files (`PROJECT_STATE.md` lean, `CLAUDE.md` thin).

---

## 3. New `AGENTS.md` outline

```
# AGENTS.md — AI Coding Agent Guide

## 1. Project identity & non-negotiable rules
    - Project summary (one paragraph)
    - The 12 non-negotiable rules currently in AGENTS.md §1–§3

## 2. Project structure
    - Tree of src/aprilalgo/ (from CLAUDE.md)
    - "Adding new components" walkthroughs (from CLAUDE.md)

## 3. Coding conventions
    - Merged from current AGENTS.md §"Code Style" + CLAUDE.md "Coding Conventions"

## 4. Command reference                       ← (absorbed COMMAND_CHEAT_SHEET.md)
    4.1 Backtest
    4.2 ML pipeline (train / evaluate / predict / walk-forward)
    4.3 ML meta-label + regime routing + SHAP
    4.4 Information bars
    4.5 UI / Streamlit

## 5. Testing                                  ← NEW, replaces tests.md
    - Full suite:            uv run pytest tests/ -v
    - Quick green check:     uv run pytest tests/ -q
    - Collect-only (live test inventory): uv run pytest tests/ --collect-only -q
    - By module slice:
        uv run pytest tests/test_trainer_sample_weight.py
        uv run pytest tests/test_meta_bundle.py
        uv run pytest tests/test_oof.py
        uv run pytest tests/test_ml_walk_forward.py
        uv run pytest tests/test_cli_ml.py
        uv run pytest tests/test_reporting.py
        ... (one line per logical test module)
    - Keyword filter:        uv run pytest tests/ -k "<substring>"
    - Skipped + reasons:     uv run pytest tests/ -rs
    - Stop at first failure: uv run pytest tests/ -x
    - With coverage:         uv run pytest tests/ --cov=aprilalgo
    - HMM-extra smoke:       uv sync --extra hmm; uv run pytest tests/ -k hmm

## 6. Model routing (Cursor)                  ← (absorbed docs/MODEL_ROUTING.md)
    - Fast / Standard / Heavy tiers + slugs
    - Heuristics + scope-shift rule

## 7. Session protocol                        ← (absorbed docs/SESSION_HANDOFF.md)
    - Start-of-session gate
    - End-of-session handoff template
    - Standing rules (green tests, DATA_SCHEMA updates mandatory)

## 8. Dependencies / changelog discipline
```

§5 "Testing" directly answers your "I want a way to have documentation for
running different tests I can run on my own."

---

## 4. New `PROJECT_STATE.md` outline (lean rewrite)

```
# AprilAlgo — Project State

## 1. Status at a glance
    - Current version: <from pyproject.toml>
    - Test count: run `uv run pytest tests/ --collect-only -q` (live source)
    - Last release: 0.5.0 (2026-04-17)
    - Active branch: main

## 2. Today's handoff
    - Executive status (1–3 lines)
    - What changed this session (bullet list)
    - Known caveats / follow-ups
    - Recommended next action

## 3. Next sprint / active work
    - Theme + goal
    - Task checklist (unchecked)
    - Model tier

## 4. Warning zone — tech debt & traps
    - (seed with the 23 unresolved items from AUDIT_FINDINGS.md,
       ranked by severity, so every tech-debt concern has one home)

## 5. Operational contracts
    - Load-bearing invariants (no-lookahead, parameterized columns, etc.)

## Appendix A — Sprint handoff log (collapsed)
    - Sprint 1 — 2026-04-17 — sample-weight plumbing — ✓
    - Sprint 2 — 2026-04-18 — sequential bootstrap + uniqueness — ✓
    - ... (one line each, sprints 1–12)

## Appendix B — CLI edge cases (from DEBUG_LOG.md)
```

All the v0.5-era narrative (what's built, feature inventory, indicator
catalog, dual-signal principle, etc.) is removed from `PROJECT_STATE.md` —
it belongs in README / AGENTS / ARCHITECTURE, not in a living tracker.

---

## 5. New `CLAUDE.md` — thin pointer

```
# CLAUDE.md — Claude-specific context for AprilAlgo

> **This file is intentionally thin.** The canonical rules, command
> reference, testing guide, model routing, and session protocol live in
> `AGENTS.md`. Read that file first. Everything below is additive,
> Claude-only preference or context.

## Single source of truth

- Project rules / coding conventions / commands / tests: see `AGENTS.md`
- System design: see `ARCHITECTURE.md`
- Column contracts: see `docs/DATA_SCHEMA.md`
- Today's state + next sprint: see `PROJECT_STATE.md`
- Release history: see `CHANGELOG.md`

## Claude-specific preferences

- When in doubt about tool use, prefer edit-over-rewrite
- Surface TODOs as checklists in PROJECT_STATE.md §3 "Next sprint"
- Use ASCII for diagrams in ARCHITECTURE.md (no mermaid) — keeps them
  readable in terminal viewers
- On multi-file refactors, emit a short diff summary at the end

## Things Claude must not do

- (ported from the "NEVER" list in AGENTS.md — keep in sync by reading
  AGENTS.md at session start)
```

This gives Claude a dedicated file without duplicating the 100-line AGENTS
contract.

---

## 6. File-count delta (final)

| | Before | After |
|---|---:|---:|
| Active `.md` files in root | 10 | 6 |
| Active `.md` files in `docs/` | 7 | 3 |
| Archived `.md` files in `docs/archive/` | 0 | 2 |
| **Total markdown files** | **17** | **11** |
| **Files updated per session** | **9** | **2** |

The per-session update count is the key metric: `PROJECT_STATE.md` and
`CHANGELOG.md` are the only two files an agent routinely touches.

---

## 7. Migration steps (post-approval)

On your "go ahead", I will execute in this order:

1. **Create `docs/archive/`** and move `LEARNING.md` + `REPO_ANALYSIS.md` into it.
2. **Expand `AGENTS.md`** — merge `COMMAND_CHEAT_SHEET.md`, `docs/MODEL_ROUTING.md`,
   and `docs/SESSION_HANDOFF.md` "Cursor Handoff Protocol" section; add the new
   §"Testing" block.
3. **Rewrite `PROJECT_STATE.md`** to the §4 outline, seeding §"Warning zone"
   from `AUDIT_FINDINGS.md`, absorbing `DEBUG_LOG.md` as Appendix B, and
   collapsing the 12 sprint handoff entries into Appendix A one-liners.
4. **Rewrite `CLAUDE.md`** to the §5 thin-pointer form.
5. **Trim `ARCHITECTURE.md`** §2 "Versioned Roadmap" — remove shipped
   content, keep forward-looking placeholder.
6. **Update `README.md`** doc-link list to reflect the new tree.
7. **Search-and-fix cross-references** across the repo (`grep -r` for the
   deleted file names in `.md`, `.py` docstrings, YAML comments, etc.) —
   update or remove stale links.
8. **Delete the obsolete files**:
   - `BACKLOG.md`
   - `COMMAND_CHEAT_SHEET.md`
   - `DEBUG_LOG.md`
   - `tests.md`
   - `docs/SESSION_HANDOFF.md`
   - `docs/MODEL_ROUTING.md`
9. **Run `uv run pytest tests/ -q`** — confirm no test imports a deleted
   doc path or asserts on `tests.md` line counts.
10. **Log the change** in `CHANGELOG.md` `[Unreleased]` → "Changed: consolidated
    documentation from 17 → 11 files; per-session update surface is now
    `PROJECT_STATE.md` + `CHANGELOG.md`."

No code files will be modified at any step other than §7 (cross-reference
fixes, if any are discovered) and §10 (CHANGELOG).

---

## 8. Ready to execute?

If §1 "Final target structure" and §7 "Migration steps" match what you
want, reply with a go-ahead and I'll run the migration end-to-end and
return the test-suite status.

If you want to adjust anything first (e.g., keep `BACKLOG.md` as an
archive, change which sprint handoff entries survive, add/remove a bullet
in the `CLAUDE.md` preferences list), say so before I execute.
