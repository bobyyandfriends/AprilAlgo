# AprilAlgo — documentation alignment audit

Date: 2026-04-18  
Scope: root `*.md`, active `docs/*.md` (excluding `docs/archive/`), `.cursorrules`, cross-checked against code and `PROJECT_STATE.md`.  
Canonical backlog: **`PROJECT_STATE.md` §3** (`BACKLOG.md` was removed in doc consolidation).

---

## Contradictions and stale content

### 1. `ARCHITECTURE.md` §2 “Next target — v0.6.0+ (candidate themes)” vs shipped code

**Doc says:** Themes 1–4 list backtest hardening (`metrics_v2`, margin/borrow), ML pipeline / `ml/pipeline.py`, indicator cleanups, UI subprocess safety as *future* considerations.

**Code / `PROJECT_STATE.md` say:** These items were implemented in the **2026-04-18** architectural sweep (`PROJECT_STATE.md` §2 handoff). `metrics_v2.py`, `ml/pipeline.py`, `Portfolio` margin/borrow, RSI `mode`, symmetric embargo option, and many UI/analyzer fixes are **already present**.

**Severity:** High for anyone using `ARCHITECTURE.md` as a roadmap without reading `PROJECT_STATE.md`.

**Recommended update:** Replace the four bullet “candidate themes” with a short pointer: *“v0.6+ themes are listed in `PROJECT_STATE.md` §3; the former candidate list here largely shipped in v0.5.x follow-up work (see `CHANGELOG.md` `[Unreleased]` / Fixed).”* Optionally add one bullet for **remaining** themes only (e.g. metrics_v2 **adoption** in reports/UI, default symmetric embargo, SHAP narrative UX).

---

### 2. `ARCHITECTURE.md` §2 last sentence vs `AUDIT_FINDINGS.md` / `PROJECT_STATE.md`

**Doc says:** *“the audit's unfixed architectural concerns are enumerated in `PROJECT_STATE.md` §4 … (seeded from `AUDIT_FINDINGS.md`).”*

**Current state:** `AUDIT_FINDINGS.md` §B states B1–B23 are **addressed**; `PROJECT_STATE.md` §4 now holds **design decisions** and follow-ups, not an open §B backlog.

**Severity:** Medium (agent confusion).

**Recommended update:** Reword to: *“Residual design decisions and follow-ups live in `PROJECT_STATE.md` §4; historical audit narrative remains in `AUDIT_FINDINGS.md`.”*

---

### 3. `CHANGELOG.md` references to `BACKLOG.md`

**Doc says:** Older entries mention `BACKLOG.md` (e.g. sprint hygiene).

**Repo:** `BACKLOG.md` **does not exist** (by design).

**Severity:** Low (historical changelog entries are fine); avoid **new** references.

**Recommended update:** None required for history; if adding new notes, cite `PROJECT_STATE.md` §3 only.

---

### 4. `AGENTS.md` §6 vs default pytest behavior (resolved this session)

**Previously:** Examples implied optional `--cov=aprilalgo` only.

**Now:** Default `uv run pytest tests/` runs coverage with **`--cov-fail-under=55`** per `pyproject.toml`; `AGENTS.md` §6.1 documents this and `--no-cov`.

**Severity:** Resolved.

---

### 5. `DOC_CONSOLIDATION_PLAN.md` vs current tree

**Doc:** Describes a planned consolidation including deleting `BACKLOG.md`.

**Repo:** Matches outcome; plan file is **historical**.

**Severity:** Low.

**Recommended update:** Optional one-line banner at top: *“Executed; retained for history.”*

---

### 6. `docs/HANDOFF.md` vs `PROJECT_STATE.md`

**Risk:** Two handoff surfaces can drift.

**Severity:** Medium if both are edited independently.

**Recommended update:** State in `HANDOFF.md` that **`PROJECT_STATE.md` §2 is canonical** for day-to-day handoff, or merge unique content and point to §2.

---

## Consistency checks (pass)

- **`.cursorrules`** — Defers to `AGENTS.md` on conflict; aligned.
- **`docs/DATA_SCHEMA.md`** — Still referenced from rules and `PROJECT_STATE`; no schema change in this session.
- **`README.md` links** — Point to `AGENTS.md`, `ARCHITECTURE.md`, `docs/DATA_SCHEMA.md`; still valid.
- **`CLAUDE.md`** — Thin pointer to `AGENTS.md`; aligned with consolidation intent.

---

## Link hygiene (spot check)

- Internal links in `ARCHITECTURE.md` / `README.md` use paths that exist.
- Do not link to `docs/archive/*` from “current” docs without labeling archived.

---

## Suggested consolidation order (when you next edit docs)

1. Refresh **`ARCHITECTURE.md` §2** (items above).  
2. Add a one-line banner to **`DOC_CONSOLIDATION_PLAN.md`** if desired.  
3. Clarify **`docs/HANDOFF.md`** vs **`PROJECT_STATE.md` §2**.  
4. Keep **`AUDIT_FINDINGS.md`** as historical + status banner; avoid duplicating §B text into new docs.
