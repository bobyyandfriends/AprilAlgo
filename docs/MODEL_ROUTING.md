# Model routing (Cursor agents)

Use this file **before** starting any new task. Match the task to a **tier**, then use the **model slug** for that tier when launching agents or choosing a chat model. Exact slugs may change with Cursor releases; update this table when your workspace model list changes.

## Tiers

| Tier | When to use | Typical model slugs (pick what your Cursor lists) |
|------|-------------|-----------------------------------------------------|
| **Fast** | Docs, YAML, version bumps, mechanical edits, Jinja templates, regenerating inventories | `composer-2-fast`; fallback `composer-2` |
| **Standard** | Normal coding against a fixed spec; tests; CLI wiring; Streamlit pages; single-module changes | `claude-4.6-sonnet-medium-thinking`; fallback `gpt-5.4-medium` |
| **Heavy** | Public APIs other modules import; trainer or logger **schemas**; no-lookahead strategy paths; non-trivial math (CV, sampling, meta-labels) | `claude-opus-4-7-thinking-high`; fallback `gpt-5.3-codex` |

Illustrative mapping (same idea, different vendors): reasoning-first models for heavy math/ML logic; Sonnet-class models for feature implementation; fast models for boilerplate—**always prefer the slug in the table above** over vendor marketing names.

## Heuristics

- New **function signatures** or **artifacts** imported across packages → **Heavy**.
- Touches **time order**, **labels**, **purged CV**, or strategy **`on_bar`** → **Heavy**.
- Single module, tests already specified, low coupling → **Standard**.
- Pure prose, config, or template edits → **Fast**.
- Estimated diff **> ~400 lines** or **≥ 4 files** → bump **one** tier.
- If a **Fast** run fails the same test **twice** → retry at **Standard**; if **Standard** fails twice on logic-heavy code → switch to **Heavy**.

## Sprint todo routing (AprilAlgo v0.3 / v0.4 — historical)

| ID | Tier | Primary slug |
|----|------|----------------|
| T0 | Fast | `composer-2-fast` |
| T1 | Standard | `claude-4.6-sonnet-medium-thinking` |
| T2 | Heavy | `claude-opus-4-7-thinking-high` |
| T3 | Standard | `claude-4.6-sonnet-medium-thinking` |
| T4 | Fast | `composer-2` |
| T5 | Standard | `claude-4.6-sonnet-medium-thinking` |
| T6 | Heavy | `claude-opus-4-7-thinking-high` |
| T7 | Standard | `claude-4.6-sonnet-medium-thinking` |
| T8 | Standard | `claude-4.6-sonnet-medium-thinking` |
| T9 | Fast | `composer-2-fast` |
| U1 | Heavy | `claude-opus-4-7-thinking-high` |
| U2 | Heavy | `gpt-5.3-codex` or `claude-opus-4-7-thinking-high` |
| U3 | Standard | `claude-4.6-sonnet-medium-thinking` |
| U4 | Standard | `claude-4.6-sonnet-medium-thinking` |
| U5 | Standard | `claude-4.6-sonnet-medium-thinking` |
| U6 | Fast | `composer-2` |
| U7 | Standard | `claude-4.6-sonnet-medium-thinking` |
| U8 | Fast | `composer-2-fast` |

## Scope shift

If work moves from one tier to another mid-task, **stop**, tell the user, and recommend the new tier and slug—do not continue with an under-matched model on Heavy-tier work.
