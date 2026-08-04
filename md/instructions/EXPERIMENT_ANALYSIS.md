# Organising experiment analyses

A reusable pattern for recording experiment analyses so **RESEARCH.md stays high level**
and per-experiment detail is stored in files agents fetch **on demand** — discoverable
via a lean index, never loaded into context by default.

## The idea in one line

RESEARCH.md holds only cross-experiment synthesis; each experiment gets one analysis
file in `md/experiments/`; a lean top-level `EXPERIMENTS.md` index makes past work
discoverable without loading it; `results/` holds only raw artifacts (run.db, logs, plots).

## The three layers

1. **`RESEARCH.md` — synthesis only.** What the completed experiments *collectively* say
   about the research questions. No per-run detail, no per-experiment write-ups. It links
   to [`EXPERIMENTS.md`](../../EXPERIMENTS.md) (at the repo root) for anything deeper.
2. **`EXPERIMENTS.md` (top level, next to RESEARCH.md/TODO.md) — the index.** One table
   row per experiment: `id | date | question | one-line finding | file`. Lean enough that
   reading it is always cheap. This is the entry point for "what have we already done?".
3. **`md/experiments/<YYYY-MM-DD>-<slug>.md` — one file per experiment**, covering all
   its runs, sweeps, and iterations: hypothesis/question, setup (config file, run_ids
   pointing at `results/<run_id>/run.db` — the run's canonical record, see
   [`EXPERIMENT_TRACKING_SQLITE.md`](EXPERIMENT_TRACKING_SQLITE.md)), findings, and
   conclusion. The conclusion's one-liner is mirrored into the `EXPERIMENTS.md` row.

## Context semantics: plain links only

Cross-reference with **plain markdown links** (`[text](path)`) — they behave the same
in every agent: inert pointers, fetched on demand. Never use bare `@path` refs: in
Claude Code they auto-load their target into context every session when the referencing
file is in the AGENTS.md import chain (recursively), while Codex treats them
as plain text — so a bare `@` ref either bloats context or silently does nothing,
depending on the agent.

- To consult past work: read `EXPERIMENTS.md` first, then Read only the relevant
  `md/experiments/` file(s). Don't load all analyses into context.

## Workflow

When an experiment completes:

1. Write (or update) its `md/experiments/<YYYY-MM-DD>-<slug>.md` file.
2. Add or update its row in `EXPERIMENTS.md`.
3. Update `RESEARCH.md` **only if the synthesis changes** — i.e. the result shifts what
   we believe about a research question, not merely adds a data point.

Before designing or re-running an experiment, check `EXPERIMENTS.md` for prior related
work instead of re-deriving it.

## Per-experiment file template

```markdown
# <experiment title>

- **Question / hypothesis:** what this experiment tests and why.
- **Setup:** config file(s) (e.g. `configs/<name>.yaml`), run_ids
  (e.g. `results/<run_id>/run.db`), and anything needed to reproduce.

## Findings

Per-run / per-sweep observations, key metrics, plots worth referencing
(link into `results/`).

## Conclusion

What the experiment tells us about the question. End with the one-line finding
mirrored in EXPERIMENTS.md.
```
