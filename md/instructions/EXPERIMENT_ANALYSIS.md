# Organising experiment analyses

A reusable pattern for recording experiment analyses so **RESEARCH.md stays high level**
and per-experiment detail is stored in files agents fetch **on demand** — discoverable
via a lean index, never loaded into context by default.

## The idea in one line

RESEARCH.md holds only cross-experiment synthesis; each experiment gets one analysis
file in `md/experiments/`; a lean top-level `EXPERIMENTS.md` index makes past work
discoverable without loading it; `results/<id>/` holds that experiment's raw artifacts
(run.db, logs, recorded media) plus its regenerable HTML report.

## Experiment ids

An experiment's `<id>` is **semantically informative**: one to three hyphenated words saying
what the experiment did — `field-sparsity`, `hidden-trajectory-fieldfree` — never a bare
number or a random codename.

- A trailing number means only "a repeat of that experiment" — at most different
  hyperparameters, conceptually the same: `field-sparsity-2`.
- A *new* experiment whose natural name would collide with an existing one adds a word naming
  the difference (`field-sparsity-anneal`), not a counter.
- The id is **the same string everywhere it appears**: `configs/<id>.yaml`, `results/<id>/`,
  the `EXPERIMENTS.md` row, the analysis file `md/experiments/<YYYY-MM-DD>-<id>.md`, and the
  report title. One experiment, one name.

## The three layers

1. **`RESEARCH.md` — synthesis only.** What the completed experiments *collectively* say
   about the research questions. No per-run detail, no per-experiment write-ups. It links
   to [`EXPERIMENTS.md`](../../EXPERIMENTS.md) (at the repo root) for anything deeper.
2. **`EXPERIMENTS.md` (top level, next to RESEARCH.md/TODO.md) — the index.** One table
   row per experiment: `id | date | question | server | one-line finding | file`. Lean
   enough that reading it is always cheap. This is the entry point for "what have we
   already done?".
3. **`md/experiments/<YYYY-MM-DD>-<id>.md` — one file per experiment**, covering all
   its runs, sweeps, and iterations: hypothesis/question, setup (config file, run_ids
   pointing at `results/<id>/<run_id>/run.db` — the run's canonical record, see
   [`EXPERIMENT_TRACKING_SQLITE.md`](EXPERIMENT_TRACKING_SQLITE.md)), findings, and
   conclusion. The conclusion's one-liner is mirrored into the `EXPERIMENTS.md` row.

Each experiment also gets a **visual companion**: `results/<id>/report.html`, sitting in the
experiment's own results folder beside the runs it was built from, generated from `run.db` per
[`EXPERIMENT_REPORTS.md`](EXPERIMENT_REPORTS.md). It is a regenerated artifact, not a tracked
one — `results/` is gitignored, and the `generate-report` skill rebuilds it locally whenever
it's missing or stale. The
analysis file owns the argument; the report owns the figures and the interactive comparison.
Keep figures and raw numeric tables out of the analysis file — link the report instead.

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

Experiments have a live status, mirroring TODO notation: planned (a `[ ]` item in
`TODO.md`'s `Experiments` section) → running (a `RUNNING` row in `EXPERIMENTS.md`, the
item `[~]`) → completed (the row carries the one-line finding, the item `[x]`).

When an experiment **launches** (locally or remotely):

1. Add its row to `EXPERIMENTS.md` immediately: the `server` column set to
   `<target> (<job/session id>)` — the machine (its `~/.ssh/config` name) or cluster,
   precise enough that the `fetch-results` skill knows where to pull `results/<id>/`
   from — the finding column set to `RUNNING — launched <YYYY-MM-DD>`, and `—` in the
   file column until the analysis exists. The row is the live record of what is running
   where.
2. Tag its node `:::wip` in the experiments map, and keep its `TODO.md` item `[~]` for as
   long as the run executes — a running remote job counts as work in progress even with
   no agent attached.
3. Commit the row update — a `RUNNING` marker is only useful if other agents and machines
   can see it. In-flight experiment rows never block a commit (unlike `Implementation`
   WIP items — see AGENTS.md).

When an experiment completes:

1. Write (or update) its `md/experiments/<YYYY-MM-DD>-<id>.md` file.
2. Update its row in `EXPERIMENTS.md`: the `RUNNING` marker in the finding column gives
   way to the one-line finding, and the file column gets the analysis link.
3. Generate (or regenerate) its HTML report with the `generate-report` skill (pulling remote
   runs first with `fetch-results` if needed) and check the verdict box matches the
   conclusion one-liner verbatim.
4. Update `RESEARCH.md` **only if the synthesis changes** — i.e. the result shifts what
   we believe about a research question, not merely adds a data point.

Failed or abandoned experiments are indexed too: keep the row, prefix its one-line
finding with `ABORTED —` plus why (and record what's known in the analysis file). A
recorded dead end is a re-run avoided.

Before designing or re-running an experiment, check `EXPERIMENTS.md` for prior related
work instead of re-deriving it.

## Per-experiment file template

```markdown
# <experiment title>

- **Question / hypothesis:** what this experiment tests and why.
- **Setup:** config file (`configs/<id>.yaml`), run_ids
  (e.g. `results/<id>/<run_id>/run.db`), and anything needed to reproduce. The server
  it runs on is in the index row's `server` column.
- **Report:** [`results/<id>/report.html`](../../results/<id>/report.html) — the
  figures and the interactive comparison. Gitignored; if the link dangles, rebuild it
  with the `generate-report` skill (after `fetch-results` if the runs are remote).

## Findings

Per-run / per-sweep observations and key metrics. Figures live in the report, not here —
link to it (a report URL carries its control state in the hash, so it can point at exactly
the comparison being discussed).

## Conclusion

What the experiment tells us about the question. End with the one-line finding
mirrored in EXPERIMENTS.md.
```
