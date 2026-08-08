# HTML experiment reports

Each experiment's results are presented as a **single self-contained HTML page** — lean,
concise, with toggles to swap between experimental conditions the way a W&B dashboard does.
One report per experiment, generated from `run.db`, never hand-written — and never tracked:
like every other projection of the run records, it is rebuilt locally on demand.

This file holds the project-facing conventions and this project's specifics. The full design
doctrine (report anatomy, interactivity, numbers policy, template contract, payload shape)
travels with the **`generate-report` skill**, which applies it when scaffolding or adapting
the generator; the vendored generator in `scripts/` embodies it, and its docstrings state
each part's boundaries.

## The idea in one line

`results/<id>/<run_id>/run.db` is the **record**, `md/experiments/<date>-<id>.md` is the
**argument**, and `results/<id>/report.html` is the **evidence you can look at and poke** —
regenerated from the record whenever needed, never committed.

## Who owns what

The three layers must not duplicate each other. If a fact appears in two of them, one of them
is wrong later.

| layer | owns | never contains |
|---|---|---|
| `results/<id>/<run_id>/run.db` | every observation, at full precision | anything hand-edited |
| `md/experiments/<date>-<id>.md` | question, setup, findings, conclusion — the prose of record | figures, raw numeric tables |
| `results/<id>/report.html` | figures, interactive comparison, numeric tables | the argument (it links to the analysis) |

A report is not a substitute for the analysis file. It shows *what happened*; the analysis
says *what it means*. A reader who wants the reasoning follows the link. The report's verdict
line is the `EXPERIMENTS.md` row's one-line finding, read verbatim — one string, three places.

## Location and naming

**The report lives with the experiment it describes**, not in a separate reports tree —
`results/<id>/report.html`, in the same folder as the runs it was built from. One folder
per experiment, everything about that experiment in it:

```
results/                  # regenerable output — entirely gitignored
  <id>/
    report.html           # the report — rebuilt locally, never committed
    <run_id>/run.db       # one subdirectory per run
    <run_id>/run.db
```

- `<id>` is the experiment's **semantic id** (see
  [`EXPERIMENT_ANALYSIS.md`](EXPERIMENT_ANALYSIS.md)): one to three words naming what the
  experiment did — `field-sparsity`, `hidden-trajectory-fieldfree` — never a bare number.
  The same string names `configs/<id>.yaml`, `results/<id>/`, the analysis file, the
  [`EXPERIMENTS.md`](../../EXPERIMENTS.md) row (at the repo root) and the report title, so
  every artifact of one experiment carries one name.
- **Nothing in `results/` is tracked.** The report, any assets beside it, and the run records
  are all regenerable — and reports that embed the animations a simulation-heavy experiment
  produces are far too large for git, which doesn't carry the raw results they project either.
  One gitignore pattern covers it:

  ```gitignore
  results/
  ```

## Generated, never hand-written

Reports are built by the **`generate-report` skill**. It uses the generator the repo already
has; if none exists, it offers to vendor its versioned starter (stamped `SCAFFOLD_VERSION`)
into `scripts/` and adapt it together with the user through the
[Customise before the first report](#customise-before-the-first-report) interview. The
vendored copy is committed and owned by the repo — a clone must regenerate its reports with
no skill installed. When the skill's canonical starter later improves, it may offer a
supervised update; the repo's copy never changes unasked.

**No agent ever retypes a number into a report.** Transcribed values drift from `run.db` the
moment a metric is fixed or a run is extended; a generated value cannot. Regenerating after a
metric fix, a new figure, or a relabelled condition is a re-run of the script, not of the
experiment.

## The generator in `scripts/`

Once vendored, the generator has three parts — their docstrings state the boundaries:

- **`scripts/make_report.py` + `scripts/report_template.html` — the core**, the project's
  house style, written once: read-only record access, digest-verified evidence, page shell,
  shared controls, URL-hash state, one number format, the runs table, offline assets
  (vendored into `scripts/assets/`, tracked), CLI.
- **`scripts/reports/sections.py` — the section library**: reusable section builders, shared
  by every experiment. Report content accumulates here, not in per-experiment files.
- **`scripts/reports/<id>.py` — one thin manifest per experiment**: factors, figure set, and
  its `SECTIONS` composed from the library in order. An experiment the defaults cover needs
  no manifest at all.

Extending a report means composing or growing the library through a manifest — not writing
another builder script, re-running a model, or embedding a saved image of something the
record can render (**embed inputs, render derivations**). Git carries what regeneration
needs — the generator, its template, the section library and the manifests, the vendored
libraries, configs and analyses — and never what it produces.

## Customise before the first report

**The figure set and the toggleable factors are project-specific — this doc deliberately does
not guess them.** Before generating a project's *first* report, STOP and ask the user:

- Which factors are toggleable, and are any single-select? (condition, model, dataset, seed…)
- Which metric is primary, and which are secondary?
- What is the standard figure set every experiment in this project should show?
- Is a diagram warranted, and does it belong in the analysis md?
- Any project-specific number formatting (units, percentages, fixed budgets)?

Record the answers in [`## Project specifics`](#project-specifics) at the bottom of this
file. Later reports follow that section without asking again. Revisit it only when the user
says the project's shape has changed.

## Workflow

When an experiment completes, after its analysis file exists (per
[`EXPERIMENT_ANALYSIS.md`](EXPERIMENT_ANALYSIS.md)):

1. If the runs happened remotely, pull them back with the **`fetch-results` skill** — it
   reads the `server` column in the experiment's `EXPERIMENTS.md` row (per
   [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md)) and syncs
   `results/<id>/` to the local machine.
2. Generate the report with the **`generate-report` skill** — it writes
   `results/<id>/report.html`, beside that experiment's run directories.
3. Check the verdict box matches the analysis conclusion and the `EXPERIMENTS.md` row verbatim.
4. Link it from the analysis file's `**Report:**` line. The report is gitignored, so the link
   dangles on a fresh clone — whoever follows it regenerates the report with the same two
   skills.

Regenerate whenever the runs, the metrics, or the analysis file's diagram change. The report
is a projection — it is always cheaper to rebuild than to patch. A generated report is one
portable file (data and libraries inlined): to share it outside the repo, send the file as-is.

## Project specifics

<!-- Filled in at first use, per "Customise before the first report" above.
     Until then, an agent must STOP and interview the user rather than guess. -->

- **Toggleable factors (single-select marked):** —
- **Primary metric:** —
- **Secondary metrics:** —
- **Standard figure set:** —
- **Number formatting overrides:** —
