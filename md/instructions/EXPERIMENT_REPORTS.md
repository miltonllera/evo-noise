# HTML experiment reports

Each experiment may have a single neutral HTML results view at
`results/<id>/report.html`, beside its run directories. It is generated from recorded results
only when the user explicitly asks for a report—for example, “generate the report for
`<id>`” or `$generate-report <id>`. Experiment completion and result fetching do not generate
it automatically.

The report behaves like a local W&B dashboard: it presents recorded configs, metrics,
figures, artifacts, and computed comparisons without interpreting them. It contains no
verdict, conclusion, recommendation, causal claim, or qualitative judgement such as one
condition being “better” or “worse.” Captions describe what is shown and, when useful, how it
was computed. When a report covers a batch or sweep of runs, the sticky filter menu at the
top includes a control for how many to display, defaulting to the top few — what they are
ranked by and the default count are agreed in the interview below.

## Source and ownership

| layer | owns | never contains |
|---|---|---|
| `results/<id>/<run_id>/run.db` | recorded observations at full precision | hand-edited results |
| `results/<id>/report.html` | figures, interactive comparisons, numeric tables | scientific interpretation |

The report depends only on `run.db`, digest-recorded artifacts, the experiment config, and
the repo’s report generator/manifests. It must not require, link to, or import content from an
analysis Markdown file. An optional Mermaid diagram may be declared in the report manifest
from the config or recorded experimental structure; it neutrally describes pipeline or
topology and contains no conclusions.

Nothing in `results/` is tracked. Reports are regenerated, never hand-written or committed,
and no number is retyped into them. Building a report never edits results or runs experiment
code; missing evidence produces an explicit skipped item.

## Generator

The **`generate-report` skill** runs only after a direct report request. It reads this file,
uses the repo’s existing generator, and writes `results/<id>/report.html`. If no generator
exists, it offers to vendor its versioned starter into `scripts/` after the user supplies the
project specifics below.

The generator has one shared core/template, reusable section builders, and thin
per-experiment manifests. Section builders stay generic; which ones an experiment uses, on
which data and with which settings, is declared in its manifest. When an experiment needs
new content: use an existing function if it already fits; extend one only when the extension
provably leaves its output for existing manifests unchanged (new optional parameter, current
default); otherwise write a new function. Older reports must stay regenerable without
re-testing them after every change. Derived figures and animations are rendered from
recorded data; a file is embedded directly only when the run wrote it and logged its sha256
in the `run.db` artifacts table (see
[`EXPERIMENT_TRACKING_SQLITE.md`](EXPERIMENT_TRACKING_SQLITE.md)) — anything without that
record is not evidence.

Before the project’s first report, ask the user:

- Which factors are toggleable, and which are single-select?
- Which metric is primary and which are secondary?
- Which figures and sections belong in every report, and which to single experiments?
- What project-specific units or number formatting apply?
- For batches or sweeps of runs, what are the top results ranked by, and how many show by default?

## Project specifics

<!-- Filled in at first use, per "Customise before the first report" above.
     Until then, an agent must STOP and interview the user rather than guess. -->

- **Toggleable factors (single-select marked):** —
- **Primary metric:** —
- **Secondary metrics:** —
- **Standard figure set:** —
- **Number formatting overrides:** —
