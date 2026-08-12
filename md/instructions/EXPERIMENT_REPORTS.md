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
was computed.

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

The expected generator has one shared core/template, reusable section builders, and optional
thin per-experiment manifests. Derived figures and animations are rendered from recorded
data; files are embedded only when the file itself is experimental evidence and its digest
was recorded.

Before the project’s first report, ask the user:

- Which factors are toggleable, and which are single-select?
- Which metric is primary and which are secondary?
- What standard figures should every report show?
- What project-specific units or number formatting apply?

## Project specifics

<!-- Filled in at first use, per "Customise before the first report" above.
     Until then, an agent must STOP and interview the user rather than guess. -->

- **Toggleable factors (single-select marked):** —
- **Primary metric:** —
- **Secondary metrics:** —
- **Standard figure set:** —
- **Number formatting overrides:** —
