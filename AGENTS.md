# AGENTS.md

## Repo structure

```
src/          # code
md/           # durable reference knowledge (second brain)
md/experiments/   # optional, explicitly requested human/agent analyses
md/instructions/  # standing instruction docs (DIAGRAMS.md, EXPERIMENT_TRACKING_SQLITE.md, EXPERIMENT_REPORTS.md, CLUSTER_INSTRUCTIONS.md + the remote-target docs it routes to)
md/DONE.md    # archive of completed TODO items
results/<id>/            # one folder per experiment: outputs, logs, artifacts (results/ is entirely gitignored)
results/<id>/report.html # its HTML report — regenerated locally in place, never tracked
results/<id>/<run_id>/   # one subdirectory per run, holding run.db
scripts/      # launch scripts; report core
configs/      # experiment YAMLs, one per experiment: <id>.yaml (optional)
data/         # inputs / datasets (optional)
AGENTS.md RESEARCH.md TODO.md EXPERIMENTS.md   # top-level md
```

- **`AGENTS.md`** — standing conventions and agent guidelines for the project. Treat as read-only during normal work: don't edit it unless the user explicitly asks.
- **`RESEARCH.md`** — the project and scientific framing: research questions, technical specifications, plan, and factual status. Running an experiment never triggers scientific synthesis or interpretation here.
- **`EXPERIMENTS.md`** — the lean execution index: one row per experiment (`id | date | question | server | status | human analysis | agent analyses`). Analysis cells contain links only and remain `—` until analysis is explicitly requested or supplied.
- **`TODO.md`** — an actionable queue of the project's TODOs, split into two sections, `Implementation` (codebase features) and `Experiments` (experiments-to-be-run). Items are functional bullets describing *what* we want (a behaviour, function, outcome, or experiment to run), at whatever level of detail is useful; written by the user, or by the agent during planning. Notation: `[ ]` planned/queued, `[~]` WIP, `[x]` done and verified — an item moves `[ ]` → `[~]` → `[x]` as work progresses. The `Uncategorized` inbox contains raw, unsorted captures (todos, notes, experiment ideas), not yet triaged into the sections above.
- **Before every non-trivial commit, make sure the following are current:**
  - `TODO.md`: keep it lean. When a feature is done, move a concise record of what was built (outcome plus key files touched) to `md/DONE.md` and leave a one-liner marked `[x]` in `TODO.md`.
  - `RESEARCH.md`: keep framing, plan, and factual project status current. Do not add analysis or synthesis merely because an experiment completed.
  - `EXPERIMENTS.md`: execution statuses reflect reality. Analysis links and HTML reports are independent and never prerequisites for completion.
  - Plan diagrams: verify the `mermaid` diagrams in `RESEARCH.md`, `EXPERIMENTS.md`, and `TODO.md` still match the text and status of their files (see [`DIAGRAMS.md`](md/instructions/DIAGRAMS.md)).
  - `md/`: any doc this change makes stale is updated or deleted in the same commit.

## Agent guidelines

- **Use plain English**: avoid technobabble and idiomatic expressions.
- **Design features with the user first.** When asked to implement a feature, start by interviewing the user to reach a shared understanding of a plan and specs *together*: present the available options and their trade-offs, then build from the chosen one. 
  - Scale planning effort to the feature size: the larger the feature, the more comprehensive the plan and specs; the more open-ended it is, the more the user should be involved in defining the features and specs. 
  - Ask **abundantly** the user to establish an aligned and fully defined plan. Record the agreed items as `[ ]` entries in `TODO.md` (outcome-centered bullets: what we want, not how to build it) before implementing.

- **Build only what was asked / what's in the plan — no feature creep.** If something unspecified seems necessary or useful, check with the user rather than just adding it.
- **Don't make assumptions: when anything is ambiguous or underspecified, check in with the user**. Don't be shy to **ask clarifying questions** to identify ambiguities, edge cases, underspecified behaviors, design preferences, and performance needs.
  - **"Godspeed"** — the keyword for an overnight/over-weekend run where the user can't reply. The plan is already agreed; lean towards working things out yourself within its scope rather than blocking on a question, keep a record of the decisions you take, and surface them when the user comes back.
- **Avoid over-engineering solutions; value simplicity and modularity.**
- **Be extra careful to avoid silent failures.**
- Use **subagents** whenever possible to delegate and parallelise work efficiently. Choose subagents modes based on task complexity: for trivial and simple tasks such as verifying if test pass, simple implementation, boilerplate, etc. use token-efficient models for the subagents: i.e., Claude's Sonnet or Codex's Terra / gpt-5.6-terra.
- Keep `TODO.md` up-to-date. Always verify that TODO.md is up-to-date before committing and PR code changes. Don't commit changes or PR if `TODO.md` has `Implementation` items marked as WIP `[~]`. `Experiments` items are exempt: a launched experiment stays `[~]` for the whole life of its run, and launching *requires* committing and pushing its live record to GitHub via `origin` — an in-flight `RUNNING` row in `EXPERIMENTS.md` never blocks a commit.
- `md/` is the project's second brain for durable reference knowledge (code and experiment scripts docs, methodology, model/hyperparameter rationale, dataset descriptions, preprocessing, workflows, agent instructions), not results or data (those live in `results/` and `data/`). Analysis is written only after a direct user request, never as routine experiment bookkeeping.
  - **`md/` is not a dump — every write must earn its place.** Put factual status in `RESEARCH.md`, execution state and optional analysis links in `EXPERIMENTS.md`, tasks in `TODO.md`, completed work in `md/DONE.md`, and outputs/data in `results/` and `data/`. Add something to `md/` only when you can name how and when it will be consulted again.
  - **Updating an existing `md/` file is autonomous — but keep it lean; creating a new one is not — ask the user first** (same rule as feature creep: if a new doc seems necessary, check rather than just adding it). The only exceptions are the transient `md/plan-<topic>.md` files from the TODO workflow below and explicitly requested `md/experiments/` analyses.
  - **Keep `md/` current: fix or flag stale docs on contact.** When consulting `md/`, if a doc contradicts the code or reality, correct it (or tell the user if unsure) — don't silently work around it. When a change makes a doc stale, update or delete that doc in the same commit (per the pre-commit checklist above).
  - Link syntax: always use plain markdown links (`[text](path)`) for cross-references, never bare `@file.md` refs — `@` refs auto-load the target into context every session in Claude Code, while Codex treats them as plain text. When a link climbs out of a subfolder (`../`, `../../`), add a brief locator note like "(at the repo root)".
- Match the user's own terms in notes, comments, commits, and docs; don't paraphrase, relabel, or "improve" their wording.

## TODO.md workflow

- Keep `TODO.md` live during work: mark an item `[ ]` when planned, `[~]` when you begin it, and `[x]` when it's done and verified, so the queue always reflects current state, not just at commit time. 
- Items in `TODO.md` define outcomes: what we want, not how to build it. The *how* (chosen approach, file sequence) stays internal to the session and is not a tracked artifact — with one exception: for a feature large enough that losing the session would lose the agreed design, offload the plan to a transient `md/plan-<topic>.md` and link it from the item, so a successor agent can pick up mid-implementation.
  - The plan file holds **only what was agreed with the user** — the scope and decisions from the planning discussion, at the level of detail they were actually discussed. Don't elaborate beyond that or improvise specifics the user never signed off on: plans feature-creep easily, and an over-detailed plan reads as agreed scope when it isn't.
  - Lifecycle (per the `md/` rule above): the plan file lives only while its item is `[ ]`/`[~]`; whoever completes the item deletes it, folding anything durable into `md/` proper or `md/DONE.md`.
- An item should only be marked as WIP `[~]` if, and only if, there is an active agent working on it (implementation or testing). An experiment stays `[~]` while its run executes and becomes `[x]` when execution ends; fetching, reporting, and analysis are separate and do not gate experiment completion.

- **Triage `Uncategorized`:** these are unsorted items. First sort each with the user into `Implementation` or `Experiments` as a functional bullet (or `RESEARCH.md`, or drop it), then plan it like any item before implementing. Don't implement straight from an unsorted capture.
- **"Address the next TODOs" ⇒ batch, don't bottleneck.** When the user asks to tackle the next TODO(s), default to picking a *handful* of orthogonal items and parallelizing them across subagents (using isolated worktrees when needed) — confirm *which* items with the user. **Each subagent marks its own item `[~]` on claim and `[x]` once done and verified**, so `TODO.md` stays an accurate live view of what's in flight.

## Plan diagrams

- Each plan carries a standing `mermaid` diagram: the research plan in `RESEARCH.md`, the experiments map in `EXPERIMENTS.md`, and the implementation roadmap in `TODO.md` — drawn and maintained per [`DIAGRAMS.md`](md/instructions/DIAGRAMS.md).
- Keep the diagrams, like `TODO.md`, up to date: update a diagram in the same edit as the plan/status text it reflects, so diagrams and text never drift apart.

## Experiments

- **Favour fully defining each experiment via a YAML config file.** The YAML should be enough to *fully specify* the experiment — every parameter, input, and setting needed to run and reproduce it — with no hidden state in code or the environment.
- **Agree every experiment name/id with the user during experiment definition.** Use the same semantic id in `configs/<id>.yaml`, `results/<id>/`, `EXPERIMENTS.md`, and the report manifest `scripts/reports/<id>.py`, authored at definition time with the report's control parameters chosen with the user — the config's parameters are candidates, not the answer (see [`EXPERIMENT_REPORTS.md`](md/instructions/EXPERIMENT_REPORTS.md)); an explicitly requested analysis, if any, uses `md/experiments/<date>-<id>.md`.
- **The user says where an experiment runs (machine, and GPU(s) if GPU-bound); within that, execution is the agent's:** batch to make full use of the indicated hardware, order and retry runs, keep bookkeeping — decide and report, don't ask. Ask only when execution can't follow the agreement (see the header of `EXPERIMENTS.md`).
- **Stamp code provenance at launch:** the launcher records the current git commit hash in the run's `run.db` metadata and saves a stamped copy of the YAML (with the hash) next to it in `results/<id>/<run_id>/`. Reproducing a run = check out that commit + run its YAML.
- **When defining an experiment, agree with the user whether it's exploratory ("run and see") or verifiable, and what each run records (see [`EXPERIMENT_TRACKING_SQLITE.md`](md/instructions/EXPERIMENT_TRACKING_SQLITE.md)); if verifiable, set an explicit success criterion and enforce it via a manual check/loop or the `/goal` feature.**
- **If the codebase runs simulations or trains models, track experiments as specified in [`EXPERIMENT_TRACKING_SQLITE.md`](md/instructions/EXPERIMENT_TRACKING_SQLITE.md).**
- **Track execution live in `EXPERIMENTS.md`:** add a `PLANNED` row when the experiment is defined; at launch set `server` and `RUNNING`; when execution ends set `COMPLETED`, `FAILED`, or `ABORTED`. These statuses describe execution only. Never populate analysis columns or infer findings as part of this lifecycle.
- **Analysis is explicit-only:** do not analyze, interpret, summarize findings, create `md/experiments/` files, or update analysis links because an experiment ran, finished, failed, was fetched, or was reported. Do so only after a direct user request to analyze named results.
- **HTML reports are explicit-only:** generate `results/<id>/report.html` with `generate-report` only after a direct user request. Reports are neutral, untracked projections of recorded results and are independent of analysis Markdown.
- **To run experiments/jobs on the cluster or remote servers, start at [`CLUSTER_INSTRUCTIONS.md`](md/instructions/CLUSTER_INSTRUCTIONS.md)** — it routes to the doc for the specific target (SSH box, ITU HPC, other clusters). Always use **subagents** to run and manage jobs in remote machines/cluster. Retrieve a finished experiment's runs with the `fetch-results` skill.

## Testing

- **Test-driven for code changes:** write the relevant test first, confirm it fails, then make it pass. Tests verify the specified features and specs — their integrity and functionality — not implementation trivia. Favour a few meaningful integration/smoke tests over many trivial unit tests.
  - Trust the run's own emitted outputs (logged metrics, saved artifacts, tracking records), don't re-derive them in the check; a check that recomputes the answer hides bugs in what the run actually produced.
  - Be explicit about what was *not* tested, which regimes, configs, seeds, scales; a result that holds for one setting often silently breaks in another.	
  
- **Verify with a fresh agent** — the agent that did the work doesn't grade it. After non-trivial implementation or revision of models, simulations or experiments (not minor or localized edits): spawn a clean-context subagent to confirm tests run and pass, the behaviour in `TODO.md` is really there, and the implementation matches the experiment's stated goal (not something adjacent that only looks right). For experiment runs: confirm the run genuinely happened — outputs, logs, and tracking records exist and match what was claimed.
- **Match test scope to the change.** For new experiment definitions/configs and minor or localized changes, run only the directly relevant tests or smoke checks — not the full codebase suite. Reserve the full suite for broad changes to shared code, explicit user requests, or required commit/CI gates.
- Use formatting, linting, and type-checking proportionately, scoped to the touched code when possible. Run the project-relevant checks before broad code changes or required gates — Python: `ruff format`, `ruff check --fix`, `pyrefly`; TS: `biome check --write`; Rust: `cargo fmt`, `cargo clippy -- -D warnings`; C++: `clang-format -i`, `clang-tidy`. Skip tools irrelevant to the files changed.

## Code style

- **Avoid default arguments in functions** unless told otherwise.
- **Avoid `try`/`except`** unless told otherwise — let errors surface.
- **Prefer typed code where the language supports it**: annotate params, returns, and fields, using the language's idiomatic typing (static annotations, hints, gradual typing). Favor typed over dynamic within a language, rather than switching languages to get it.
- **Python: use `uv`, never `pip`**, i.e., `uv add`, `uv sync`, `uv run`, etc.
- **Before a major commit or PR, consider whether the code (or part of it) needs a refactor** — run, or suggest the user run, `/code-review` and/or `/simplify`.

## Git

- **Tracked content moves only local checkout → GitHub over SSH → execution machine.** GitHub is the sole source of truth: never use HTTPS or a machine-local/bare Git remote, and never edit, commit, or push tracked files on an execution machine. Verify the URL, not the remote's name; anything else → **STOP and ask the user**.
- Commit messages are ONLY a one-liner with a high-level summary of the commit followed by bullet-point list of the changes made — nothing else. 
  - If the commit addresses an existing issue or PR, reference it in the message with `#<number>`.
- Split orthogonal changes into separate commits where possible.
- Work directly on `main` by default. Use a separate branch only when the user explicitly requests one; for a massive feature, ask whether they want a separate branch before starting.

## Secrets

- **Keep secrets (API keys, tokens) in `.env`, and keep `.env` in `.gitignore`.** For projects with CI/CD, store the keys as repository secrets.
