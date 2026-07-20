# AGENTS.md

## Repo structure

```
src/          # code
md/           # durable reference knowledge (second brain)
results/      # experiment outputs, logs, artifacts
scripts/			# bash/python scripts to launch experiments (optional)
configs/      # experiment YAMLs (optional)
data/         # inputs / datasets (optional)
AGENTS.md/CLAUDE.md RESEARCH.md TODO.md   # top-level md
```

- **`AGENTS.md`** — standing conventions and agent guidelines for the project. Treat as read-only during normal work: don't edit it unless the user explicitly asks.
- **`RESEARCH.md`** — the project and scientific framing: research questions, technical specifications, current status, and the analysis of experiment results. As the project grows, new information goes into a specific `md/` file and is cross-referenced here via `@` (i.e. `@md/filename.md`).
- **`TODO.md`** — an actionable queue of the project's TODOs, split into two sections, `Implementation` (codebase features) and `Experiments` (experiments-to-be-run). Items are functional bullets describing *what* we want (a behaviour, function, outcome, or experiment to run), at whatever level of detail is useful; written by the user, or by the agent during planning. Notation: `[ ]` planned/queued, `[~]` WIP, `[x]` done and verified — an item moves `[ ]` → `[~]` → `[x]` as work progresses. The *how* (chosen approach, file sequence) stays internal to the session and is not a tracked artifact. The `Uncategorized` inbox contains raw, unsorted captures (functional points, notes, experiment ideas), not yet triaged into the sections above.
- **Before every non-trivial commit, make sure both are current:**
  - `TODO.md`: keep it lean. When a feature is done, move a concise record of what was built (outcome plus key files touched) to `DONE.md` and leave a one-liner marked `[x]` in `TODO.md`.
  - `RESEARCH.md`: keep status, experiment results, and project direction up to date; keep it lean, don't bloat it. If unsure whether it needs an update, ask the user.

## Agent guidelines

- **Design features with the user first.** When asked to implement a feature, start by interviewing the user to reach a shared understanding a plan and specs *together*: present the available options and their trade-offs, then build from the chosen one. Scale planning effort to the feature size: the larger the feature, the more comprehensive the plan and specs; the more open-ended it is, the more the user should be involved in defining the features and specs. Ask **abundantly** the user to establish an aligned and fully defined plan. **Record the agreed items as `[ ]` entries in `TODO.md` (functional bullets: what we want, not how to build it) before implementing.**
- **Build only what was asked / what's in the plan — no feature creep.** If something unspecified seems necessary or useful, check with the user rather than just adding it.
- **Don't make assumptions: when anything is ambiguous or underspecified, check in with the user**. Don't be shy to **ask clarifying questions** to identify ambiguities, edge cases, underspecified behaviors, design preferences, and performance needs.
- **Avoid over-engineering solutions; value simplicity and modularity.**
- **Be extra careful to avoid silent failures.**
- **Use subagents whenever possible** to delegate and parallelise work efficiently.
- Keep RESEARCH.md lean: `md/` is the project's second brain for durable reference knowledge (code and experiment scripts docs, methodology, model/hyperparameter rationale, dataset descriptions, preprocessing, workflows, agent instructions), not results or data (those live in `results/` and `data/`). Record what's worth keeping long-term and consulting again, not transient or one-off detail; consult `md/` before re-deriving something that may already be documented. Put new detail in a specific `md/` file and cross-reference it with `@` (i.e. `@md/filename.md`) in RESEARCH.md. 
- Match the user's own terms in notes, comments, commits, and docs; don't paraphrase, relabel, or "improve" their wording.

## TODO.md workflow

- Keep `TODO.md` live during work: mark an item `[ ]` when planned, `[~]` when you begin it, and `[x]` when it's done and verified, so the queue always reflects current state, not just at commit time. Always verify that TODO.md is up-to-date before committing and PR code changes.
- **Triage `Uncategorized`:** these are unsorted items. First sort each with the user into `Implementation` or `Experiments` as a functional bullet (or `RESEARCH.md`, or drop it), then plan it like any item before implementing. Don't implement straight from an unsorted capture.
- **"Address the next TODOs" ⇒ batch, don't bottleneck.** When the user asks to tackle the next TODO(s), default to picking a *handful* of orthogonal items and parallelizing them across subagents (worktree-isolated, one branch/PR each) — confirm *which* items with the user. **Each subagent marks its own item `[~]` on claim and `[x]` once done and verified**, so `TODO.md` stays an accurate live view of what's in flight.

## Experiments

- **Favour fully defining each experiment via a YAML config file.** The YAML should be enough to *fully specify* the experiment — every parameter, input, and setting needed to run and reproduce it — with no hidden state in code or the environment.
- **When defining an experiment, agree with the user whether it's exploratory ("run and see") or verifiable; if verifiable, set an explicit success criterion and enforce it via a manual check/loop or the `/goal` feature.**
- **If the codebase runs simulations or trains models, track experiments as specified in [`EXPERIMENT_TRACKING_SQLITE.md`](EXPERIMENT_TRACKING_SQLITE.md).**
- **To run experiments/jobs on the cluster or remote servers, follow [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md).**

## Testing

- **Test-driven: tests verify the specified features and specs** — their integrity and functionality — not implementation trivia. Favour a few meaningful integration/smoke tests over many trivial unit tests.
  - Trust the run's own emitted outputs (logged metrics, saved artifacts, tracking records), don't re-derive them in the check; a check that recomputes the answer hides bugs in what the run actually produced.
  - Be explicit about what was *not* tested, which regimes, configs, seeds, scales; a result that holds for one setting often silently breaks in another.	
  
- **Verify with a fresh agent**. When implementing or revising models, simulations or experiments: spawn a clean-context subagent to confirm it works and actually does what the experiment intends, tests run and pass, the behaviour in `TODO.md` is really there, and the implementation matches the experiment's stated goal (not something adjacent that only looks right). For experiment runs: confirm the run genuinely happened, outputs, logs, and tracking records exist and match what was claimed, rather than taking the agent's word. The agent that did the work doesn't grade it.
- Run the integration and smoke tests before any large commit or PR.
- Before non-trivial commits, format + lint + typecheck and ensure all pass clean — Python: `ruff format`, `ruff check --fix`, `pyrefly`; TS: `biome check --write`; Rust: `cargo fmt`, `cargo clippy -- -D warnings`; C++: `clang-format -i`, `clang-tidy`.

## Code style

- **Avoid default arguments in functions** unless told otherwise.
- **Avoid `try`/`except`** unless told otherwise — let errors surface.
- **Prefer typed code where the language supports it**: annotate params, returns, and fields, using the language's idiomatic typing (static annotations, hints, gradual typing). Favor typed over dynamic within a language, rather than switching languages to get it.
- **Python: use `uv`, never `pip`**, i.e., `uv add`, `uv sync`, `uv run`, etc.
- **Before a major commit or PR, consider whether the code (or part of it) needs a refactor** — run, or suggest the user run, `/code-review` and/or `/simplify`.

## Git

- **Commit messages are ONLY a one-liner with a high-level summary of the commit followed by bullet-point list of the changes made — nothing else.**
- Substantially large features go on their own branch and are PR'd into `main`.** If unsure whether something needs its own branch or can go straight to `main`, ask.

## Secrets

- **Keep secrets (API keys, tokens) in `.env`, and keep `.env` in `.gitignore`.** For projects with CI/CD, store the keys as repository secrets.
