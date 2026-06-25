# AGENTS.md — working conventions

Shared conventions for working across these repos. A project's own `CLAUDE.md` /
`AGENTS.md` may add to or override these; when it does, the more specific file wins.

First time set-up: Run `ln -sf AGENTS.md CLAUDE.md` so all content lives in `AGENTS.md`. Then delete this line. 

All top-level markdown files containing code documentation, workflows or agent instructions should live inside a md/ folder and be cross-references (@md/filename.md) in AGENTS.md. Only md allowed in the top-level path are: AGENTS.md, CLAUDE.md, CODEX.md, RESEARCH.md and TODO.md; sub-folders may have markdown in them. Delete this line one verified that the md live in the right folders.

## Agent behaviour

- **Never assume — when anything is ambiguous, check in with the user** before acting.
- **Design features with the user first.** When asked to implement a feature, start by
  agreeing on a plan and specs *together*: present the available options and their
  trade-offs, then build from the chosen one. Scale the effort to the feature — the larger
  it is, the more comprehensive the plan and specs; the more open-ended it is, the more the
  user should be involved in defining the specs.
  - **For a very large project, ask whether they want to use OpenSpec to plan it.** If so,
    have them run `openspec init` in the repo, then `/opsx:propose "your idea"` in the
    coding agent. (Requires a prior global install: `npm install -g @fission-ai/openspec@latest`.)
- **Build only what was asked / what's in the plan — no feature creep.** If something
  unspecified seems necessary or useful, check with the user rather than adding it.
- **Avoid over-engineering solutions; value simplicity.**
- **Be extra careful to avoid silent failures.**
- **Use subagents whenever possible** to delegate and parallelise work.

## Planning & docs

- **`TODO.md`** — an actionable queue of the project's TODOs, split into two sections,
  `Implementation` (codebase features) and `Experiments` (experiments-to-be-run). Notation:
  `[ ]` todo, `[~]` WIP, `[x]` done.
- **`RESEARCH.md`** — the scientific framing: research questions, current status, and the
  analysis of experiment results.
- **Before every non-trivial commit, make sure both are current:**
  - `TODO.md`: keep it lean. When a feature is done, move its detailed completed plan to
    `DONE.md` and leave a concise one-liner marked `[x]` in `TODO.md`.
  - `RESEARCH.md`: keep status, experiment results, and project direction up to date; keep
    it lean, don't bloat it. If unsure whether it needs an update, ask the user.

## Experiments

- **Favour fully defining each experiment via a YAML config file.** The YAML should be
  enough to *fully specify* the experiment — every parameter, input, and setting needed to
  run and reproduce it — with no hidden state in code or the environment.
- **When defining an experiment, agree with the user whether it's exploratory ("run and
  see") or verifiable; if verifiable, set an explicit success criterion and enforce it via
  a manual check/loop or the `/goal` feature.**
- **If the codebase runs simulations or trains models, track experiments as specified in
  [`EXPERIMENT_TRACKING_SQLITE.md`](EXPERIMENT_TRACKING_SQLITE.md).**
- **To run experiments/jobs on the cluster or remote servers, follow
  [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md).**

## Testing

- **Test-driven: tests verify the specified features and specs** — their integrity and
  functionality — not implementation trivia. Favour a few meaningful integration/smoke
  tests over many trivial unit tests.
- **Verify with a separate agent, not the implementer.** After implementation, spawn a
  fresh subagent to check the work against the agreed plan/specs and the tests for those
  specs — the agent that did the work shouldn't be the one that grades it.
- **Run the integration and smoke tests before any large commit or PR.**
- Before any commit, format + lint + typecheck and ensure all pass clean — Python: `ruff format`, `ruff check --fix`, `pyright`; TS: `biome check --write`; Rust: `cargo fmt`, `cargo clippy -- -D warnings`; C++: `clang-format -i`, `clang-tidy`.

## Code style

- **Avoid default arguments in functions** unless told otherwise.
- **Avoid `try`/`except`** unless told otherwise — let errors surface.
- **Python: use `uv`, never `pip`**, i.e., `uv add`, `uv sync`, `uv run`, etc.
- **Before a major commit or PR, consider whether the code (or part of it) needs a
  refactor** — run, or suggest the user run, `/code-review` and/or `/simplify`.

## Git

- **Commit messages are ONLY a bullet-point list of the changes made — nothing else.**
  Split orthogonal changes into separate commits where possible.
- **Substantially large features go on their own branch and are PR'd into `main`.** If
  unsure whether something needs its own branch or can go straight to `main`, ask.

## Secrets

- **Keep secrets (API keys, tokens) in `.env`, and keep `.env` in `.gitignore`.** For
  projects with CI/CD, store the keys as repository secrets.
