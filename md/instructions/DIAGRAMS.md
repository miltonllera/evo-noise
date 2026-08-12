# Plan diagrams

Each plan carries a standing `mermaid` diagram, embedded next to the content it visualises, so the user can see the shape of the project at a glance. Draw them at project instantiation and keep them current thereafter. All diagrams are **high level**: they visualise the shape of the plan, not its details.

## The three diagrams

- **RESEARCH.md — research plan**: a question tree grouped in phase subgraphs. Phases are `subgraph` boxes linked in sequence; inside them, research questions branch into the experiment groups that answer them.
- **EXPERIMENTS.md — experiments map**: a DAG of concrete experiment ids. An edge means “motivated/unlocked” and may state why the child experiment was launched; it never states or infers a finding. Node style mirrors execution status: `PLANNED` → `:::queued`, `RUNNING` → `:::wip`, and `COMPLETED`/`FAILED`/`ABORTED` → `:::done`.
- **TODO.md — implementation roadmap** (top of the `Implementation` section): a left-to-right flowchart of features/components in dependency order, at a coarser grain than the bullets (feature-level nodes, not sub-tasks).

## Conventions

- **Status classes**, mirroring TODO notation `[x]` / `[~]` / `[ ]` — copy this snippet verbatim into every diagram so they all look alike:
  ```
  classDef done fill:#1a7f37,color:#fff
  classDef wip fill:#d4a72c,color:#000
  classDef queued fill:none,stroke-dasharray:4 3
  ```
  Tag nodes with `:::done`, `:::wip`, `:::queued`.
- **Diagrams are maps, not dumps**: ~15 nodes max, short node/edge labels. If a diagram outgrows that, raise the abstraction level and shorten or drop edge labels first.
- **Update a diagram in the same edit as the plan/status text it reflects**, so diagrams and text never drift apart.
