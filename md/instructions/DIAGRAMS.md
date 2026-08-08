# Plan diagrams

Each plan carries a standing `mermaid` diagram, embedded next to the content it visualises, so the user can see the shape of the project at a glance. Draw them at project instantiation and keep them current thereafter. All diagrams are **high level**: they visualise the shape of the plan, not its details.

## The three diagrams

- **RESEARCH.md — research plan**: a question tree grouped in phase subgraphs. Phases are `subgraph` boxes linked in sequence; inside them, research questions branch into the experiment groups that answer them.
- **EXPERIMENTS.md — experiments map** (below the index table): a DAG of concrete experiments (nodes are the experiment ids — already semantic, e.g. `field-sparsity`); an edge means "motivated/unlocked" and is labelled with the one-line finding that led to the child experiment — the diagram reads as the narrative of the research. Edge labels are high-level comments relevant for the research (what the finding means for the research questions), not low-level details (parameters, metrics, implementation notes) — those stay in the table and `md/experiments/`. Node status mirrors the index: `:::wip` while an experiment's row says `RUNNING`, `:::done` once it has its finding, `:::queued` while it's still a `[ ]` item in TODO.md.
- **TODO.md — implementation roadmap** (top of the `Implementation` section): a left-to-right flowchart of features/components in dependency order, at a coarser grain than the bullets (feature-level nodes, not sub-tasks).

## Conventions

- **Status classes**, mirroring TODO notation `[x]` / `[~]` / `[ ]` — copy this snippet verbatim into every diagram so they all look alike:
  ```
  classDef done fill:#1a7f37,color:#fff
  classDef wip fill:#d4a72c,color:#000
  classDef queued fill:none,stroke-dasharray:4 3
  ```
  Tag nodes with `:::done`, `:::wip`, `:::queued`.
- **Diagrams are maps, not dumps**: ~15 nodes max, short node/edge labels. If a diagram outgrows that, raise the abstraction level; for the experiments map, shorten or drop edge labels first — the full findings stay in the table.
- **Update a diagram in the same edit as the plan/status text it reflects**, so diagrams and text never drift apart.
