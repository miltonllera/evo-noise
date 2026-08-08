# Tracking experiments with SQLite

A reusable pattern for recording experiment results so the **on-disk record is the
single source of truth** and every visualization — local plots or a dashboard render
— is a **regenerable projection** of it, never the source.

## The idea in one line

The experiment runs once and writes a complete, self-contained SQLite database; every
plot, GIF, summary, or tracker (Weights & Biases, MLflow, …) is downstream **replay**
read back from that database. The expensive thing (compute / API calls) happens once;
everything visual is cheap to regenerate.

## Why SQLite, not numpy arrays written straight to disk

- **A `.npy`/`.npz` dump freezes one view.** It captures the exact arrays you happened
  to need for one plot. Want a different slice, a fixed metric, or a new chart later?
  You re-run the experiment. A queryable DB lets you `SELECT` any slice after the fact,
  so a new visualization is a new query — not a new run.
- **numpy buffers in memory and dumps at the end** → a crash mid-run loses everything.
  SQLite `INSERT`s transactionally as the run progresses, so a crashed long run still
  leaves a valid, partial, queryable record.
- **Experiment data is usually relational and text-heavy** (one run → config + N items
  + M steps + free-text logs/outputs), not a single fixed-shape numeric matrix. numpy
  is built for the latter; SQLite stores the heterogeneous, typed, self-describing
  former in one file.

> Rule of thumb: numpy/CSV win only when the per-run data is one large homogeneous
> numeric matrix you want to do vectorized math on. For typed, relational,
> incrementally-written records with small numeric summaries, push everything into
> SQLite and treat the arrays you feed a plot as a transient projection.

## The procedure: regenerate any visualization by reading the DB

1. **Write once.** During the run, the producer `INSERT`s every observation into a
   SQLite file (one DB per run) as it happens — config, per-step metrics, per-item
   state, optional full traces.
2. **Read back to visualize.** To draw a plot, build the numpy arrays *on demand* from
   a `SELECT`, then hand them to matplotlib/your renderer. To push to a dashboard, read
   the same rows and drive the tracker's API. Neither path re-runs the experiment.
3. **So regenerating is free.** Changed a colormap, fixed a metric, want a different
   axis, deleted the W&B project, want to render offline now and sync later? Re-read the
   DB and re-render. The compute never repeats; the visualization is pure replay.

This is also why the tracker can sit behind a failure boundary: if the dashboard write
fails, it degrades (offline → disabled) while the authoritative DB on disk is untouched.

## How a project implements it (reference)

A concrete shape this typically takes:

- Each run writes `results/<id>/<run_id>/run.db` (SQLite) as the canonical store — one
  folder per experiment, one subdirectory per run inside it — with tables for
  run metadata/config, per-step metrics, per-item state, and an optional gated full
  input/output trace. Run metadata includes the git commit hash of the code at launch,
  stamped by the launcher; a stamped copy of the config sits beside the DB.
- **Runs that write files** (images, media, checkpoints) also record each file's relative
  path and sha256 digest in an artifacts table. Downstream projections — reports above
  all — treat a file without a recorded digest as non-evidence (see
  [`EXPERIMENT_REPORTS.md`](EXPERIMENT_REPORTS.md)).
- The visualization tools open an **existing** `run.db` read-only and rebuild their
  arrays/frames from it — they never re-run the experiment to draw a plot or GIF.
- The tracker integration has two callers over one code path: the live runner logs rows
  from memory as the run finishes; a **backfill** rebuilds byte-identical rows by reading
  `run.db` back and replaying them to the tracker — proving the dashboard is a pure
  projection of the DB.

> Scaling note: if runs grow very large (≳1 GB) or you need compression, look into transparent page-level compression (`sqlite-zstd`), Parquet export for archival, and DuckDB as an analytics overlay over the SQLite/Parquet files.

The findings derived from `run.db` are written up per [`EXPERIMENT_ANALYSIS.md`](EXPERIMENT_ANALYSIS.md): one analysis file per experiment in `md/experiments/`, indexed in `EXPERIMENTS.md` (at the repo root).
