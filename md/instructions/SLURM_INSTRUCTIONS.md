# SLURM clusters (scheduler-generic)

Any SLURM cluster. Cluster-specific facts (host, partitions, tags, limits) live in that cluster's own doc (e.g. [`ITU_HPC.md`](ITU_HPC.md)); read both, after [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md). You don't run the work; you describe it in a job script, submit, and poll. The SSH + tmux procedure does **not** apply.

## Rules that get accounts restricted

- **Nothing heavy on the login node**: it's for editing, submitting, and reading the queue. Downloads, installs/`uv sync`, preprocessing, and any long-lived process run as jobs.
- **Never pick a GPU yourself**: no `nvidia-smi` shopping, no `CUDA_VISIBLE_DEVICES`. Request via `--gres`.
- **Never let `TMPDIR` default to a shared `/tmp`**: point it where the cluster's doc says (node-local scratch, else `$HOME/tmp`).

## Procedure

1. Commit locally and push to GitHub via `origin`; `ssh` in, clone/pull from GitHub over SSH into `~/<repo>` (the GitHub repository name, always) and verify `HEAD` is the exact pushed commit. Conflict, dirty tree, a checkout under another name, or other surprise → STOP and ask.
2. Build the env **as a job** (`uv sync`). Prefer a uv installed in `$HOME` over `module load uv`; module trees can be architecture-specific and missing on the node the job lands on.
3. One job script per experiment, committed alongside its config.
4. `mkdir -p results/<id>/logs` on the cluster: job output goes there (gitignored, and `fetch-results` brings it home with the runs; an array job writes one file per task, so it stays out of the run directories), then `sbatch` it; back in the local checkout, set the experiment row's `server` to `<cluster> (job <jobid>)` and `status` to `RUNNING`, then commit and push it to GitHub via `origin`.
5. Verify it started (`sacct -j <jobid>`, tail the output file); re-check a minute later; submission success ≠ run success. Then check about hourly until it ends. **`sacct -j <jobid>` is the state source**: it answers whether the job is running and, once it ends, how it ended, and it keeps answering long afterwards (`squeue` drops a job the moment it finishes, so a job vanishing from it means nothing on its own). `sacct` can't see a stall, though: a job that holds its allocation while computing nothing still reads `RUNNING`, and the wall clock will kill it with no error of its own. So also confirm the output file is advancing and `sstat -j <jobid> --format=JobID,AveCPU,MaxRSS` shows CPU time growing.
6. On termination, set the row to `COMPLETED`, `FAILED`, or `ABORTED` from the state `sacct` reports; `seff <jobid>` adds efficiency detail worth a look. Do not create analysis or findings.
7. Back on the local machine: `rsync` the results over, the `fetch-results` skill's job
   (`results/` is gitignored, so git never carries them).

```bash
#!/bin/bash
#SBATCH --job-name=<id>
#SBATCH --output=results/<id>/logs/slurm-%j.out   # `mkdir -p` it first or the job dies at launch
#SBATCH --partition=<queue>             # see the cluster's doc
#SBATCH --gres=gpu:1                    # omit for CPU-only
#SBATCH --constraint=<a>|<b>            # widen: any feature that fits
#SBATCH --cpus-per-task=8
#SBATCH --mem=<size>                    # per node; size per experiment (see below), never rely on the default
#SBATCH --time=04:00:00                 # HARD kill at this limit
export TMPDIR=$HOME/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1               # else the log looks dead while print() buffers
set -a; source .env; set +a
uv run <command>
```

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # the state source, during AND after
squeue -u $USER            # queue view: PD/R + reason ·  squeue --start: est. start
seff <jobid>               # efficiency detail, once it has ended
scancel <jobid>
```

- **Maximise the candidate set; queue time is the real cost.** `--constraint` with `|`, untagged `--gres=gpu:N`, comma-listed `--partition`, honest `--time` (short jobs backfill). Pinning one node/GPU model = waiting for that one thing.
- **Size `--mem` per experiment.** Too low is an OOM kill (exceeding the request kills the job even with free RAM on the node); too high shrinks the candidate-node set and queues longer. `--mem` is per node; `--mem-per-cpu`/`--mem-per-gpu` scale with the other request and are mutually exclusive with it.
- **`--constraint` selects nodes; `--gres` allocates devices**; you need both for a GPU. Verify a feature tag exists (`sinfo -o "%N %f"`); a constraint matching nothing hangs `PENDING` forever, no error.
- Sweeps = one array job (`--array=1-N` + `$SLURM_ARRAY_TASK_ID`), not N submissions. Chain stages with `--dependency=afterok:<id>`. Debug via `srun --pty bash -i` (exit promptly; it holds the allocation).

## Output

Home is (normally) shared between login and compute nodes: jobs write to the same paths you see over SSH. Working dir = where `sbatch` ran; stdout+stderr → the `--output` file (default `slurm-<jobid>.out`). `tail -f` it; that's your terminal now. Batch jobs have no tty (prompts hang until the wall clock kills them); interactive needs `srun --pty` / `salloc`.

Looks-broken-but-isn't: Python buffering (hence `PYTHONUNBUFFERED=1`), missing `--output` directory (job dies at launch with nowhere to say so), node-local `/tmp` wiped at job end. Slow dataloaders are usually shared-home NFS; stage to node-local disk only once measured.

## Failure modes

- **`PENDING` forever**: the ask exceeds an account limit or matches no node; read the reason in `squeue`, shrink the request. Resubmitting unchanged queues a second dead job.
- **Killed at wall clock**: checkpoint long runs; set `--time` from a measured short run.
- **Runs on login node, `ModuleNotFoundError` in job**: env drift; never `conda init`, keep the env self-contained (`uv run`).
- **OOM**: `seff` shows RAM OOM; **VRAM OOM shows only in the job log**.
