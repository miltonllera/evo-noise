# Running experiments / jobs on remote machines

Index for every remote target we run work on. **Pick the right doc before doing anything**: the procedures are not interchangeable, and applying the wrong one does real damage (a tmux training run on a cluster login node gets the account restricted).

## Non-negotiable code path

**Tracked content moves only local checkout → GitHub over SSH → execution machine.** Never use HTTPS or a machine-local/bare Git remote, and never edit, commit, or push tracked files on an execution machine. Treat contrary config or documentation as an error, not an exception; **STOP and ask the user**. Gitignored results return through the documented fetch path.

## Who is where

- **Local machines**: where agent sessions run (here: `pro`, `air`, `mini`): they coordinate the work: hold a checkout, launch and manage jobs over SSH, fetch results, build reports. **"Local" always means the machine the current agent session runs on**; an agent SSH'd into another machine works *on that machine*, and files land there. `results/` is gitignored, per-machine state, so each local machine that needs results or reports fetches its own copy, from the remote server or from another local machine that already holds the runs.
- **Remote servers**: where jobs run, and nothing else: agents never run there; they SSH in to launch and manage. Scheduler clusters (submit from the login node; the login node never computes) and bare SSH boxes are two flavours of the same role; the table below routes each to its doc.

| Target | Doc | How to tell |
|---|---|---|
| A box we SSH into and own the GPUs on | [`REMOTE_SSH.md`](REMOTE_SSH.md) | `which sbatch` finds nothing |
| ITU HPC SLURM cluster (`hpc.itu.dk`) | [`ITU_HPC.md`](ITU_HPC.md) | the user names ITU HPC, or the host is `hpc.itu.dk` |
| Any other SLURM cluster | [`SLURM_INSTRUCTIONS.md`](SLURM_INSTRUCTIONS.md) + that cluster's own doc | `sbatch`/`squeue` on PATH |
| PBS (`qsub`) / LSF (`bsub`) | none yet; **STOP and ask the user** | `qsub` / `bsub` on PATH |

One doc per scheduler cluster: [`SLURM_INSTRUCTIONS.md`](SLURM_INSTRUCTIONS.md) holds scheduler-generic behaviour, and each cluster's doc holds only its own facts (host, partitions, GPU tags, limits). Adding a cluster means adding a file here and a row above, never editing another cluster's doc. SSH-machine inventory and notes stay together in [`REMOTE_SSH.md`](REMOTE_SSH.md); never create per-machine instruction files.

## True on every target

`<repo>` is the GitHub repository name, and **the checkout lives at `~/<repo>` on every target**, the same name as on GitHub, no per-machine variation. A checkout that is missing, elsewhere, or under a different name → **STOP and ask the user**: don't launch in it, and don't go hunting for it at fetch time.

- **Before launch,** push to GitHub via `origin` and verify the execution checkout is at `~/<repo>`, uses GitHub SSH, is clean, and is at that exact commit. Any surprise → **STOP and ask the user.**
- **Set up the env with `uv`, never `pip`:** `uv sync`. **No `uv` → STOP and ask**; don't fall back to `pip` / `conda` / `venv` / system Python on your own.
- **Secrets live in a gitignored `.env`** at the repo root on the remote (`~/<repo>/.env`), not exported by default. Source it with `set -a; source .env; set +a` **inside the shell that runs the job**: tmux sessions and batch jobs both start fresh shells, so sourcing in your SSH session doesn't carry in. A job missing a key should fail loud; never hard-code a secret to work around it.
- **Record execution locally in `EXPERIMENTS.md`:** add the row as `PLANNED` when defined; once the job is confirmed started, set `server` to `<target> (<job/session id>)` and `status` to `RUNNING`, then commit and push via `origin`. When execution ends, set `COMPLETED`, `FAILED`, or `ABORTED`. Analysis columns remain untouched.
- **Launching a run is not the end of your job: check on it periodically (about hourly) until it ends.** A run can die, fail, or stall without ever returning a non-zero exit: the process or job still "exists" but nothing is being computed. So a live session/queue entry and a quiet log prove nothing on their own; also confirm the log is still advancing and the GPU/CPU the run should be using actually shows load (`nvidia-smi`, `htop`; on a scheduler, the job's own accounting). Dead, stalled, or idle → deal with it now (restart, fix, or set `FAILED`/`ABORTED`) and say so; don't discover it at fetch time.
  - The run records how it ended, but nothing tells you it has, so **keep yourself on that cadence rather than waiting to be prompted**: set up a recurring check when you launch. A run outliving the session is normal and needs no handover notes: the `EXPERIMENTS.md` row says where it ran, and the target itself says how it ended.
- **Retrieve results with the `fetch-results` skill**: it reads the experiment's `server` column in `EXPERIMENTS.md`, then pulls `results/<id>/` back additively with `rsync -avz`. Fetching never generates a report or analysis and never populates analysis columns. It reports stale execution status rather than silently changing it.
- **Use subagents** to run and manage remote jobs; the main agent coordinates them and consolidates results, keeping its context lean.
