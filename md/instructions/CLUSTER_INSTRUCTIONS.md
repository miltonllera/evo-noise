# Running experiments / jobs on a remote machine (SSH + tmux)

Generic procedure for launching work on a remote GPU/CPU box from any repo (LLM or not). `<repo>` is the repository we're working in; `<machine>` is the host name from the user's `~/.ssh/config`.

## The procedure

1. **Commit and push locally first.** On whatever branch we're on, commit and push before going to the remote — the remote pulls from the git remote, so anything uncommitted or unpushed won't be there. If there are unrelated dirty changes you didn't make, surface them rather than sweeping them into a commit.
   
2. **SSH by config name:** `ssh <machine>`. The repo lives at `~/<repo>` (so `cd <repo>`). If it isn't there yet, `git clone` it first, then `cd` in and check out the branch.
   
3. **Pull the branch** you just pushed (skip if you just cloned — already current). **If `git pull` reports a conflict** — or the remote tree is dirty / on an unexpected branch / diverged — **STOP and ask the user.** Don't force, reset, or stash-discard it yourself.
   
4. **Set up the env with `uv`, never `pip`:** `uv sync` (this also creates the env from scratch the first time). **If `uv` isn't available, STOP and ask** — don't fall back to `pip` / `conda` / `venv` / system Python on your own.
   
5. **Sanity-check the remote before launching** (each is a real failure mode):
   - Any file the job points at (weights, datasets, config paths) exists on the remote — `~` is the REMOTE home, not your laptop's.
   - The binaries/CLIs it needs are on `PATH` (see PATH gotcha below).
   - Pick a GPU with `nvidia-smi` (see GPU selection below); `htop` / `free -h` for CPU/RAM.
   
6. **Launch in a descriptively-named tmux session** so it survives disconnects, then **confirm it started** — `tmux new -d` reports success even if the job crashes on startup, so wait a few seconds, then check the session still exists and tail the log (and re-check a minute later — slow imports can delay a crash past the first tail).

```bash
# local: commit + push your work first
git commit -am "<message>" && git push          # whatever branch you're on

# remote
ssh <machine>
cd <repo>                                        # git clone first if absent
git pull                                         # conflict? STOP, ask
uv sync                                          # no uv? STOP, ask
tmux new -d -s <name> "zsh -ic 'set -a; source .env; set +a; CUDA_VISIBLE_DEVICES=<gpu> uv run <command> > <name>.log 2>&1'"
sleep 10; tmux has-session -t <name> && tail -n 50 <name>.log   # confirm it started; re-check in a minute
# reattach: tmux attach -t <name>  ·  detach: Ctrl-b d  ·  stop: tmux kill-session -t <name>
```

## GPU selection — ask before grabbing, unless you're autonomous

`nvidia-smi` shows which GPUs are free/busy, whose process is on each, and memory used. **Pin a job with `CUDA_VISIBLE_DEVICES=X`** in front of the command (comma-separated, `0,1`, for several); the job then sees only those GPUs, so pinned jobs run concurrently without colliding.

**"Busy" != "unusable":** one GPU can host several jobs — what matters is free memory, not whether *a* job is already running. If your footprint clearly fits what's free, sharing is fine; just watch for OOM (it would crash the existing job or yours). Check whether a job on
it is already yours, so you don't stack a second run on your own by accident.

**When in doubt, ask — it costs almost nothing.** A few-second confirmation beats crashing someone's multi-hour job. Specifically:

- **Interactive (default):** if no GPU was specified, show what's free vs. busy and ask — don't silently grab one. If one was specified, use it; if it's full, report and ask rather than reassigning yourself.
- **Autonomous** (user handed off open-endedly — "keep the sweep going"): reuse your current GPU; if it's now busy, share it when your job clearly fits, else move to a free one — no need to ask, just don't OOM the box. Only the *first* launch with no GPU
  specified still warrants one quick check.

## Conventions

- **One tmux session per job**, named for what it runs (`train-baseline`, `eval-sweep`). Separate jobs on separate GPUs run concurrently — the point of a multi-GPU box.
- **Redirect to a log (`> log 2>&1`), never `tee`:** with `| tee log` the shell reports `tee`'s exit code, so a crash looks like success (exit 0).
- Use subagents to run and manage remote jobs; the main agent coordinates them and consolidates results, keeping its context lean.

## PATH gotcha — launch from an interactive shell

User-installed tools (uv-tool binaries, custom CLIs) are often only on the **interactive** shell PATH (from `~/.zshrc` / `~/.bashrc`). A non-interactive `ssh <machine> 'bash -lc …'` may not see them — so launch from inside the tmux pane (interactive shell) or via `zsh -ic`, never `bash -lc`, or subprocess-spawned binaries won't be found. This is why the launch block above wraps the command in `zsh -ic` — a bare `tmux new -d '<cmd>'` runs through non-interactive `sh` and would miss them too.

## Batch schedulers (Slurm, PBS, LSF, …) — NOT COVERED YET

**⚠️ This doc only covers direct SSH + tmux boxes.** If the target machine uses a batch scheduler (`sbatch`/`squeue`, `qsub`, `bsub` on PATH, or the user mentions Slurm/PBS/LSF), the procedure above does NOT apply — don't grab GPUs manually or launch tmux jobs on login nodes. **STOP and ask the user how to proceed.**

<!-- TODO: fill in scheduler instructions (sbatch conventions, submit-then-verify loop, srun, etc.) -->

## Secrets / environment variables

Secrets (API keys, tokens) typically live in a gitignored `.env` at the repo root on the remote (`~/<repo>/.env`), not exported by default — source it with `set -a; source .env; set +a` **inside the tmux command** (as the launch block above does): the tmux session starts a fresh shell, so sourcing in your SSH session doesn't carry in.

A job missing a key should fail loud; never hard-code a secret to work around it.
