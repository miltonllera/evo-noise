# Remote machines over SSH (+ tmux)

Launching work on a remote GPU/CPU box we own the GPUs on, from any repo (LLM or not). `<machine>` is the host name from the user's `~/.ssh/config`. Read [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md) first — it holds the rules that apply on every target (commit-push-first, `uv`, `.env`, `rsync`, subagents).

**Wrong doc?** If `sbatch`/`squeue` are on PATH, this procedure does not apply — go back to the index.

## The procedure

1. **Commit and push locally**, then **SSH by config name:** `ssh <machine>`. The repo lives at `~/<repo>` (so `cd <repo>`); `git clone` it first if absent, then check out the branch.

2. **Pull the branch** you just pushed (skip if you just cloned). Conflict or surprise → STOP and ask.

3. **`uv sync`** to create/update the env.

4. **Sanity-check the remote before launching** (each is a real failure mode):
   - Any file the job points at (weights, datasets, config paths) exists on the remote — `~` is the REMOTE home, not your laptop's.
   - The binaries/CLIs it needs are on `PATH` (see PATH gotcha below).
   - Pick a GPU with `nvidia-smi` (see GPU selection below); `htop` / `free -h` for CPU/RAM.

5. **Launch in a descriptively-named tmux session** so it survives disconnects, then **confirm it started** — `tmux new -d` reports success even if the job crashes on startup, so wait a few seconds, then check the session still exists and tail the log (and re-check a minute later — slow imports can delay a crash past the first tail). Once confirmed, fill the experiment's `EXPERIMENTS.md` row — `server` column `<machine> (tmux <name>)`, finding column `RUNNING — launched <date>` — and commit it, so any agent can see what runs where.

6. **Back on the local machine, when the job finishes**: retrieve results with `rsync` —
   the `fetch-results` skill's job. Every local machine that needs the results fetches its
   own copy.

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

# local, when the job finishes: pull the results back (results/ is gitignored)
rsync -avz <machine>:~/<repo>/results/<id>/ results/<id>/
```

## GPU selection — ask before grabbing, unless you're autonomous

`nvidia-smi` shows which GPUs are free/busy, whose process is on each, and memory used. **Pin a job with `CUDA_VISIBLE_DEVICES=X`** in front of the command (comma-separated, `0,1`, for several); the job then sees only those GPUs, so pinned jobs run concurrently without colliding.

**"Busy" != "unusable":** one GPU can host several jobs — what matters is free memory, not whether *a* job is already running. If your footprint clearly fits what's free, sharing is fine; just watch for OOM (it would crash the existing job or yours). Check whether a job on it is already yours, so you don't stack a second run on your own by accident.

**When in doubt, ask — it costs almost nothing.** A few-second confirmation beats crashing someone's multi-hour job. Specifically:

- **Interactive (default):** if no GPU was specified, show what's free vs. busy and ask — don't silently grab one. If one was specified, use it; if it's full, report and ask rather than reassigning yourself.
- **Autonomous** (user handed off open-endedly — "keep the sweep going"): reuse your current GPU; if it's now busy, share it when your job clearly fits, else move to a free one — no need to ask, just don't OOM the box. Only the *first* launch with no GPU specified still warrants one quick check.

## Conventions

- **One tmux session per job**, named for what it runs (`train-baseline`, `eval-sweep`). Separate jobs on separate GPUs run concurrently — the point of a multi-GPU box.
- **Redirect to a log (`> log 2>&1`), never `tee`:** with `| tee log` the shell reports `tee`'s exit code, so a crash looks like success (exit 0).

## PATH gotcha — launch from an interactive shell

User-installed tools (uv-tool binaries, custom CLIs) are often only on the **interactive** shell PATH (from `~/.zshrc` / `~/.bashrc`). A non-interactive `ssh <machine> 'bash -lc …'` may not see them — so launch from inside the tmux pane (interactive shell) or via `zsh -ic`, never `bash -lc`, or subprocess-spawned binaries won't be found. This is why the launch block above wraps the command in `zsh -ic` — a bare `tmux new -d '<cmd>'` runs through non-interactive `sh` and would miss them too.
