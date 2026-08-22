# Remote machines over SSH (+ tmux)

Launching work on a remote GPU/CPU box we own the GPUs on, from any repo (LLM or not). `<machine>` is the host name from the user's `~/.ssh/config`. Read [`CLUSTER_INSTRUCTIONS.md`](CLUSTER_INSTRUCTIONS.md) first; it holds the rules that apply on every target (commit-push-first, `uv`, `.env`, `rsync`, subagents).

**Wrong doc?** If `sbatch`/`squeue` are on PATH, this procedure does not apply; go back to the index.

## Available machines

Keep this inventory and all machine-specific notes here; never create one instruction file per SSH machine.

| Machine | CPU | RAM | GPU | Notes |
|---|---|---:|---|---|
| `tripper2` | Threadripper 3990X (64 cores / 128 threads) | 125 GiB | 1× RTX 4090 (24 GiB) | — |
| `tripper4` | Threadripper 3990X (64 cores / 128 threads) | 251 GiB | 1× RTX 3090 (24 GiB) | Tailscale network |
| `miami` | Threadripper 3990X (64 cores / 128 threads) | 251 GiB | 1× RTX 3070 Ti (8 GiB) | — |
| `disco` | 2× EPYC (128 cores / 256 threads) | 503 GiB | 8× RTX 6000 Ada (48 GiB each) | — |
| `grime` | 2× EPYC 7763 (128 cores / 256 threads) | 503 GiB | 8× RTX 6000 Ada (48 GiB each) | — |
| `jungle` | 2× EPYC 9554 (128 cores / 256 threads) | 503 GiB | 8× RTX 6000 Ada (48 GiB each) | — |
| `dada` | Threadripper PRO 5995WX (64 cores / 128 threads) | 503 GiB | 4× RTX 6000 Ada (48 GiB each) | — |
| `vapor` | EPYC 9124 (16 cores / 32 threads) | 377 GiB | 4× RTX A6000 (48 GiB each) | — |

## Notes

- Prefer `tripper2`, `tripper4`, and `miami` for CPU-heavy, multicore/multithread experiments with no or light GPU use.

## The procedure

1. **Commit locally and push to GitHub via `origin`**, then **SSH by config name:** `ssh <machine>`. The repo lives at `~/<repo>`, the GitHub repository name, always. **If `~/<repo>` does not exist, the first step is `git clone` from GitHub over SSH**; a missing repo is expected on a fresh machine, not a surprise. Then `cd ~/<repo>` and confirm it is the right checkout with `git -C ~/<repo> remote get-url origin`; a checkout under any other name → STOP and ask.

2. **Pull and verify `HEAD` is the exact pushed commit.** Conflict, dirty tree, or other surprise → STOP and ask.

3. **`uv sync`** to create/update the env. No `uv` on the remote? Install it (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and re-check `PATH`; if the install fails, ask the user to do it rather than working around it.

4. **Sanity-check the remote before launching** (each is a real failure mode):
   - Any file the job points at (weights, datasets, config paths) exists on the remote; `~` is the REMOTE home, not your laptop's.
   - The binaries/CLIs it needs are on `PATH`; check with the same wrapper the launch uses: `$SHELL -ic 'command -v <tool>'` (see PATH gotcha below).
   - Pick a GPU with `nvidia-smi` (see GPU selection below); `htop` / `free -h` for CPU/RAM.

5. **Launch in a descriptively-named tmux session** so it survives disconnects, then **confirm it started**. The launch also writes the run's exit code next to its log (see below); without that, a finished run leaves no record of *how* it finished. Once confirmed, return to the local checkout, set the experiment row's `server` to `<machine> (tmux <name>)` and `status` to `RUNNING`, then commit and push. **Then check about hourly until it ends:**
   - **Session alive** → log still advancing, and `nvidia-smi` / `htop` showing the run's process actually working. A stalled or hung process looks identical to a running one until you look at its load.
   - **Session gone** → it ended; read the exit code file for the verdict and set `COMPLETED` or `FAILED`. The log usually explains a failure, but the deaths that matter most leave nothing in it: an OOM kill, a `kill -9`, a reboot all look exactly like a clean finish without that file.

   Log and status file both live in `results/<id>/logs/`: gitignored, and already the path `fetch-results` syncs, so they come back with the runs rather than being stranded on a box that may later be wiped.

   Set `COMPLETED`, `FAILED`, or `ABORTED` from execution state; do not write analysis.

6. **Back on the local machine, when the job finishes**: retrieve results with `rsync`:
   the `fetch-results` skill's job. Every local machine that needs the results fetches its
   own copy.

```bash
# remote, after the exact GitHub commit has been pulled
ssh <machine>
[ -d ~/<repo> ] || git clone git@github.com:<owner>/<repo>.git ~/<repo>   # fresh machine: clone first
cd ~/<repo>                                      # always the GitHub repo name
git remote get-url origin                        # must be this repo's GitHub SSH URL
git pull --ff-only                               # conflict/surprise? STOP, ask
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # install fails? ask the user
uv sync

mkdir -p results/<id>/logs; rm -f results/<id>/logs/<name>.status   # stale status would read as this run's verdict
tmux new -d -s <name> "$SHELL -ic 'set -a; source .env; set +a; CUDA_VISIBLE_DEVICES=<gpu> uv run <command> > results/<id>/logs/<name>.log 2>&1; echo \$? > results/<id>/logs/<name>.status'"
sleep 10; tmux has-session -t <name> && tail -n 50 results/<id>/logs/<name>.log   # confirm it started; re-check in a minute

# ~hourly until it ends; alive is not enough (a hung process still 'exists'), gone needs a verdict
tmux has-session -t <name> 2>/dev/null \
  && { tail -n 5 results/<id>/logs/<name>.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; } \
  || { echo "ended: exit=$(cat results/<id>/logs/<name>.status 2>/dev/null || echo NO-STATUS)"; tail -n 20 results/<id>/logs/<name>.log; }
# CPU-only run: top -bn1 | head -20 in place of nvidia-smi
# NO-STATUS = died before it could write one (OOM kill, kill -9, reboot); that is itself the finding
# reattach: tmux attach -t <name>  ·  detach: Ctrl-b d  ·  stop: tmux kill-session -t <name>

# local, when the job finishes: pull the results back (results/ is gitignored)
rsync -avz <machine>:~/<repo>/results/<id>/ results/<id>/
```

## GPU selection: ask before grabbing, unless you're autonomous

`nvidia-smi` shows which GPUs are free/busy, whose process is on each, and memory used. **Pin a job with `CUDA_VISIBLE_DEVICES=X`** in front of the command (comma-separated, `0,1`, for several); the job then sees only those GPUs, so pinned jobs run concurrently without colliding.

**"Busy" != "unusable":** one GPU can host several jobs; what matters is free memory, not whether *a* job is already running. If your footprint clearly fits what's free, sharing is fine; just watch for OOM (it would crash the existing job or yours). Check whether a job on it is already yours, so you don't stack a second run on your own by accident.

**When in doubt, ask. It costs almost nothing.** A few-second confirmation beats crashing someone's multi-hour job. Specifically:

- **Interactive (default):** if no GPU was specified, show what's free vs. busy and ask; don't silently grab one. If one was specified, use it; if it's full, report and ask rather than reassigning yourself.
- **Autonomous** (user handed off open-endedly, "keep the sweep going"): reuse your current GPU; if it's now busy, share it when your job clearly fits, else move to a free one; no need to ask, just don't OOM the box. Only the *first* launch with no GPU specified still warrants one quick check.

## Conventions

- **One tmux session per job**, named for what it runs (`train-baseline`, `eval-sweep`). Separate jobs on separate GPUs run concurrently, the point of a multi-GPU box.
- **Redirect to a log (`> log 2>&1`), never `tee`:** with `| tee log` the shell reports `tee`'s exit code, so a crash looks like success (exit 0).
- **Don't nest a multi-command loop in quotes:** for anything remote or backgrounded, write the commands to a script on the target and run that. If you must nest, escape every `$var` and confirm the expansion landed before trusting the launch.

## PATH gotcha: one wrapper for everything

Run every remote command (launches and checks alike) through **`$SHELL -ic '<cmd>'`**. User-installed tools (uv-tool binaries, custom CLIs) are often only on the **interactive** shell PATH (from `~/.zshrc` / `~/.bashrc`), and `-ic` is what loads those rc files; `$SHELL` picks each machine's actual shell, so the same line works everywhere. A bare `tmux new -d '<cmd>'` or `ssh <machine> '<cmd>'` skips the rc files and won't find those tools; that's a fact about the missing wrapper, not about the machine.

Interactive rcs may print banners or escape codes on any machine; match command output by substring, not exact equality.
