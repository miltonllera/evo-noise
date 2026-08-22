# ITU HPC SLURM cluster (`hpc.itu.dk`)

ITU facts only; procedure in [`SLURM_INSTRUCTIONS.md`](SLURM_INSTRUCTIONS.md). Docs: <https://hpc.itu.dk/> (expired TLS cert: `curl -k`). Support: `hpc@itu.dk`.

## Access and storage

- `ssh itu` (`<itu-username>@hpc.itu.dk`); needs eduroam / ITU++ / ITU VPN. Rocky Linux, no root.
- Login banner prints the full hardware inventory (`ssh itu 'cat /etc/motd'`), the way to refresh this doc.
- `/home` is one shared NFS mount on all nodes, unquota'd, **not backed up**; **no `/scratch`**: job output lands in home, nothing to copy back. Not on InfiniBand: fine for logs/checkpoints, slow for small-file-heavy dataloaders.
- Shared datasets: `/home/common/datasets/`.

## Environment: `uv`, not conda

Verified working end-to-end in jobs (compute nodes have outbound internet; uv manages its own Pythons).

- **Don't `module load uv`**: the module tree is architecture-specific and uv exists only for AMD Zen nodes; the job dies wherever else it lands.
- Instead: uv installed once in `$HOME/.local/bin` (static binary, shared home → works on every node); `export PATH="$HOME/.local/bin:$PATH"` in every job script.
- `.venv`, interpreters, and wheel cache all live on shared home: `uv sync` once, every node reuses it.

## Partitions

Website is stale: trust `sinfo`. Preemption is **OFF** cluster-wide: jobs are never killed for priority; cost is wait-to-start only.

| Partition | Nodes | Max time | Priority |
|---|---|---|---|
| `dgpu` | desktop[1,6-8,12,15], 8 cores, RTX 2070 or none | 10 d | 10 |
| `cores` | cn[8,14-18] | 7 d | 8 |
| `acltr` | cn[3-7,12-18], the public GPU queue | 7 d | 7 |
| `cores_any` | cn[3-7,12,16-18], spare cores on GPU nodes | 7 d | 7 |
| `scavenge` (default) | everything | 1 d | 1 |

Defaults that bite: **`DefaultTime=01:00:00`** (always set `--time`); **`DefMemPerCPU=2048M`** everywhere except `acltr` (always set `--mem`); no `--partition` → `scavenge`.

Memory requests are hard-enforced (cgroup OOM kill on crossing the request, verified: free RAM on the node doesn't help). There's no fair-share/billing penalty for over-asking (all priority weights are 0, despite what the website says). The only cost is fitting on fewer nodes and queueing longer.

**The best GPUs (cn11 H100, cn19 L40S, cn10 RTX 6000 Ada, cn9 A30) are researcher-owned: we have no owner access; `scavenge` is the only route.** ≤24 h jobs: submit there and wait, they run to completion. Longer: checkpoint into ≤1-day chunks or use `acltr` hardware. Don't ask about owner access: the answer is no.

## Nodes

Snapshot; refresh with `sinfo -N -o "%12N %5c %9m %80f %28G %8t"`.

| Node | Cores | MEM | GPUs | CC | Via |
|---|---|---|---|---|---|
| cn11 | 96 | 1.5 TiB | 2x H100 94 GiB | 9.0 | `scavenge` only |
| cn19 | 64 | 515 GiB | 4x L40S 48 GiB | 8.9 | `scavenge` only |
| cn10 | 48 | 250 GiB | 2x RTX 6000 Ada 48 GiB | 8.9 | `scavenge` only |
| cn9 | 192 | 250 GiB | 2x A30 24 GiB | 8.0 | `scavenge` only |
| cn7 | 96 | 257 GiB | 4x A30 + 1x A100 80 GiB | 8.0 / 7.0* | `acltr` |
| cn13 | 32 | 128 GiB | 4x A100 40 GiB | 8.0 | `acltr` |
| cn18 | 96 | 250 GiB | 2x A30 24 GiB | 8.0 | `acltr` |
| cn12 | 80 | 380 GiB | 2x RTX 8000 48 GiB | 7.5 | `acltr` |
| cn3-6 | 48/48/32/64 | 190-500 GiB | 3/4/2/6x V100 32 GiB | 7.0 | `acltr` |
| cn16-17 | 40 | 250 GiB | 2x GTX 1080 Ti 11 GiB | 6.1 | `acltr`, `cores` |
| cn8 | 256 | 250 GiB | — | | `cores` |
| cn14-15 | 48 | 120 GiB | — (cn15: FPGA) | | `cores` |
| desktops | 8 | 30 GiB | RTX 2070 8 GiB or none | 6.1 | `dgpu` |

*cn7 advertises `gpu_cc_7_0` despite Ampere cards: constrain by GPU-model tag there, not CC.

## Picking GPUs

- Ready-made wide constraints: any modern card `--constraint="gpu_cc_8_0|gpu_cc_8_9|gpu_cc_9_0"`; any ≥40 GiB card `--constraint="gpu_a100_40gb|gpu_a100_80gb|gpu_l40s|gpu_rtx6000|gpu_rtx8000"` (+ untagged `--gres=gpu:1`). Partition pairs: `cores,cores_any`; `acltr,scavenge`.
- **bf16 needs CC ≥ 8.0**: V100/RTX 8000 are fp16/fp32 only; a bf16 script queues happily and then fails there.
- **Skip 1080 Ti / RTX 2070 as GPUs** (CC 6.1: no tensor cores, tiny VRAM); cn16/17 remain good **CPU** candidates via `cores`/`cores_any`.
- Sweet spot: A30 (cn7/cn18) and A100 40 GiB (cn13): bf16-capable, in `acltr`, less contended. Biggest `acltr` card: cn7's lone A100 80 GiB. Multi-GPU only if genuinely distributed.
- Tag gotcha: cn16/17 have **no `gpu_gtx1080ti` feature**: use `--gres=gpu:gtx1080ti:N`. InfiniBand (`ib`) only on cn[3-6],cn8.

## Limits

Researcher account: 7-day wall time, no CPU-minute cap (students: 3 d + per-queue cap). Extensions of *running* jobs and limit raises via `hpc@itu.dk` (cc supervisor): the user's call, and only while the job is alive.
