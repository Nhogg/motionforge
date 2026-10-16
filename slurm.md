# Slurm cluster reference

Surveyed 2026-10-15 from `head0` with read-only commands and one 2-minute test job on g0. Anything marked **(unverified)** was not checked directly.

## Partitions

| Partition | Nodes | Default | DefaultTime | MaxTime | DefMemPerGPU |
|---|---|---|---|---|---|
| `debug` | g0, g5, g6 | no | **5 min** | 1 day | 16 GB |
| `gpu` | g1–g4 | **yes** | **5 min** | 1 day | 64 GB |

- No partition QoS. AllowAccounts and AllowQos are both ALL. OverSubscribe=NO, no preemption.
- Scheduler: `sched/backfill`, `select/cons_tres` with `CR_CORE_MEMORY`, so CPUs and memory are tracked per job. cgroup confinement is on.
- **Always set `--time`.** The 5-minute default will kill longer jobs.
- If you request a GPU without `--mem`, you get DefMemPerGPU: 16 GB per GPU on debug, 64 GB per GPU on gpu.

## GPU nodes

| Node | Partition | GPU (gres name) × count | CPUs (S:C:T) | Memory (RealMemory) | State at survey |
|---|---|---|---|---|---|
| g0 | debug | V100-PCIE-32GB (`v100`) × 2 | 32 (2:8:2) | 190,000 MB | idle |
| g5 | debug | RTX 2080 (`rtx_2080`) × 1 | 12 (1:6:2) | 62,000 MB | idle |
| g6 | debug | RTX 2080 (`rtx_2080`) × 1 | 12 (1:6:2) | 30,000 MB | idle |
| g1 | gpu | L40S (`l40s`) × 4 | 128 (2:32:2) | 1000 GB | mixed |
| g2 | gpu | L40S (`l40s`) × 4 | 128 (2:32:2) | 1000 GB | mixed |
| g3 | gpu | RTX PRO 6000 (`rtx_pro_6000`) × 8 | 256 (2:64:2) | 1500 GB | mixed |
| g4 | gpu | RTX PRO 6000 (`rtx_pro_6000`) × 8 | 256 (2:64:2) | 1500 GB | idle |

- No node features are defined (`AvailableFeatures=(null)`), so `--constraint` is useless. Pick hardware with a typed gres such as `--gres=gpu:v100:1`, or with `--nodelist`.
- No nodes were down or drained at survey time. Check with `sinfo -R`.
- g0 test job: driver 580.178.04, CUDA 13.0 (driver API), `/usr/local/cuda-12.9` installed. `nvcc` is not on PATH and there is no `module` command.
- The g5 and g6 GPU models come from Slurm's gres names only. nvidia-smi was not run on them **(unverified)**.
- `sinfo -N -l` truncates the MEMORY column for g1–g4 (it shows 102400 and 153600). Use `scontrol show node` instead.

## Account and QOS

- **Accounting is disabled** (`AccountingStorageType=(null)`, `AccountingStorageEnforce=none`). `sacctmgr` and `sacct` don't work.
- There is no account or QOS to set. Leave `--account` and `--qos` out.
- With no `sacct`, finished jobs leave no history. Rely on your own log files.

## Submitting jobs: `debug.sbatch` / `prod.sbatch`

Two generic launchers live in `scripts/slurm/`. Each one runs its arguments as a command (`"$@"`) in the directory you submitted from (`$SLURM_SUBMIT_DIR`). Any `{job}` in the arguments becomes the job id. Both set `MUJOCO_GL=disable` (Warp rendering) and `UV_NO_SYNC=1`.

| file | job name | partition | gres | time |
|---|---|---|---|---|
| `debug.sbatch` | `debug-motion` | `debug` | `gpu:1` | 10 min |
| `prod.sbatch` | `prod-motion` | `gpu` | `gpu:rtx_pro_6000:1`, 32 CPU, 64G | 24 h |

Submit **from the repo root**. Relative `--output` paths resolve against the submit directory, and **Slurm does not create `logs/slurm/` for you**, so run `mkdir -p logs/slurm` once first. Run `uv sync` on the login node before submitting, because jobs don't sync. Override resources with sbatch flags, which take precedence over `#SBATCH` lines:

```bash
mkdir -p logs/slurm
sbatch scripts/slurm/debug.sbatch uv run scripts/demo.py video='logs/slurm/video-{job}/clip.mp4'
sbatch --nodelist=g4 --time=06:00:00 --array=0-7%8 scripts/slurm/prod.sbatch bash my_sweep.sh
# quick interactive check instead:
srun -p debug --gres=gpu:1 -c 2 --mem=4G -t 2 nvidia-smi
```

## Monitoring

```bash
squeue -u $USER                                   # your jobs
squeue -o "%.10i %.9P %.10u %.8T %.10M %R %b"     # all jobs + GPU request
sinfo -o "%P %a %l %D %N %G" ; sinfo -R           # gres per node; down/drain reasons
scontrol show job <jobid>                         # details incl. reason pending
scontrol show node g0                             # AllocTRES, FreeMem, state
tail -f logs/slurm/<name>-<jobid>.out
srun --jobid=<jobid> --overlap nvidia-smi         # peek at GPU of a running job
scancel <jobid>                                   # only your own jobs
```

## Gotchas

- **head0 has no GPU and no EGL.** It has no `nvidia-smi`, and `ldconfig -p | grep -i egl` returns nothing. Anything that needs CUDA or EGL must run through Slurm.
- **The compute node has EGL** (checked on g0): `libEGL.so.1`, `libEGL_nvidia.so.0`, `libEGL_mesa.so.0` and `libGL.so.1` are in `/lib64`. Whether `MUJOCO_GL=egl` rendering actually works was **not tested**. g1–g6 were not checked **(unverified)**.
- **Shared filesystem:** `/home` is NFSv4 from `10.0.0.1:/home`, mounted on head0 and on g0. That covers the repo, `.venv`, and uv's Python at `/home/mhyatt000/.local/share/uv/...`. The repo and `.venv/bin/python` (3.11.11) both work on g0. Other nodes are assumed to mount the same share **(unverified)**.
- `/home/mhyatt000` is a symlink to `/home/mhyatt`, so both paths reach the same directory.
- **`/tmp` is not shared.** It sits on each node's local root xfs: 70 GB on both head0 and g0. A file created in head0's `/tmp` was not visible on g0. Don't pass `/tmp` paths between the head node and jobs.
- `/home` is **96% full** (1.4 TB free of 29 TB). Keep large checkpoints and videos under control.
- TmpDisk=0 on every node, so a nonzero `--tmp` request cannot be satisfied.
- Other users keep long array jobs on g1–g2 (`l40s`) and on g3. The debug nodes and g4 were idle at survey time.
