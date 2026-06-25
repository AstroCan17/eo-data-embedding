# 07 — Engineering notes: getting the change probe onto real hardware

**TL;DR.** The phase 5b change probe needs a real accelerator (frozen Clay is a ViT-L). Getting it
to run end to end on rented cloud hardware — without leaving an orphaned GPU burning trial credit —
turned into the hardest *engineering* part of the project: ten environment fixes, a capacity/quota
wall that killed the GPU path entirely, and a deliberate pivot to CPU. This note records what broke,
why, and how it was solved, because the debugging is as much a portfolio signal as the science.

## 1. Goal and constraints

Run `scripts/phase5b_change_probe.py` (frozen Clay v1.5 over OSCD) on a real machine and pull the
results back, under three hard constraints:

- **No orphaned GPUs.** A forgotten GPU VM quietly drains the trial credit. Every run must delete
  its VM on *any* exit — success, failure, or Ctrl-C.
- **No source upload.** The repo is private at build time; the runner clones it at run time with a
  short-lived PAT handed over out-of-band (never on a command line, never printed).
- **One file, three platforms.** The same runner (`kaggle/run_change_probe.py`) must work on
  Kaggle, Colab, and a bare GCP Deep Learning VM, with environment-variable fallbacks for tokens,
  repo path, and work dir. The GCP wrapper (`gcp/run_change_probe.sh`) provisions → runs → tears down.

## 2. The GCP environment: ten iterations

A bare Deep Learning VM is not a Kaggle/Colab notebook. Each assumption that holds on Kaggle broke
on GCP and had to be fixed explicitly. In rough order:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `claymodel` installs as `UNKNOWN-0.0.0` | Image was Ubuntu 22.04 / Python 3.10; claymodel needs **py ≥ 3.11** | Pin **Ubuntu 24.04 / py3.12** image (`pytorch-2-9-cu129-ubuntu-2404-nvidia-580`) |
| 2 | `python` not found / wrong interpreter | DL image puts Python at `/opt/conda/bin/python`, not on a stable name | Probe `/opt/conda/bin/python`, fall back to `python3`/`python` |
| 3 | `huggingface_hub` missing | Ships on Kaggle/Colab images, absent on a bare DL VM | Pin it explicitly in the install list (no-op where present) |
| 4 | `pip install` refuses (PEP 668) and "cannot uninstall typing_extensions" | 24.04 system Python is externally managed; pip can't touch apt-owned packages | Run inside a **venv created with `--system-site-packages`**: CUDA torch stays visible from the system, everything pip-installed lands in the user venv. No sudo, no PEP 668, no uninstall conflict |
| 5 | `venv` creation fails (no ensurepip) | Distro Python ships `venv` without `ensurepip` | `apt-get install python3.X-venv` (version derived at runtime) **before** creating the venv |
| 6 | SSH hangs forever when the VM dies mid-command | A Spot preemption kills the box; the TCP connection waits out the full kernel timeout | SSH keepalive (`ServerAliveInterval=30`, `ServerAliveCountMax=4`, `ConnectTimeout=30`) → dead VM drops in ~2 min |
| 7 | `create` fails with `ZONE_RESOURCE_POOL_EXHAUSTED` | GPU capacity is **per-zone** and frequently exhausted | Try a list of zones in order; first one that accepts the VM wins |

(Two earlier dead ends are folded into the above: running pip as root system-wide — abandoned for
the cleaner venv approach — and an initial wrong image family / conda path.)

The recurring lesson: **a managed notebook hides a dozen environment decisions.** Reproducing it on
raw infrastructure means making every one of them explicit.

## 3. The capacity & quota wall

With the environment fixed, the GPU itself became the blocker — and this is where the project hit a
hard external limit, not a code problem.

- **On-demand T4 was exhausted across every EU zone tried** (europe-west4-a/b/c, europe-west1-b/c/d).
  Plain on-demand provisioning simply returned stockouts.
- **Spot T4 had capacity, so the first runs used Spot** — but a Spot VM was **preempted ~1 h in**,
  mid-install. Spot is fine for a job measured in minutes, risky for anything longer.
- **On-demand L4 (`g2-standard-4`) in `us-central1`** did provision (zone `-b` accepted it) and the
  probe ran through install, train-split embedding, and the GPU forward pass — proving the whole
  pipeline on a real accelerator. It only stopped on a *data* bug (see §4).
- **The retry never got a second GPU.** A later attempt found `us-central1-a/b/c` all in
  `STOCKOUT` (capacity swings minute to minute), and meanwhile the account's **single** GPU slot
  was taken: `GPUS_ALL_REGIONS = 1`, fully used by an unrelated training VM.
- **A quota bump 1 → 2 was auto-denied** within the same minute — the documented behaviour for a
  trial project. A second concurrent GPU was simply not available.

### The pivot: CPU

Rather than wait on capacity or a quota appeal that wouldn't come, the runner gained a **`DEVICE=cpu`
mode**: no accelerator attached, no NVIDIA driver, so it needs **no GPU quota at all**. The CUDA
build of PyTorch in the image imports fine on a GPU-less box and runs on CPU; the encoder code
already threads `device` through (`map_location` + `.to(device)`), so nothing else changed. On an
`e2-standard-8`, the probe ran **end to end in ~11 minutes** — no quota, no capacity gamble, no
preemption. That run produced the numbers in [`06-change-analysis.md`](06-change-analysis.md) §7.

The takeaway is a real engineering judgement call: when the constraint is *quota*, not *compute
time*, and the job is short, **the right move can be to drop the accelerator entirely** rather than
fight the quota system.

## 4. The tiling bug (`241×385`)

The L4 run failed on the OSCD *test* split with
`ValueError: image (241×385) is smaller than the tile size (256)`. `tile_image()` rejected any scene
smaller than one tile — but reflect padding only requires each pad amount to be `< its dimension`,
and `241×385` needs pads of `15`/`127`, both well within bounds, so it was perfectly tileable. The
check was stricter than the actual constraint. Fix: bind the check to the real reflect limit and
fall back to **replicate** (edge) padding for scenes genuinely too small for reflect. Verified
locally on five synthetic shapes before re-running — cheaper than spending a GPU slot to find out.

## 5. Lessons

1. **Tear-down is a feature, not an afterthought.** The `trap … EXIT` that deletes the VM on every
   exit path is the single most important line in the script — it's what makes iterating on a
   trial budget safe.
2. **Hand secrets over out-of-band.** The PAT is piped to a `umask 077` file over SSH and deleted
   after clone; it never appears in a command line, a log, or the repo.
3. **Verify cheap, fail cheap.** The tiling fix was confirmed on the local CPU in seconds; the
   device-agnostic encoder was confirmed by reading the code, not by burning a GPU run.
4. **Know when a wall is external.** The quota denial wasn't a bug to fix — recognising it as a
   fixed limit and pivoting to CPU was faster than any amount of retrying.
5. **A managed notebook is a stack of hidden decisions.** Porting to raw infra surfaced ten of
   them; the portable runner now encodes those decisions so the next environment is one flag away.
