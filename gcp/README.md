# GCP GPU run — phase 5b change probe

A third GPU backend for the change probe, alongside Kaggle and Colab. Uses the **same**
platform-agnostic runner (`kaggle/run_change_probe.py`) — `gcp/run_change_probe.sh` just
provisions a T4 VM, runs it, pulls the results, and deletes the VM.

Use this when you want to spend the GCP trial credit (€/$) instead of the Kaggle weekly GPU quota,
or for longer jobs than Kaggle's session limits allow.

## One-time account setup (you must do these — they involve billing / account changes)

1. **Activate a full account.** The free trial cannot create GPUs. In the console, click
   **Activate** (top banner) and add a payment method. Your trial credit stays valid and is spent
   first; you won't be charged until it runs out and you opt in.
2. **Enable the Compute Engine API** (console: *APIs & Services* → enable *Compute Engine API*),
   if not already on.
3. **Request GPU quota.** *IAM & Admin → Quotas*, filter for **GPUs (all regions)**, and request
   an increase to at least **1**. New accounts often start at 0. Approval is usually quick but can
   take from minutes to a couple of days.

## Local setup (one-time)

- Install the `gcloud` CLI and authenticate:
  ```bash
  gcloud auth login
  gcloud config set project <your-project-id>
  ```

## Run

```bash
export GH_PAT=ghp_xxx                  # GitHub PAT with read access to the repo
bash gcp/run_change_probe.sh           # provision T4 -> run -> fetch -> delete VM
```

Results land in `gcp/_out/change_probe_results.md`. Then copy the metrics table into
`research/06-change-analysis.md` and the README.

### Knobs (env vars)

| var | default | notes |
|-----|---------|-------|
| `ZONE` | `europe-west4-a` | must have T4 capacity |
| `GPU` | `nvidia-tesla-t4` | e.g. `nvidia-l4` for a newer card |
| `MACHINE` | `n1-standard-4` | T4 pairs with N1 |
| `SPOT` | `1` | Spot VM = cheaper; set `0` for an on-demand VM |
| `PROJECT` | current gcloud project | |

## Cost & safety

- The VM is **deleted on every exit** (success, failure, or Ctrl+C) via a `trap`, so a forgotten
  GPU instance can't drain the credit. A T4 Spot VM is roughly **$0.15–0.40/hour**; the probe runs
  well under an hour, so a full run costs cents against the trial credit.
- The PAT is handed to the VM over stdin into a `umask 077` file (never on the command line, never
  in instance metadata), read into an env var, then deleted.
- `gcp/_out/` is local scratch — git-ignored.
