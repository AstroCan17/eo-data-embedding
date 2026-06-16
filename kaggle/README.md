# Kaggle GPU run — phase 5b change probe

The Clay encoder needs CUDA, so the change-detection follow-ups (patch-token distance maps +
supervised Δembedding probe, see `research/06-change-analysis.md` §5) run as a Kaggle GPU kernel.

## What's here

- `run_change_probe.py` — kernel entry. Clones the repo (`main`) from GitHub at run time using the
  PAT in the private `gh-pat` dataset, installs Clay (pinned commit) + torchgeo, fetches Clay's
  band metadata and v1.5 checkpoint, downloads OSCD from the verified HF mirror, then runs
  `scripts/phase5b_change_probe.py` on GPU. Writes `change_probe_results.md` to the kernel output.
- `kernel-metadata.json` — GPU + internet on; mounts the `gh-pat` token dataset.
- `push.sh` — pushes the kernel (no source upload — the kernel clones the repo itself).

## Prerequisites

- Private Kaggle dataset **`candenizkaya/gh-pat`** holding a GitHub PAT with read access to the
  repo. The script auto-detects the file (first file in the mount) and accepts plain text or JSON
  (`{"token": "..."}` / `pat` / `GITHUB_TOKEN`).
- The change-probe code must be on `main` (merge the PR first) — the kernel clones `-b main`.

## Run

```bash
kaggle/push.sh                                              # push kernel
kaggle kernels status candenizkaya/geo-embed-eo-change-probe
kaggle kernels output candenizkaya/geo-embed-eo-change-probe -p kaggle/_out   # when complete
```

Then copy the metrics table from `kaggle/_out/change_probe_results.md` into
`research/06-change-analysis.md` and the README. The kernel is private.

## Notes

- The PAT is never printed: the clone runs with `stdout/stderr` suppressed and the token-bearing
  URL is never logged.
- Clay checkpoint comes from HuggingFace `made-with-clay/Clay` (`v1.5/clay-v1.5.ckpt`) — public.
- The Clay commit pin (`f14e698…`) matches the Dockerfile for reproducibility.
- `_out/` is local scratch — git-ignored.
