# GPU run — phase 5b change probe (Kaggle or Colab)

The Clay encoder needs CUDA, so the change-detection follow-ups (patch-token distance maps +
supervised Δembedding probe, see `research/06-change-analysis.md` §5) run on a GPU. The runner
(`run_change_probe.py`) is platform-agnostic — use **Kaggle** (below) or **Colab**
(`colab/change_probe.ipynb`) when the Kaggle GPU quota is busy with another project.

## What's here

- `run_change_probe.py` — kernel entry. Clones the repo (`main`) from GitHub at run time using the
  PAT in the private `gh-pat` dataset, installs Clay (pinned commit) + torchgeo, fetches Clay's
  band metadata and v1.5 checkpoint, downloads OSCD from the verified HF mirror, then runs
  `scripts/phase5b_change_probe.py` on GPU. Writes `change_probe_results.md` to the kernel output.
- `kernel-metadata.json` — GPU + internet on; mounts the `gh-pat` token dataset.
- `push.sh` — pushes the kernel (no source upload — the kernel clones the repo itself).

## Prerequisites

- A GitHub PAT with read access to the repo, supplied one of two ways:
  - **Kaggle:** private dataset **`candenizkaya/gh-pat`**. The runner scans every mounted dataset
    (mount-name-agnostic, dotfiles included), prefers a file whose name looks like a token, and
    accepts plain text or JSON (`{"token": "..."}` / `pat` / `GITHUB_TOKEN`).
  - **Colab / local:** the `GH_PAT` (or `GITHUB_TOKEN`) env var — takes priority over dataset
    scanning. On Colab set it from a Secret; see `colab/change_probe.ipynb`.
- The change-probe code must be on `main` (merge the PR first) — both runners clone `-b main`.

## Run on Kaggle

```bash
kaggle/push.sh                                              # push kernel
kaggle kernels status candenizkaya/geo-embed-eo-change-probe
kaggle kernels output candenizkaya/geo-embed-eo-change-probe -p kaggle/_out   # when complete
```

Then copy the metrics table from `kaggle/_out/change_probe_results.md` into
`research/06-change-analysis.md` and the README. The kernel is private.

## Run on Colab

Open `colab/change_probe.ipynb`, pick a GPU runtime, add a `GH_PAT` Colab Secret, and run all
cells. Results land at `/content/change_probe_results.md` (rendered inline by the last cell).
Same code, separate GPU pool — handy when the Kaggle weekly quota is reserved for another project.

## Notes

- The PAT is never printed: the clone runs with `stdout/stderr` suppressed and the token-bearing
  URL is never logged.
- Clay checkpoint comes from HuggingFace `made-with-clay/Clay` (`v1.5/clay-v1.5.ckpt`) — public.
- The Clay commit pin (`f14e698…`) matches the Dockerfile for reproducibility.
- `_out/` is local scratch — git-ignored.
