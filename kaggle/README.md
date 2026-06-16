# Kaggle GPU run — phase 5b change probe

The Clay encoder needs CUDA, so the change-detection follow-ups (patch-token distance maps +
supervised Δembedding probe, see `research/06-change-analysis.md` §5) run as a Kaggle GPU kernel.

## What's here

- `run_change_probe.py` — kernel entry. Installs Clay (pinned commit) + torchgeo, fetches Clay's
  band metadata and v1.5 checkpoint, downloads OSCD from the verified HF mirror, then runs
  `scripts/phase5b_change_probe.py` on GPU. Writes `change_probe_results.md` to the kernel output.
- `kernel-metadata.json` — GPU + internet on; mounts the `geo-embed-eo-src` code dataset.
- `dataset-metadata.json` — metadata for that code dataset.
- `push.sh` — stages `src/ scripts/ configs/` into the dataset, creates/versions it, pushes the kernel.

## Run

```bash
kaggle/push.sh                                              # build dataset + push kernel
kaggle kernels status candenizkaya/geo-embed-eo-change-probe
kaggle kernels output candenizkaya/geo-embed-eo-change-probe -p kaggle/_out   # when complete
```

Then copy the metrics table from `kaggle/_out/change_probe_results.md` into
`research/06-change-analysis.md` and the README. The kernel and code dataset are private.

## Notes

- Clay checkpoint comes from HuggingFace `made-with-clay/Clay` (`v1.5/clay-v1.5.ckpt`) — public,
  no token needed. If HF ever gates it, add it as a Kaggle dataset and list it in
  `dataset_sources` instead.
- The Clay commit pin (`f14e698…`) matches the Dockerfile for reproducibility.
- `_payload/` and `_out/` are local scratch — git-ignored.
