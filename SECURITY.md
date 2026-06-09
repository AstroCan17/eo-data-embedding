# Security Policy

## Reporting a vulnerability

Please report security issues privately by email to **candenizkaya17@gmail.com** rather than
opening a public issue. You'll get an acknowledgement within a few days.

## Notes

- **No secrets in the repo.** Credentials (e.g. a HuggingFace token) are read from the environment
  only — set `HF_TOKEN` in your shell/container, never commit it.
- Model **checkpoints and datasets are downloaded at runtime** (not stored in git); the
  `check-added-large-files` pre-commit hook blocks accidental commits of large binaries.
- `torch.load` is used to load the Clay checkpoint from the official HuggingFace repo
  (`made-with-clay/Clay`) — only load checkpoints you trust.
