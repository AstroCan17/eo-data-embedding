# Contributing

## Dev setup

Two options — Docker (reproducible, GPU) or a local venv (CPU, fast for tests/lint):

```bash
# Docker (everything baked: Clay, SSL4EO loader, CUDA) — for the real GPU phases
make build && make shell

# or local venv — enough for tests + lint
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,test]"
pre-commit install
```

## Quality gates (run before pushing — CI enforces them)

```bash
make lint     # ruff check + format --check
make fmt      # ruff format (apply)
make test     # pytest (unit tests, CPU, no downloads)
```

`pre-commit run --all-files` runs the same hooks (ruff, large-file guard, etc.).

## Phase ↔ make-target map

| Phase | Make target | Script |
|---|---|---|
| 0 sanity / smoke | `make sanity` / `make smoke` | `scripts/phase0_*.py` |
| Clay integration | `make clay-smoke` | `scripts/phase1_clay_smoke.py` |
| 1 extract | `make extract` | `scripts/phase1_extract.py` |
| 2 search / 3 probe | (run in container) | `scripts/phase{2,3}_*.py` |
| 4 demo | `make app` | `scripts/phase4_app.py` |
| 5 change / 6 cross-modal | `make fetch-oscd` + `make change` / `make crossmodal` | `scripts/phase{5,6}_*.py` |

Defaults live in `configs/default.yaml`; CLI flags override. See `research/` for the design
decision records behind each dataset/model/phase choice.

## Conventions
- Python 3.11, ruff (lint + format), line length 110.
- Library code in `src/geo_embed_eo/` is typed and side-effect-free; runnable scripts in `scripts/`.
- Branch off `main`, open a PR, keep CI green.
