# Convenience wrappers around docker compose. No conda, no local venv needed.
#
#   make build         build the GPU dev image
#   make smoke         run the Phase-0 green-light smoke test (CPU-safe)
#   make sanity        run the Phase-0 sanity check
#   make clay-smoke    verify Clay loads + embeds both modalities (needs claymodel + ckpt)
#   make extract       run Phase-1 embedding extraction (needs GPU)
#   make cnn-baseline  train the supervised ResNet-18 baseline (Phase 3b, needs GPU)
#   make fetch-oscd    download the OSCD zips from the verified HF mirror (CPU-safe)
#   make change        run Phase-5 OSCD change detection (needs ckpt; GPU recommended)
#   make app           launch the Gradio demo (CPU image) on :7860
#   make shell         drop into a shell in the GPU dev container
#   make shell-cpu     shell in the CPU dev container (laptops without nvidia)
#   make gpu-check      verify torch sees the GPU inside the container
#   make test          run unit tests (pytest)  ·  make lint / make fmt  (ruff)

COMPOSE ?= docker compose

.PHONY: build build-cpu shell shell-cpu sanity smoke clay-smoke extract cnn-baseline fetch-oscd \
        change crossmodal app gpu-check test lint fmt clean

build:
	$(COMPOSE) build dev

build-cpu:
	$(COMPOSE) build app

shell:
	$(COMPOSE) run --rm dev bash

shell-cpu:
	$(COMPOSE) run --rm dev-cpu bash

sanity:
	$(COMPOSE) run --rm dev-cpu python scripts/phase0_sanity.py

smoke:
	$(COMPOSE) run --rm dev-cpu python scripts/phase0_smoke.py

clay-smoke:
	$(COMPOSE) run --rm dev python scripts/phase1_clay_smoke.py --device cuda

extract:
	$(COMPOSE) run --rm dev python scripts/phase1_extract.py

cnn-baseline:
	$(COMPOSE) run --rm dev python scripts/phase3_cnn_baseline.py --device cuda

fetch-oscd:
	$(COMPOSE) run --rm dev-cpu python scripts/fetch_oscd.py --root data/

change:
	$(COMPOSE) run --rm dev python scripts/phase5_change.py --root data/ --checkpoint v1.5/clay-v1.5.ckpt --device cuda

crossmodal:
	$(COMPOSE) run --rm dev python scripts/phase6_crossmodal.py --n 300 --checkpoint v1.5/clay-v1.5.ckpt --device cuda

app:
	$(COMPOSE) up app

gpu-check:
	$(COMPOSE) run --rm dev python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

clean:
	$(COMPOSE) down -v
