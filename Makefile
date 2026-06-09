# Convenience wrappers around docker compose. No conda, no local venv needed.
#
#   make build         build the GPU dev image
#   make smoke         run the Phase-0 green-light smoke test (CPU-safe)
#   make sanity        run the Phase-0 sanity check
#   make clay-smoke    verify Clay loads + embeds both modalities (needs claymodel + ckpt)
#   make extract       run Phase-1 embedding extraction (needs GPU)
#   make app           launch the Gradio demo (CPU image) on :7860
#   make shell         drop into a shell in the GPU dev container
#   make shell-cpu     shell in the CPU dev container (laptops without nvidia)
#   make gpu-check      verify torch sees the GPU inside the container

COMPOSE ?= docker compose

.PHONY: build build-cpu shell shell-cpu sanity smoke clay-smoke extract app gpu-check clean

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

app:
	$(COMPOSE) up app

gpu-check:
	$(COMPOSE) run --rm dev python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

clean:
	$(COMPOSE) down -v
