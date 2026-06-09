# GPU dev image — pip-only (no conda), CUDA 12.1, Python 3.11.
# Runs every phase: Phase-0 smoke (CPU) and Phase-1 embedding extraction (GPU).
# CUDA 12.1 torch wheels cover both Pascal (P40, sm_61) and Turing (T4, sm_75).
# Python 3.11 (via deadsnakes) is required by `claymodel` (Clay foundation model).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

# Python 3.11 + system libs rasterio/opencv/faiss need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        build-essential \
        libgl1 libglib2.0-0 git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA torch FIRST from the cu121 index so later installs don't pull a CPU build.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Clay foundation model (needs py>=3.11). The PyPI `claymodel` 1.5.0 wheel is mis-packaged
# (flat modules with `from src.model import ...` that don't resolve), so install from the
# official GitHub repo, which ships the proper `claymodel` package. Re-pin cu121 torch if a
# transitive dep clobbered it.
RUN python -m pip install "git+https://github.com/Clay-foundation/model.git" && \
    ( python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" || \
      python -m pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 )

# Clay's module reads configs/metadata.yaml (band wavelengths/means/stds). It isn't shipped in
# the package, so fetch it to a fixed path; ClayEmbedder looks here via CLAY_METADATA search.
RUN mkdir -p /opt/clay && \
    curl -sSL https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml \
        -o /opt/clay/metadata.yaml

# SSL4EO-S12 v1.1 streaming loader (cross-modal). The repo is NOT a pip package (no setup.py),
# so we clone it and put it on PYTHONPATH; its deps are in requirements.txt above.
RUN git clone --depth 1 https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1.git /opt/ssl4eos12
ENV PYTHONPATH=/opt/ssl4eos12:${PYTHONPATH}

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install -e .

# The rest of the tree is bind-mounted in dev (see docker-compose.yml),
# but copy it so the image also works standalone.
COPY . .

CMD ["bash"]
