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

# Clay foundation model (needs py>=3.11). Install after torch; re-pin cu121 torch if it got clobbered.
RUN python -m pip install claymodel && \
    ( python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" || \
      python -m pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 )

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install -e .

# The rest of the tree is bind-mounted in dev (see docker-compose.yml),
# but copy it so the image also works standalone.
COPY . .

CMD ["bash"]
