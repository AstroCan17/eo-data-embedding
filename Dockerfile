# GPU dev image — pip-only (no conda), CUDA 12.1.
# Runs every phase: Phase-0 smoke (CPU) and Phase-1 embedding extraction (GPU).
# CUDA 12.1 torch wheels cover both Pascal (P40, sm_61) and Turing (T4, sm_75).
# On a CPU-only host the image still runs — the CUDA libs simply go unused.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

# System deps: python + libs rasterio/opencv/faiss need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 \
        git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA torch FIRST from the cu121 index so the later -r requirements
# (torch>=2.1) is already satisfied and pip never pulls a CPU build over it.
RUN pip install --upgrade pip && \
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# The rest of the tree is bind-mounted in dev (see docker-compose.yml),
# but copy it so the image also works standalone.
COPY . .

CMD ["bash"]
