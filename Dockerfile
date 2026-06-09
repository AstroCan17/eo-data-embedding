# Minimal CPU image for the Phase 4 inference/demo service.
# For GPU embedding (Phase 1) run on the P40/Kaggle directly, or swap the base
# for an nvidia/cuda runtime and faiss-gpu.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for rasterio/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 7860
CMD ["python", "scripts/phase4_app.py"]
