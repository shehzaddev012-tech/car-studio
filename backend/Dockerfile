FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, SAM2, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ai.txt .

# CPU-only PyTorch first; SAM2 with --no-build-isolation avoids 20+ min pip build-deps download
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir lpips==0.1.4 "huggingface-hub>=0.26.0" \
    && pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/sam2.git

COPY . .

# Pre-download SAM2 checkpoint at build time (optional — speeds first job)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('facebook/sam2.1-hiera-base-plus', 'sam2.1_hiera_base_plus.pt')" \
    || true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
