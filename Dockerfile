FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# -----------------------------
# DOWNLOAD MODELS (ONLINE)
# -----------------------------

# Summary model (FLAN-T5-XXL)
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/flan-t5-xxl",
    local_dir="/models/hf/t5-summary",
    local_dir_use_symlinks=False
)
EOF

# Translation model (Marian)
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Helsinki-NLP/opus-mt-ru-en",
    local_dir="/models/hf/marian-ru-en",
    local_dir_use_symlinks=False
)
EOF

# -----------------------------
# ENFORCE OFFLINE AT RUNTIME
# -----------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf

ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# -----------------------------
# App
# -----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
