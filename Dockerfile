FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Qwen 2.5 7B Instruct (FP16 – stable)
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/models/hf/qwen",
    local_dir_use_symlinks=False
)
EOF

# Marian RU → EN
RUN python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Helsinki-NLP/opus-mt-ru-en",
    local_dir="/models/hf/marian-ru-en",
    local_dir_use_symlinks=False
)
EOF

ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
