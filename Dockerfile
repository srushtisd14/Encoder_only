FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip

RUN pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p \
    /app/results \
    /app/outputs \
    /app/huggingface_cache

ENV HF_HOME=/app/huggingface_cache

CMD ["python3", "run_experiments.py"]