# melody-miner — GPU-capable Docker image (CUDA 12.1 + Python 3.12 + Ubuntu 22.04).
#
# Build:
#   docker build -t melody-miner .
#
# Run (Gradio UI, GPU):
#   docker run --gpus all -p 7860:7860 \
#     -v ./checkpoints:/app/checkpoints \
#     -v ./references:/app/references \
#     -v ./input:/app/input \
#     -v ./output:/app/output \
#     melody-miner
#
# Run (CLI, GPU):
#   docker run --gpus all \
#     -v ./checkpoints:/app/checkpoints \
#     -v ./references:/app/references \
#     -v ./input:/app/input \
#     -v ./output:/app/output \
#     melody-miner run.py \
#       --input input/your_vocal.wav \
#       --m2a-checkpoint checkpoints/m2a/best.ckpt \
#       --tnp-checkpoint checkpoints/tnp/latest.pt \
#       --reference references/target.wav \
#       --out output/run1
#
# CPU-only (no NVIDIA GPU):
#   Change the --index-url below from cu121 to cpu.
#   Remove the `deploy: resources:` block from docker-compose.yml.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face model cache (persisted via named volume in docker-compose.yml)
    HF_HOME=/app/.cache/huggingface

# ── System deps ────────────────────────────────────────────────────────────────
# Python 3.12 via deadsnakes PPA (Ubuntu 22.04 ships 3.10 by default).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        # audio I/O and MIDI render
        ffmpeg libsndfile1 libportaudio2 fluidsynth && \
    rm -rf /var/lib/apt/lists/*

# pip for python3.12 (not shipped with the deadsnakes package)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Convenience alias so scripts can use bare `python`
RUN ln -s /usr/bin/python3.12 /usr/local/bin/python

WORKDIR /app

# ── Python deps (split into layers for cache efficiency) ──────────────────────
# requirements.txt and pyproject.toml change rarely → copy first.
COPY requirements.txt pyproject.toml ./

RUN python -m pip install -U pip setuptools wheel

# PyTorch CUDA 12.1 build (~2.5 GB — cached after first build)
RUN python -m pip install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# Core + Branch A/B deps (numpy forced to wheel — see requirements.txt header)
RUN python -m pip install --only-binary=numpy -r requirements.txt

# basic-pitch without TensorFlow — ONNX backend auto-selected (see requirements.txt)
RUN python -m pip install --no-deps "basic-pitch==0.4.0"

# ── Source code (changes often → late in Dockerfile for cache hits) ───────────
COPY src/ src/
COPY configs/ configs/
COPY app.py run.py ./

RUN python -m pip install -e . --no-deps

# ── Fallback GM soundfont (low quality; override by bind-mounting soundfonts/) ─
RUN mkdir -p soundfonts && \
    curl -fL -o soundfonts/default.sf2 \
        "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2" \
    2>/dev/null \
    || echo "[warn] soundfont download failed — bind-mount soundfonts/<your>.sf2 at runtime"

# ── Runtime bind-mount directories ────────────────────────────────────────────
RUN mkdir -p checkpoints/m2a checkpoints/tnp references input output

# Gradio default port; app.py already binds to 0.0.0.0 by default
EXPOSE 7860

ENTRYPOINT ["python"]
CMD ["app.py"]
