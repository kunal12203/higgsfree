FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ffmpeg libsndfile1 libsndfile1-dev \
    git wget curl gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install Python dependencies (torch with CUDA)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk /opt/MuseTalk && \
    pip install --no-cache-dir -r /opt/MuseTalk/requirements.txt || true

COPY . .

ENV MUSETALK_DIR=/opt/MuseTalk

CMD ["python", "worker.py"]
