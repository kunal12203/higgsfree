#!/bin/bash
# Install EchoMimicV2 + F5-TTS on EC2 Ubuntu 22.04 with CUDA GPU.
# Run as ubuntu user on the EC2 instance.
# Usage: bash install_pipeline.sh
set -e

VENV=/home/ubuntu/venv-avatar
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

echo "===== 1. System dependencies ====="
sudo apt-get update -q
sudo apt-get install -y -q git ffmpeg libsndfile1 libportaudio2

echo "===== 2. Python venv ====="
python3.10 -m venv $VENV
source $VENV/bin/activate

pip install --upgrade pip wheel

echo "===== 3. PyTorch (CUDA 12.1) ====="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "===== 4. EchoMimicV2 ====="
if [ ! -d /home/ubuntu/echomimic_v2 ]; then
  git clone https://github.com/antgroup/echomimic_v2 /home/ubuntu/echomimic_v2
fi
cd /home/ubuntu/echomimic_v2
pip install -r requirements.txt
# Download pretrained weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BadToBest/EchoMimicV2',
    local_dir='pretrained_weights',
    local_dir_use_symlinks=False,
)
"

echo "===== 5. F5-TTS ====="
pip install f5-tts

echo "===== 6. Whisper (for voice profile transcription) ====="
pip install faster-whisper

echo "===== 7. InsightFace (face extraction) ====="
pip install insightface onnxruntime-gpu

echo "===== 8. Misc deps ====="
pip install soundfile torchaudio moviepy omegaconf diffusers accelerate einops av

echo ""
echo "===== DONE ====="
echo "Test with:"
echo "  source $VENV/bin/activate"
echo "  cd /home/ubuntu/echomimic_v2"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
