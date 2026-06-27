#!/bin/bash
# Provision the avatar_studio (Sonic) stack on an EC2 Ubuntu 22.04 + CUDA GPU box.
#
# This installs EXACTLY what pipelines/avatar_studio/run.py and the worker's
# Sonic path need — matching the env-var/import contract in core/steps/*:
#
#   venv-sonic       → run.py launcher + Sonic + PuLID + CodeFormer + Kokoro + img2img + insightface
#   venv-chatterbox  → Chatterbox voice cloning (separate venv, isolated deps)
#   /home/ubuntu/Sonic       (SONIC_DIR)
#   /home/ubuntu/PuLID       (PULID_DIR)
#   /home/ubuntu/CodeFormer  (CODEFORMER_DIR)
#   /home/ubuntu/kokoro/     (KOKORO_MODEL_PATH / KOKORO_VOICES_PATH)
#
# Run as the ubuntu user:  bash deploy/install_sonic_stack.sh
#
# NOTE: this is intentionally for a DEDICATED test/CI box. Do not run it on the
# live production worker instance — the model downloads are large and the deps
# differ from the deployed MuseTalk/SVD stack.
set -euo pipefail

VENV_SONIC=/home/ubuntu/venv-sonic
VENV_PULID=/home/ubuntu/venv-pulid
VENV_CHATTERBOX=/home/ubuntu/venv-chatterbox
SONIC_DIR=/home/ubuntu/Sonic
PULID_DIR=/home/ubuntu/PuLID
CODEFORMER_DIR=/home/ubuntu/CodeFormer
KOKORO_DIR=/home/ubuntu/kokoro

echo "===== 1. System dependencies ====="
sudo apt-get update -q
sudo apt-get install -y -q git git-lfs ffmpeg libsndfile1 libportaudio2 build-essential wget
git lfs install || true

echo "===== 2. venv-sonic (launcher + Sonic + PuLID + CodeFormer + Kokoro) ====="
python3.10 -m venv "$VENV_SONIC"
PY="$VENV_SONIC/bin/python"
"$VENV_SONIC/bin/pip" install --upgrade pip wheel

echo "--- PyTorch (CUDA 12.1) ---"
"$VENV_SONIC/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "--- Core libs needed just to import run.py + steps ---"
# cv2/numpy/pillow are imported at module load by avatar_gen + face_composite + tts
"$VENV_SONIC/bin/pip" install \
    opencv-python-headless numpy pillow \
    insightface onnxruntime-gpu \
    diffusers transformers accelerate safetensors \
    kokoro-onnx soundfile

echo "===== 3. Sonic (audio-driven portrait animation) ====="
if [ ! -d "$SONIC_DIR" ]; then
  git clone https://github.com/jixiaozhong/Sonic.git "$SONIC_DIR"
fi
cd "$SONIC_DIR"
[ -f requirements.txt ] && "$VENV_SONIC/bin/pip" install -r requirements.txt || true
# Sonic uses diffusers 0.29.0; peft (pulled in by diffusers) needs hub>=0.25, and
# torch needs numpy<2. Pin both so Sonic inference doesn't break on import.
"$VENV_SONIC/bin/pip" install "huggingface_hub==0.25.2" "numpy<2"
# Sonic checkpoints (Sonic weights + SVD-XT base + whisper). Repo layout expects
# them under $SONIC_DIR/checkpoints. Confirm repo IDs against the Sonic README.
"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
import os
dst = os.path.join("$SONIC_DIR", "checkpoints")
os.makedirs(dst, exist_ok=True)
# Sonic released weights:
snapshot_download(repo_id="LeonJoe13/Sonic", local_dir=dst)
# SVD-XT image-to-video backbone used by Sonic:
snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                  local_dir=os.path.join(dst, "stable-video-diffusion-img2vid-xt"))
PYEOF

echo "===== 4. PuLID (identity-preserving portrait) — ISOLATED venv ====="
# PuLID pins diffusers==0.25.0, which clashes with Sonic's diffusers==0.29.0.
# So PuLID — and the img2img refine step, which also runs via PULID_VENV_PYTHON —
# get their own venv. The pipeline/CI must set:
#   export PULID_VENV_PYTHON=$VENV_PULID/bin/python
if [ ! -d "$PULID_DIR" ]; then
  git clone https://github.com/ToTheBeginning/PuLID.git "$PULID_DIR"
fi
python3.10 -m venv "$VENV_PULID"
"$VENV_PULID/bin/pip" install --upgrade pip wheel
"$VENV_PULID/bin/pip" install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
# PuLID reqs minus torch (pinned above) and gradio (demo-only, heavy)
grep -viE '^(torch|torchvision|torchaudio|gradio)' "$PULID_DIR/requirements.txt" > /tmp/pulid_reqs.txt
"$VENV_PULID/bin/pip" install -r /tmp/pulid_reqs.txt
# Pins discovered during bring-up (without these the pipeline crashes):
#   torchsde                — imported by pulid.utils, missing from requirements.txt
#   numpy==1.26.4           — torch 2.2.1 needs numpy<2 ("Numpy is not available")
#   huggingface_hub==0.25.2 — diffusers 0.25.0 imports cached_download (gone in hub 0.26+)
"$VENV_PULID/bin/pip" install torchsde "numpy==1.26.4" "huggingface_hub==0.25.2"
# PuLID downloads its ID-encoder + SDXL-Lightning weights on first load_pretrain().

echo "===== 5. CodeFormer (per-frame face restoration) ====="
if [ ! -d "$CODEFORMER_DIR" ]; then
  git clone https://github.com/sczhou/CodeFormer.git "$CODEFORMER_DIR"
fi
cd "$CODEFORMER_DIR"
"$VENV_SONIC/bin/pip" install -r requirements.txt || true
"$PY" basicsr/setup.py develop || true
# Pre-fetch CodeFormer + facelib detection weights (otherwise fetched on 1st run)
"$PY" scripts/download_pretrained_models.py facelib || true
"$PY" scripts/download_pretrained_models.py CodeFormer || true

echo "===== 6. RealVisXL img2img model (skin-texture refinement) ====="
# Pre-cache so first generation isn't a cold fetch (env: REALVIS_MODEL)
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SG161222/RealVisXL_V4.0")
PYEOF

echo "===== 7. Kokoro TTS fallback model files ====="
mkdir -p "$KOKORO_DIR"
wget -q -O "$KOKORO_DIR/kokoro-v0_19.onnx" \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget -q -O "$KOKORO_DIR/voices.bin" \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin

echo "===== 7b. Seed the CI test sample (Jenkins expects this exact path) ====="
# The smoke test uses /home/ubuntu/ci_test/sample.MOV as the pipeline input.
# Seed it from the PRIVATE CI bucket (not CloudFront-fronted, IAM-only access).
# Override CI_SAMPLE_S3 to use a different clip.
CI_SAMPLE_S3="${CI_SAMPLE_S3:-s3://avatar-graperoot-ci/ci/sample.MOV}"
mkdir -p /home/ubuntu/ci_test
if [ ! -f /home/ubuntu/ci_test/sample.MOV ]; then
  aws s3 cp "$CI_SAMPLE_S3" /home/ubuntu/ci_test/sample.MOV \
    || echo "WARN: could not fetch CI sample from $CI_SAMPLE_S3 — place /home/ubuntu/ci_test/sample.MOV manually"
fi

echo "===== 8. venv-chatterbox (voice cloning, isolated) ====="
python3.10 -m venv "$VENV_CHATTERBOX"
"$VENV_CHATTERBOX/bin/pip" install --upgrade pip wheel
"$VENV_CHATTERBOX/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
"$VENV_CHATTERBOX/bin/pip" install chatterbox-tts

echo ""
echo "===== DONE ====="
cat <<EOF

Add these to the worker/CI environment (e.g. /home/ubuntu/worker.env or Jenkins):
  export SONIC_DIR=$SONIC_DIR
  export PULID_DIR=$PULID_DIR
  export CODEFORMER_DIR=$CODEFORMER_DIR
  export SONIC_VENV_PYTHON=$VENV_SONIC/bin/python
  export PULID_VENV_PYTHON=$VENV_SONIC/bin/python
  export CHATTERBOX_VENV_PYTHON=$VENV_CHATTERBOX/bin/python
  export KOKORO_MODEL_PATH=$KOKORO_DIR/kokoro-v0_19.onnx
  export KOKORO_VOICES_PATH=$KOKORO_DIR/voices.bin

Smoke test (no CI needed):
  $VENV_SONIC/bin/python pipelines/avatar_studio/run.py \\
      /home/ubuntu/ci_test/sample.MOV /tmp/out.mp4 /tmp/run \\
      --text "Hello, this is a test." --scene studio --aspect 9:16

For CI also place the sample at /home/ubuntu/ci_test/sample.MOV.
EOF
