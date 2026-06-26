"""Animate a still image into video using Stable Video Diffusion XT."""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/home/ubuntu/models")
SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd-xt")
VENV_PYTHON = os.environ.get("AVATAR_VENV_PYTHON", "/home/ubuntu/venv-avatar/bin/python")

_SCRIPT = '''
import sys, os, torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

model_path = sys.argv[1]
image_path = sys.argv[2]
output_path = sys.argv[3]
num_frames = int(sys.argv[4])

print("Loading SVD-XT...", flush=True)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
print("Loaded!", flush=True)

img = Image.open(image_path).convert("RGB")
# Crop to upper 60% (head+shoulders) so face stays large for lip-sync
if img.height > img.width:
    crop_h = int(img.height * 0.6)
    img = img.crop((0, 0, img.width, crop_h))
image = img.resize((576, 1024))

print(f"Generating {num_frames} frames...", flush=True)
output = pipe(
    image=image,
    num_frames=num_frames,
    num_inference_steps=25,
    decode_chunk_size=4,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, output_path, fps=8)
print(f"DONE:{output_path}", flush=True)
'''


def animate_image(image_path: str, output_path: str, duration_seconds: float = 3.0) -> str:
    """Animate a still image into a short video clip using SVD-XT."""
    num_frames = min(25, max(14, int(duration_seconds * 8)))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_SCRIPT)
        script_path = f.name

    env = os.environ.copy()
    cudnn_lib = "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"
    env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    log.info(f"Animating image -> {num_frames} frames ({duration_seconds:.1f}s): {image_path}")
    result = subprocess.run(
        [VENV_PYTHON, script_path, SVD_MODEL_PATH, image_path, output_path, str(num_frames)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    os.unlink(script_path)

    if result.returncode != 0 or "DONE:" not in result.stdout:
        raise RuntimeError(f"SVD animation failed:\n{result.stderr[-1000:]}\n{result.stdout[-400:]}")

    log.info(f"Animation done: {output_path}")
    return output_path


def loop_to_duration(clip_path: str, target_duration: float, output_path: str) -> str:
    """Loop a short animated clip to fill a longer duration using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", clip_path,
        "-t", str(target_duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
