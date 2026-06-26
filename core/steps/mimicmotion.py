"""MimicMotion full-body animation — SVD-based pose-driven video generation."""
import os
import logging
import subprocess
import tempfile
import random

log = logging.getLogger(__name__)

MIMIC_DIR = os.environ.get("MIMICMOTION_DIR", "/home/ubuntu/MimicMotion")
VENV_PYTHON = os.environ.get("AVATAR_VENV_PYTHON_3D", "/home/ubuntu/venv-3d/bin/python")
POSE_VIDEOS = ["pose1", "pose2", "pose3"]  # extend as we add more motion templates

_SCRIPT = r'''
import sys, os, torch, numpy as np
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

mimic_dir  = sys.argv[1]
ref_image  = sys.argv[2]
pose_video = sys.argv[3]
output_path = sys.argv[4]
num_frames = int(sys.argv[5])

from omegaconf import OmegaConf
from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.dwpose.preprocess import get_video_pose
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.io import write_video
from PIL import Image as PILImage
import math
from constants import ASPECT_RATIO

device = torch.device("cuda")

# ── 1. Load pipeline, move ALL submodules to GPU ──────────────────────────────
cfg = OmegaConf.create({
    "base_model_path": os.path.join(mimic_dir, "../models/svd-xt"),
    "ckpt_path": "models/MimicMotion_1-1.pth",
})
pipeline = create_pipeline(cfg, device)
pipeline.unet.to(device=device, dtype=torch.float16)
pipeline.vae.to(device=device, dtype=torch.float16)
pipeline.image_encoder.to(device=device, dtype=torch.float16)
pipeline.pose_net.to(device=device, dtype=torch.float16)

# ── 2. Load + resize reference image to 9:16 portrait ─────────────────────────
image_pixels = pil_to_tensor(pil_loader(ref_image))   # (C, H, W) uint8
h, w = image_pixels.shape[-2:]
if h > w:   # portrait
    w_target = 576
    h_target = int(w_target / ASPECT_RATIO // 64) * 64   # 1024
else:        # landscape — force portrait
    h_target = 1024
    w_target = 576
h_w_ratio = float(h) / float(w)
if h_w_ratio < h_target / w_target:
    h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
else:
    h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
image_pixels = resize(image_pixels, [h_resize, w_resize])
image_pixels = center_crop(image_pixels, [h_target, w_target])   # (C, H, W) uint8

# PIL image for pipeline (it normalises internally)
ref_pil = PILImage.fromarray(image_pixels.permute(1,2,0).numpy())

# numpy HWC uint8 for DWPose
ref_np = image_pixels.permute(1,2,0).numpy()   # HWC uint8

# ── 3. Extract pose skeleton from video (DWPose, CPU onnxruntime is fine) ─────
# sample_stride=2 on an 8s/30fps video → ~120 frames
pose_pixels = get_video_pose(pose_video, ref_np, sample_stride=2)
# pose_pixels: numpy or tensor (T, C, H, W), values in [0, 255] or [0, 1]
if isinstance(pose_pixels, np.ndarray):
    pose_pixels = torch.from_numpy(pose_pixels)
if pose_pixels.dtype == torch.uint8:
    pose_pixels = pose_pixels.float() / 255.0   # → float32 [0,1]
elif pose_pixels.max() > 1.0:
    pose_pixels = pose_pixels / 255.0

L = min(num_frames, pose_pixels.shape[0])
# pipeline.pose_net is fp16, so cast pose tensor to fp16
pose_tensor = pose_pixels[:L].to(dtype=torch.float16)

# ── 4. Run MimicMotion diffusion ───────────────────────────────────────────────
torch.manual_seed(42)
result = pipeline(
    ref_pil,
    pose_tensor,
    height=h_target,
    width=w_target,
    num_frames=L,
    num_inference_steps=20,
    min_guidance_scale=1.0,
    max_guidance_scale=3.0,
    noise_aug_strength=0.02,
    decode_chunk_size=8,
    device=device,
    output_type="pt",          # returns (batch, T, C, H, W) float tensor [0,1]
    generator=torch.manual_seed(42),
)
# result.frames shape: (1, T, C, H, W) float [0,1]
frames_pt = result.frames[0]   # (T, C, H, W) float [0,1]

# ── 5. Save: write_video needs (T, H, W, C) uint8 ─────────────────────────────
frames_uint8 = (frames_pt.permute(0, 2, 3, 1) * 255).clamp(0, 255).byte().cpu()
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
write_video(output_path, frames_uint8, fps=15)
print(f"DONE:{output_path}", flush=True)
'''


def run_mimicmotion(avatar_image_path: str, output_path: str,
                    pose_name: str = None, pose_video_path: str = None,
                    num_frames: int = 48) -> str:
    """Animate avatar image using MimicMotion (SVD-based full-body animation).

    pose_video_path: path to user's own consent video (preferred — their natural gestures)
    pose_name: fallback to preset pose video (pose1/pose2/pose3)
    """
    if pose_video_path and os.path.exists(pose_video_path):
        pose_video = pose_video_path
        log.info(f"Using user's own pose video: {pose_video_path}")
    else:
        if pose_name is None:
            pose_name = random.choice(POSE_VIDEOS)
        pose_video = os.path.join(MIMIC_DIR, f"assets/example_data/videos/{pose_name}.mp4")
        if not os.path.exists(pose_video):
            pose_video = os.path.join(MIMIC_DIR, "assets/example_data/videos/pose1.mp4")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_SCRIPT)
        script_path = f.name

    env = os.environ.copy()
    cudnn_lib = "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"
    env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    log.info(f"MimicMotion: {num_frames} frames, pose={pose_name}, ref={avatar_image_path}")
    proc = subprocess.run(
        [VENV_PYTHON, script_path,
         MIMIC_DIR, avatar_image_path, pose_video, output_path, str(num_frames)],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=MIMIC_DIR,
    )
    os.unlink(script_path)

    if proc.returncode != 0 or "DONE:" not in proc.stdout:
        log.error(f"MimicMotion stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"MimicMotion failed:\n{proc.stderr[-1500:]}")

    log.info(f"MimicMotion done: {output_path}")
    return output_path
