"""CodeFormer per-frame face polish.

Runs CodeFormer on every frame of a video at fidelity=0.7.
Falls back to the unpolished video if CodeFormer is not installed.
"""
import os
import logging
import subprocess
import tempfile
import glob

log = logging.getLogger(__name__)

CODEFORMER_DIR  = os.environ.get("CODEFORMER_DIR",  "/home/ubuntu/CodeFormer")
SONIC_VENV_PYTHON = os.environ.get("SONIC_VENV_PYTHON", "/home/ubuntu/venv-sonic/bin/python")

_POLISH_SCRIPT = r'''
import sys, os

codeformer_dir = sys.argv[1]
frames_dir     = sys.argv[2]
output_dir     = sys.argv[3]
fidelity       = float(sys.argv[4])

# Prioritize CodeFormer's bundled basicsr (patched) over PyPI's
basicsr_dir = os.path.join(codeformer_dir, 'basicsr')
sys.path.insert(0, basicsr_dir)
sys.path.insert(0, codeformer_dir)
os.chdir(codeformer_dir)

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.codeformer_arch import CodeFormer

from facelib.utils.face_restoration_helper import FaceRestoreHelper

device = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = os.path.join(codeformer_dir, "weights", "CodeFormer", "codeformer.pth")
if not os.path.exists(weights_path):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    load_file_from_url(url, model_dir=os.path.dirname(weights_path), file_name="codeformer.pth")

net = CodeFormer(
    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt = torch.load(weights_path, map_location=device)["params_ema"]
net.load_state_dict(ckpt)
net.eval()

face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model="retinaface_resnet50",
    save_ext="png",
    use_parse=True,
    device=device,
)

os.makedirs(output_dir, exist_ok=True)
frame_files = sorted([
    f for f in os.listdir(frames_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for fname in frame_files:
    img_path = os.path.join(frames_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        continue

    face_helper.clean_all()
    face_helper.read_image(img)
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    if num_det_faces == 0:
        cv2.imwrite(os.path.join(output_dir, fname), img)
        continue

    face_helper.align_warp_face()
    for cropped_face in face_helper.cropped_faces:
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        with torch.no_grad():
            output = net(cropped_face_t, w=fidelity, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=None)
    cv2.imwrite(os.path.join(output_dir, fname), restored_img)

print(f"POLISH_DONE:{len(frame_files)}", flush=True)
'''


def polish_video(input_video: str, output_video: str, fidelity: float = 0.7) -> str:
    """Apply CodeFormer per-frame polish to a video. Returns output_video path.

    Falls back to input_video if CodeFormer is unavailable.
    """
    if not os.path.isdir(CODEFORMER_DIR):
        log.warning(f"CodeFormer not found at {CODEFORMER_DIR} — skipping polish")
        return input_video

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir  = os.path.join(tmpdir, "frames")
        polished_dir = os.path.join(tmpdir, "polished")
        os.makedirs(frames_dir, exist_ok=True)

        # Extract frames
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_video,
             os.path.join(frames_dir, "frame_%06d.png")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning(f"Frame extraction failed — skipping CodeFormer: {result.stderr[-500:]}")
            return input_video

        frame_count = len(glob.glob(os.path.join(frames_dir, "*.png")))
        if frame_count == 0:
            log.warning("No frames extracted — skipping CodeFormer")
            return input_video

        log.info(f"CodeFormer polishing {frame_count} frames (fidelity={fidelity})...")

        # Write and run the polish script in venv-sonic
        import tempfile as _tf
        with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(_POLISH_SCRIPT)
            script_path = f.name

        proc = subprocess.run(
            [SONIC_VENV_PYTHON, script_path,
             CODEFORMER_DIR, frames_dir, polished_dir, str(fidelity)],
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        os.unlink(script_path)

        if proc.returncode != 0 or "POLISH_DONE:" not in proc.stdout:
            log.warning(f"CodeFormer failed (non-fatal): {proc.stderr[-1000:]}")
            return input_video

        # Get original fps from input
        fps_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", input_video],
            capture_output=True, text=True,
        )
        fps = "25"
        if fps_probe.returncode == 0 and fps_probe.stdout.strip():
            raw = fps_probe.stdout.strip()
            if "/" in raw:
                num, den = raw.split("/")
                fps = str(round(int(num) / int(den), 3))
            else:
                fps = raw

        # Reassemble with original audio
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        assemble = subprocess.run(
            ["ffmpeg", "-y",
             "-framerate", fps,
             "-i", os.path.join(polished_dir, "frame_%06d.png"),
             "-i", input_video,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "320k",
             output_video],
            capture_output=True, text=True,
        )
        if assemble.returncode != 0:
            log.warning(f"CodeFormer reassemble failed: {assemble.stderr[-500:]}")
            return input_video

        log.info(f"CodeFormer polish done: {output_video}")
        return output_video
