"""Extract per-frame pose .npy files from a video for EchoMimicV2.

Mirrors the format used in EMTD_dataset/preprocess.py:
  Each .npy has keys: bodies, hands, hands_score, faces, faces_score,
  draw_pose_params, plus normalized candidate coordinates.
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

ECHOMIMIC_DIR = os.environ.get("ECHOMIMIC_DIR", "/home/ubuntu/echomimic_v2")
VENV_PYTHON = os.environ.get("AVATAR_VENV_PYTHON", "/home/ubuntu/venv-avatar/bin/python")

_SCRIPT = r'''
import sys, os, cv2, numpy as np, torch
from pathlib import Path

echomimic_dir = sys.argv[1]
video_path    = sys.argv[2]
out_dir       = sys.argv[3]
max_frames    = int(sys.argv[4]) if len(sys.argv) > 4 else 300

sys.path.insert(0, echomimic_dir)
os.chdir(echomimic_dir)

from src.models.dwpose.dwpose_detector import DWposeDetector

os.makedirs(out_dir, exist_ok=True)

model_det  = os.path.join(echomimic_dir, "pretrained_weights/DWPose/yolox_l.onnx")
model_pose = os.path.join(echomimic_dir, "pretrained_weights/DWPose/dw-ll_ucoco_384.onnx")
detector = DWposeDetector(model_det=model_det, model_pose=model_pose, device="cuda")

TARGET = 768

cap = cv2.VideoCapture(video_path)
orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
stride = max(1, round(orig_fps / 24.0))  # sample at ~24fps

idx = 0
saved = 0
while cap.isOpened() and saved < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if idx % stride != 0:
        idx += 1
        continue

    # Center-crop to square then resize to 768x768
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    frame_sq = frame_rgb[y0:y0+side, x0:x0+side]
    frame_resized = cv2.resize(frame_sq, (TARGET, TARGET))

    # DWPose detection
    pose = detector(frame_resized)

    # draw_pose_params for a TARGET×TARGET square (no padding needed)
    pose['draw_pose_params'] = [TARGET, TARGET, 0, TARGET, 0, TARGET]
    pose['num'] = saved

    np.save(os.path.join(out_dir, f"{saved}.npy"), pose)
    saved += 1
    idx += 1

cap.release()
detector.release_memory()
print(f"POSE_DONE:{out_dir}:{saved}", flush=True)
'''


def extract_pose_for_echomimic(video_path: str, out_dir: str, max_frames: int = 300) -> str:
    """Extract per-frame DWPose .npy files from video. Returns out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_SCRIPT)
        script_path = f.name

    from pipeline.gpu_env import gpu_env
    env = gpu_env()

    proc = subprocess.run(
        [VENV_PYTHON, script_path, ECHOMIMIC_DIR, video_path, out_dir, str(max_frames)],
        capture_output=True, text=True, timeout=600, env=env, encoding="utf-8", errors="replace",
        cwd=ECHOMIMIC_DIR,
    )
    os.unlink(script_path)

    if proc.returncode != 0 or "POSE_DONE:" not in proc.stdout:
        log.error(f"pose_extract stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"Pose extraction failed:\n{proc.stderr[-1500:]}")

    log.info(f"Pose extraction done: {out_dir}")
    return out_dir
