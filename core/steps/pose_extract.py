"""Extract DWPose skeleton sequence from user's consent video.

User records a ~30s video of themselves talking → we extract their
natural gesture skeleton → stored as .npz in S3 → reused at generation time
so MimicMotion animates with THEIR natural motions, not a generic pose.
"""
import os
import logging
import subprocess
import tempfile
import json

log = logging.getLogger(__name__)

MIMIC_DIR = os.environ.get("MIMICMOTION_DIR", "/home/ubuntu/MimicMotion")
VENV_PYTHON = os.environ.get("AVATAR_VENV_PYTHON_3D", "/home/ubuntu/venv-3d/bin/python")

# Extract DWPose keypoints from video and save as .npz + preview .mp4
_EXTRACT_SCRIPT = r'''
import sys, os, torch, json
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

mimic_dir  = sys.argv[1]
video_path = sys.argv[2]
out_npz    = sys.argv[3]
out_video  = sys.argv[4]   # optional skeleton overlay video, pass "" to skip
max_frames = int(sys.argv[5]) if sys.argv[5] else 0

import decord
import numpy as np
from mimicmotion.dwpose.preprocess import get_video_pose

decord.bridge.set_bridge("torch")
vr = decord.VideoReader(video_path)
fps = vr.get_avg_fps()
total = len(vr)

if max_frames > 0 and total > max_frames:
    # sample evenly to max_frames
    indices = [int(i * total / max_frames) for i in range(max_frames)]
else:
    indices = list(range(total))

frames = vr.get_batch(indices).permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)

# get_video_pose needs ref_image to set the scale — use first frame
ref_image = frames[0]

# Write a temp video of the sampled frames for DWPose processing
import cv2, tempfile
tmp_vid = tempfile.mktemp(suffix=".mp4")
h, w = frames.shape[2], frames.shape[3]
writer = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for f in frames:
    frame_np = (f.permute(1, 2, 0).numpy() * 255).astype("uint8")
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)
writer.release()

pose_pixels = get_video_pose(tmp_vid, ref_image, sample_stride=1)
os.unlink(tmp_vid)

# pose_pixels shape: (T, C, H, W), values [0,1]
np.savez_compressed(out_npz,
    pose=pose_pixels.numpy(),
    fps=fps,
    original_hw=np.array([h, w]),
    frame_indices=np.array(indices),
)

# Optionally write skeleton overlay video
if out_video:
    writer2 = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in pose_pixels:
        frame_np = (f.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer2.write(frame_bgr)
    writer2.release()

print(f"DONE:{out_npz}:frames={pose_pixels.shape[0]}:fps={fps}", flush=True)
'''


def extract_pose(video_path: str, output_npz: str,
                 skeleton_preview_path: str = None,
                 max_frames: int = 300) -> dict:
    """Extract DWPose skeleton from user video, save as .npz.

    Returns metadata dict: {frames, fps, hw}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_EXTRACT_SCRIPT)
        script_path = f.name

    env = os.environ.copy()
    cudnn_lib = "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"
    env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    log.info(f"Extracting DWPose from {video_path} → {output_npz}")
    proc = subprocess.run(
        [VENV_PYTHON, script_path,
         MIMIC_DIR, video_path, output_npz,
         skeleton_preview_path or "", str(max_frames)],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=MIMIC_DIR,
    )
    os.unlink(script_path)

    if proc.returncode != 0 or "DONE:" not in proc.stdout:
        log.error(f"pose_extract stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"Pose extraction failed:\n{proc.stderr[-1500:]}")

    # Parse metadata from DONE line
    done_line = [l for l in proc.stdout.splitlines() if l.startswith("DONE:")][0]
    parts = dict(p.split("=") for p in done_line.split(":")[2:])
    meta = {
        "frames": int(parts.get("frames", 0)),
        "fps": float(parts.get("fps", 15)),
        "npz_path": output_npz,
    }
    log.info(f"Pose extracted: {meta['frames']} frames @ {meta['fps']}fps")
    return meta
