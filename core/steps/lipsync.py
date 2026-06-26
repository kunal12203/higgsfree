"""Lip-sync pipeline using LatentSync (ByteDance).

Two entry points:
  run_lipsync_on_video(video, audio, output) — lip-sync a video
  run_lipsync_on_image(image, audio, output) — create static video from image, then lip-sync
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

LATENTSYNC_DIR = os.environ.get("LATENTSYNC_DIR", "/home/ubuntu/LatentSync")
VENV_PYTHON = os.environ.get("LATENTSYNC_VENV_PYTHON", "/home/ubuntu/venv-latentsync/bin/python")


def run_lipsync_on_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Apply LatentSync lip sync to a video."""
    from pipeline.gpu_env import gpu_env
    env = gpu_env()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = LATENTSYNC_DIR + (":" + existing_pp if existing_pp else "")

    ckpt_path = os.path.join(LATENTSYNC_DIR, "checkpoints/latentsync_unet.pt")
    config_path = os.path.join(LATENTSYNC_DIR, "configs/unet/stage2.yaml")

    cmd = [
        VENV_PYTHON, os.path.join(LATENTSYNC_DIR, "scripts/inference.py"),
        "--unet_config_path", config_path,
        "--inference_ckpt_path", ckpt_path,
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", "20",
        "--guidance_scale", "1.5",
    ]

    log.info(f"LatentSync: {video_path} + {audio_path} -> {output_path}")
    proc = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=600, env=env,
        encoding="utf-8", errors="replace",
        cwd=LATENTSYNC_DIR,
    )

    if proc.returncode != 0:
        log.error(f"LatentSync stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"LatentSync failed:\n{proc.stderr[-1500:]}")

    if not os.path.exists(output_path):
        raise RuntimeError(f"LatentSync did not produce output: {output_path}")

    log.info(f"LatentSync done: {output_path}")
    return output_path


def run_lipsync_on_image(face_image_path: str, audio_path: str, output_path: str) -> str:
    """Create a static video from image, then apply lip sync."""
    # Get audio duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 5.0

    tmp_video = tempfile.mktemp(suffix=".mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", face_image_path,
        "-i", audio_path,
        "-t", str(duration),
        "-vf", "scale=512:512",
        "-c:v", "libx264", "-r", "25", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        tmp_video,
    ], check=True, capture_output=True)

    try:
        run_lipsync_on_video(tmp_video, audio_path, output_path)
    finally:
        if os.path.exists(tmp_video):
            os.unlink(tmp_video)

    return output_path
