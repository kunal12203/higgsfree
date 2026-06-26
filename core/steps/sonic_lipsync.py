"""Sonic (Tencent, CVPR 2025) — audio-driven portrait animation.

SVD-based end-to-end pipeline: portrait image + audio → talking head video.
No context window stitching = no glitches. Built-in RIFE frame interpolation.

~5 min for 12s video on A10G (7x faster than EchoMimicV1 full model).
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

SONIC_DIR = os.environ.get("SONIC_DIR", "/home/ubuntu/Sonic")
VENV_PYTHON = os.environ.get("SONIC_VENV_PYTHON", "/home/ubuntu/venv-sonic/bin/python")

_SCRIPT = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

image_path = sys.argv[2]
audio_path = sys.argv[3]
output_path = sys.argv[4]
dynamic_scale = float(sys.argv[5])

from sonic import Sonic

pipe = Sonic(0)

face_info = pipe.preprocess(image_path, expand_ratio=0.5)
if face_info['face_num'] < 0:
    raise RuntimeError(f"No face detected in image: {image_path}")

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
pipe.process(
    image_path,
    audio_path,
    output_path,
    min_resolution=512,
    inference_steps=25,
    dynamic_scale=dynamic_scale,
)
print(f"SONIC_DONE:{output_path}", flush=True)
'''


def run_sonic(
    image_path: str,
    audio_path: str,
    output_path: str,
    dynamic_scale: float = 1.0,
) -> str:
    """Generate talking head video from portrait image + audio using Sonic.

    Args:
        image_path: Portrait image (ideally 512x512+ with clear face)
        audio_path: Speech audio (.wav)
        output_path: Output .mp4 path
        dynamic_scale: Head motion intensity (0.0=still, 1.0=normal, 2.0=exaggerated)

    Returns:
        output_path on success
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_SCRIPT)
        script_path = f.name

    try:
        proc = subprocess.run(
            [VENV_PYTHON, script_path, SONIC_DIR, image_path, audio_path, output_path, str(dynamic_scale)],
            capture_output=True, text=True, timeout=2400,
            encoding="utf-8", errors="replace",
            cwd=SONIC_DIR,
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0 or "SONIC_DONE:" not in proc.stdout:
        log.error(f"Sonic stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"Sonic inference failed:\n{proc.stderr[-1500:]}")

    log.info(f"Sonic done: {output_path}")
    return output_path
