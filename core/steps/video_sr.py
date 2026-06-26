"""Video super-resolution via RealESRGAN 2x upscale.

Pure background upscale — no face detection dependency, works on any video.
Uses Hallo2's bundled RealESRGAN weights + basicsr.
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

HALLO2_DIR = os.environ.get("HALLO2_DIR", "/home/ubuntu/hallo2")

# Prefer hallo2 venv (has basicsr), fall back to sonic venv if it exists
import os as _os
_hallo2_py = "/home/ubuntu/venv-hallo2/bin/python"
_sonic_py  = "/home/ubuntu/venv-sonic/bin/python"
VENV_PYTHON = _os.environ.get(
    "HALLO2_VENV_PYTHON",
    _hallo2_py if _os.path.exists(_hallo2_py) else _sonic_py,
)

_SCRIPT = r'''
import os, sys, cv2, torch, subprocess

HALLO2 = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]
upscale = int(sys.argv[4])

sys.path.insert(0, HALLO2)
os.chdir(HALLO2)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2,
    model_path=os.path.join(HALLO2, "pretrained_models/realesrgan/RealESRGAN_x2plus.pth"),
    model=model, tile=512, tile_pad=10, pre_pad=0, half=True,
)

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()

h, w = frames[0].shape[:2]
out_h, out_w = h * upscale, w * upscale
silent = output_path + ".silent.mp4"
writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
for frame in frames:
    out, _ = upsampler.enhance(frame, outscale=upscale)
    writer.write(out)
writer.release()

subprocess.run([
    "ffmpeg", "-y", "-i", silent, "-i", input_path,
    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16",
    "-c:a", "aac", "-shortest", output_path
], check=True, capture_output=True)
os.remove(silent)
print(f"SR_DONE:{output_path}", flush=True)
'''


def upscale_video(input_path: str, output_path: str, upscale: int = 2) -> str:
    """Apply RealESRGAN 2x upscale to every frame of a video.

    Args:
        input_path: Source .mp4
        output_path: Destination .mp4 (2x resolution, audio preserved)
        upscale: Scale factor (default 2)

    Returns:
        output_path
    """
    import tempfile as _tmpfile

    with _tmpfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_SCRIPT)
        script_path = f.name

    try:
        proc = subprocess.run(
            [VENV_PYTHON, script_path, HALLO2_DIR, input_path, output_path, str(upscale)],
            capture_output=True, text=True, timeout=900,
            encoding="utf-8", errors="replace",
            cwd=HALLO2_DIR,
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0 or "SR_DONE:" not in proc.stdout:
        log.error(f"video_sr stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"RealESRGAN upscale failed:\n{proc.stderr[-1500:]}")

    log.info(f"video_sr done: {output_path}")
    return output_path
