"""Hallo2 audio-driven talking head synthesis.

Takes a 512x512 portrait image + audio → lipsync video.
This is the primary avatar animation step — does lipsync in one pass.
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

HALLO2_DIR = os.environ.get("HALLO2_DIR", "/home/ubuntu/hallo2")
VENV_PYTHON = os.environ.get("HALLO2_VENV_PYTHON", "/home/ubuntu/venv-hallo2/bin/python")


def run_hallo2(portrait_path: str, audio_path: str, output_path: str,
               pose_weight: float = 1.0, face_weight: float = 1.2,
               lip_weight: float = 2.0, face_expand_ratio: float = 1.2) -> str:
    """Run Hallo2 to generate a lipsync video from a portrait + audio.

    Args:
        portrait_path: 512x512 face portrait (square, front-facing)
        audio_path: WAV audio file (any sample rate, Hallo2 resamples internally)
        output_path: Destination .mp4

    Returns:
        output_path
    """
    from pipeline.gpu_env import gpu_env

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write per-job config pointing save_path to our tmpdir
        config_path = os.path.join(tmpdir, "inference.yaml")
        base_config = os.path.join(HALLO2_DIR, "configs/inference/long.yaml")
        with open(base_config) as f:
            config_content = f.read()
        lines = []
        for line in config_content.splitlines():
            if line.strip().startswith("save_path:"):
                lines.append(f"save_path: {tmpdir}/out")
            elif line.strip().startswith("inference_steps:"):
                lines.append("inference_steps: 80")
            else:
                lines.append(line)
        with open(config_path, "w") as f:
            f.write("\n".join(lines))

        env = gpu_env()
        cmd = [
            VENV_PYTHON,
            os.path.join(HALLO2_DIR, "scripts/inference_long.py"),
            "-c", config_path,
            "--source_image", portrait_path,
            "--driving_audio", audio_path,
            "--pose_weight", str(pose_weight),
            "--face_weight", str(face_weight),
            "--lip_weight", str(lip_weight),
            "--face_expand_ratio", str(face_expand_ratio),
        ]

        log.info(f"Hallo2: {portrait_path} + {audio_path} -> {output_path}")
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=900,
            env=env, encoding="utf-8", errors="replace",
            cwd=HALLO2_DIR,
        )

        if proc.returncode != 0:
            log.error(f"Hallo2 stderr: {proc.stderr[-2000:]}")
            raise RuntimeError(f"Hallo2 failed:\n{proc.stderr[-1500:]}")

        # Find the output mp4 — inference_long.py writes merge_video.mp4
        import glob
        results = glob.glob(os.path.join(tmpdir, "out/*/merge_video.mp4"))
        if not results:
            raise RuntimeError(f"Hallo2 produced no output in {tmpdir}/out/")

        import shutil
        shutil.copy2(results[0], output_path)

    log.info(f"Hallo2 done: {output_path}")
    return output_path


def prepare_portrait(face_image_path: str, output_path: str) -> str:
    """Resize/crop a face image to 512x512 portrait format required by Hallo2."""
    subprocess.run([
        "ffmpeg", "-y", "-i", face_image_path,
        "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512",
        output_path,
    ], check=True, capture_output=True)
    return output_path
