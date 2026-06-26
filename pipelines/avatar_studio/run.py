#!/usr/bin/env python3
"""Avatar Studio pipeline runner.

Usage:
    python pipelines/avatar_studio/run.py <consent_video> <output.mp4> [workdir]
        --text "Script text to speak"
        [--scene portrait|studio|cafe|outdoor|desk]
        [--aspect 9:16|1:1|16:9|4:5|4:3]

Steps:
    1. Extract best face frame from consent video
    2. Generate PuLID studio portrait (+ img2img refinement)
    3. Extract Chatterbox voice profile
    4. Synthesize speech in cloned voice
    5. Crop head+shoulders for Sonic
    6. Sonic lipsync
    7. CodeFormer polish
    8. Composite animated face onto full scene
    9. Mux audio → final output
"""
import sys
import os
import argparse
import logging
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add repo root to path so core/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.steps.avatar_gen import (
    extract_best_face_frame, crop_head_shoulders,
    generate_scene_avatar, SCENE_PRESETS, ASPECT_SIZES,
)
from core.steps.voiceclone import extract_voice_profile, synthesize_with_cloned_voice
from core.steps.tts import synthesize
from core.steps.sonic_lipsync import run_sonic
from core.steps.codeformer_polish import polish_video
from core.steps.face_composite import composite_face_onto_reference


def step_done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def run(consent_video: str, output_path: str, workdir: str,
        text: str, scene: str = "studio", aspect: str = "9:16"):

    os.makedirs(workdir, exist_ok=True)

    face_jpg    = os.path.join(workdir, "face.jpg")
    avatar_png  = os.path.join(workdir, f"avatar_{scene}_{aspect.replace(':','x')}.png")
    voice_dir   = os.path.join(workdir, "voice_profile")
    audio_out   = os.path.join(workdir, "cloned_audio.wav")
    cropped_png = os.path.join(workdir, "avatar_crop.png")
    sonic_out   = os.path.join(workdir, "sonic_face.mp4")
    polished    = os.path.join(workdir, "polished.mp4")
    composited  = os.path.join(workdir, "composited.mp4")

    crop_for_sonic = SCENE_PRESETS.get(scene, SCENE_PRESETS["studio"])["crop_for_sonic"]

    # ── Step 1: Extract best face frame ───────────────────────────────────────
    if not step_done(face_jpg):
        log.info("Step 1/9 — Extract best face frame")
        extract_best_face_frame(consent_video, face_jpg)
    else:
        log.info("Step 1/9 — skipped (cached)")

    # ── Step 2: Generate avatar ────────────────────────────────────────────────
    if not step_done(avatar_png):
        log.info(f"Step 2/9 — Generate avatar (scene={scene}, aspect={aspect})")
        generate_scene_avatar(face_jpg, avatar_png, scene_name=scene, aspect=aspect)
    else:
        log.info("Step 2/9 — skipped (cached)")

    # ── Step 3: Extract voice profile ─────────────────────────────────────────
    ref_wav = os.path.join(voice_dir, "ref_audio.wav")
    if not step_done(ref_wav):
        log.info("Step 3/9 — Extract voice profile")
        try:
            extract_voice_profile(consent_video, voice_dir)
        except Exception as e:
            log.warning(f"Voice profile extraction failed: {e}")
    else:
        log.info("Step 3/9 — skipped (cached)")

    # ── Step 4: Synthesize speech ──────────────────────────────────────────────
    if not step_done(audio_out):
        log.info("Step 4/9 — Synthesize cloned voice")
        if os.path.exists(ref_wav):
            try:
                synthesize_with_cloned_voice(text, voice_dir, audio_out)
            except Exception as e:
                log.warning(f"Chatterbox failed, falling back to Kokoro TTS: {e}")
                with open(audio_out, "wb") as f:
                    f.write(synthesize(text))
        else:
            log.warning("No voice profile — using Kokoro TTS fallback")
            with open(audio_out, "wb") as f:
                f.write(synthesize(text))
    else:
        log.info("Step 4/9 — skipped (cached)")

    # ── Step 5: Crop for Sonic ─────────────────────────────────────────────────
    sonic_input = avatar_png
    if not crop_for_sonic:
        if not step_done(cropped_png):
            log.info("Step 5/9 — Crop head+shoulders")
            try:
                crop_head_shoulders(avatar_png, cropped_png)
            except Exception as e:
                log.warning(f"Crop failed, using full portrait: {e}")
                shutil.copy(avatar_png, cropped_png)
        else:
            log.info("Step 5/9 — skipped (cached)")
        sonic_input = cropped_png
    else:
        log.info("Step 5/9 — skipped (portrait mode)")

    # ── Step 6: Sonic lipsync ──────────────────────────────────────────────────
    if not step_done(sonic_out):
        log.info("Step 6/9 — Sonic lipsync")
        run_sonic(sonic_input, audio_out, sonic_out, dynamic_scale=1.0)
    else:
        log.info("Step 6/9 — skipped (cached)")

    # ── Step 7: CodeFormer polish ──────────────────────────────────────────────
    if not step_done(polished):
        log.info("Step 7/9 — CodeFormer polish")
        try:
            polish_video(sonic_out, polished, fidelity=0.7)
        except Exception as e:
            log.warning(f"CodeFormer failed (non-fatal): {e}")
            shutil.copy(sonic_out, polished)
    else:
        log.info("Step 7/9 — skipped (cached)")

    # ── Step 8: Composite face onto scene ─────────────────────────────────────
    if not crop_for_sonic:
        if not step_done(composited):
            log.info("Step 8/9 — Composite animated face onto scene")
            composite_face_onto_reference(polished, cropped_png, composited,
                                          scene_image=avatar_png)
        else:
            log.info("Step 8/9 — skipped (cached)")
        video_before_mux = composited
    else:
        log.info("Step 8/9 — skipped (portrait mode)")
        video_before_mux = polished

    # ── Step 9: Mux audio ─────────────────────────────────────────────────────
    log.info("Step 9/9 — Mux audio")
    audio_dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_out],
        capture_output=True, text=True,
    ).stdout.strip() or "0")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_before_mux, "-i", audio_out,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        "-t", f"{audio_dur:.3f}", "-shortest",
        output_path,
    ], check=True, capture_output=True)

    log.info(f"Done: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("consent_video")
    parser.add_argument("output")
    parser.add_argument("workdir", nargs="?", default="/tmp/avatar_studio_run")
    parser.add_argument("--text", required=True)
    parser.add_argument("--scene", default="studio", choices=list(SCENE_PRESETS.keys()))
    parser.add_argument("--aspect", default="9:16", choices=list(ASPECT_SIZES.keys()))
    args = parser.parse_args()
    run(args.consent_video, args.output, args.workdir,
        text=args.text, scene=args.scene, aspect=args.aspect)
