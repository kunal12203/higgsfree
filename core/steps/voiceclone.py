"""Chatterbox TTS voice cloning — zero-shot clone from reference audio.

Flow:
  1. extract_voice_profile(audio_path, out_dir)
     - Trims reference to best 8-10s window (Chatterbox optimal)
     - Saves ref_audio.wav into out_dir

  2. synthesize_with_cloned_voice(text, voice_profile_dir, output_wav)
     - Loads ref_audio.wav from profile dir
     - Runs Chatterbox to generate output_wav in the cloned voice
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

VENV_PYTHON = os.environ.get("CHATTERBOX_VENV_PYTHON", "/home/ubuntu/venv-chatterbox/bin/python")

_CLONE_SCRIPT = r'''
import sys, os, torch
import torchaudio

ref_audio_in = sys.argv[1]
out_dir      = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)
ref_wav_out = os.path.join(out_dir, "ref_audio.wav")

# Chatterbox native sample rate
target_sr = 24000
window_sec = 10
keep = int(window_sec * target_sr)

wav, sr = torchaudio.load(ref_audio_in)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != target_sr:
    wav = torchaudio.functional.resample(wav, sr, target_sr)

total = wav.shape[1]

# Pick the loudest 10s window (best speech energy, avoids silence/noise sections)
# Slide in 1s steps, skip first 1s and last 1s
step = target_sr
best_start = int(1 * target_sr)
best_rms = -1
for start in range(int(1 * target_sr), max(int(1 * target_sr) + 1, total - keep), step):
    chunk = wav[:, start:start + keep]
    rms = chunk.pow(2).mean().sqrt().item()
    if rms > best_rms:
        best_rms = rms
        best_start = start

wav = wav[:, best_start:best_start + keep]
torchaudio.save(ref_wav_out, wav, target_sr)
print(f"PROFILE_DONE:{ref_wav_out}", flush=True)
'''

_SYNTH_SCRIPT = r'''
import sys, os, torch, torchaudio

ref_audio  = sys.argv[1]
gen_text   = sys.argv[2]
output_wav = sys.argv[3]

from chatterbox.tts import ChatterboxTTS
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

wav = model.generate(
    gen_text,
    audio_prompt_path=ref_audio,
    exaggeration=0.3,
    cfg_weight=0.5,
)

os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)

# 50ms fade-in to kill the pop/click Chatterbox sometimes emits at start
fade_samples = int(0.05 * model.sr)
if wav.shape[-1] > fade_samples:
    ramp = torch.linspace(0.0, 1.0, fade_samples, device=wav.device)
    wav[..., :fade_samples] *= ramp

torchaudio.save(output_wav, wav, model.sr)
print(f"SYNTH_DONE:{output_wav}", flush=True)
'''


def extract_voice_profile(ref_audio_path: str, out_dir: str) -> str:
    """Prepare voice profile from reference audio/video. Returns out_dir with ref_audio.wav."""
    os.makedirs(out_dir, exist_ok=True)

    # If input is a video file, extract audio track first so torchaudio can read it
    ext = os.path.splitext(ref_audio_path)[1].lower()
    if ext in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
        extracted_wav = os.path.join(out_dir, "_extracted_audio.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_audio_path, "-vn",
             "-ar", "24000", "-ac", "1", "-f", "wav", extracted_wav],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(extracted_wav):
            raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr[-1000:]}")
        ref_audio_path = extracted_wav

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_CLONE_SCRIPT)
        script_path = f.name

    proc = subprocess.run(
        [VENV_PYTHON, script_path, ref_audio_path, out_dir],
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )
    os.unlink(script_path)

    if proc.returncode != 0 or "PROFILE_DONE:" not in proc.stdout:
        log.error(f"voice profile stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"Voice profile extraction failed:\n{proc.stderr[-1500:]}")

    log.info(f"Voice profile saved to {out_dir}")
    return out_dir


def synthesize_with_cloned_voice(text: str, voice_profile_dir: str, output_wav: str) -> str:
    """Generate speech in the cloned voice. Returns output_wav path."""
    ref_audio = os.path.join(voice_profile_dir, "ref_audio.wav")
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Voice profile not found at {voice_profile_dir}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_SYNTH_SCRIPT)
        script_path = f.name

    proc = subprocess.run(
        [VENV_PYTHON, script_path, ref_audio, text, output_wav],
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    os.unlink(script_path)

    if proc.returncode != 0 or "SYNTH_DONE:" not in proc.stdout:
        log.error(f"Chatterbox stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"Chatterbox synthesis failed:\n{proc.stderr[-1500:]}")

    log.info(f"Chatterbox synthesis done: {output_wav}")
    return output_wav
