"""EchoMimicV2 semi-body animation — audio-driven lip-sync + hand gestures."""
import os
import logging
import subprocess
import tempfile
import random

log = logging.getLogger(__name__)

ECHOMIMIC_DIR = os.environ.get("ECHOMIMIC_DIR", "/home/ubuntu/echomimic_v2")
VENV_PYTHON = os.environ.get("AVATAR_VENV_PYTHON", "/home/ubuntu/venv-avatar/bin/python")
POSE_TEMPLATES = ["01", "02", "03", "04"]

# Inline inference script — mirrors infer_acc.py but for a single image/audio/pose
_SCRIPT = r'''
import sys, os, random, torch, numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from PIL import Image
from moviepy.editor import VideoFileClip, AudioFileClip
from diffusers import AutoencoderKL, DDIMScheduler

echomimic_dir = sys.argv[1]
ref_image_path = sys.argv[2]
audio_path     = sys.argv[3]
pose_dir       = sys.argv[4]
output_path    = sys.argv[5]
num_frames     = int(sys.argv[6])

sys.path.insert(0, echomimic_dir)
os.chdir(echomimic_dir)

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.utils.dwpose_util import draw_pose_select_v2
from src.models.pose_encoder import PoseEncoder

device = "cuda"
weight_dtype = torch.float16

cfg_path = os.path.join(echomimic_dir, "configs/prompts/infer_acc.yaml")
infer_cfg_path = os.path.join(echomimic_dir, "configs/inference/inference_v2.yaml")
config = OmegaConf.load(cfg_path)
infer_config = OmegaConf.load(infer_cfg_path)

W, H = 768, 768
fps = 24
steps = 6
cfg_scale = 1.0
sample_rate = 16000

print("Loading models...", flush=True)

vae = AutoencoderKL.from_pretrained(
    os.path.join(echomimic_dir, "pretrained_weights/sd-vae-ft-mse")
).to(device, dtype=weight_dtype)

reference_unet = UNet2DConditionModel.from_pretrained(
    os.path.join(echomimic_dir, "pretrained_weights/sd-image-variations-diffusers"),
    subfolder="unet",
).to(device, dtype=weight_dtype)
reference_unet.load_state_dict(
    torch.load(os.path.join(echomimic_dir, "pretrained_weights/reference_unet.pth"), map_location="cpu")
)

denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
    os.path.join(echomimic_dir, "pretrained_weights/sd-image-variations-diffusers"),
    os.path.join(echomimic_dir, "pretrained_weights/motion_module_acc.pth"),
    subfolder="unet",
    unet_additional_kwargs=infer_config.unet_additional_kwargs,
).to(device, dtype=weight_dtype)
denoising_unet.load_state_dict(
    torch.load(os.path.join(echomimic_dir, "pretrained_weights/denoising_unet_acc.pth"), map_location="cpu"),
    strict=False
)

pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(device, dtype=weight_dtype)
pose_net.load_state_dict(torch.load(os.path.join(echomimic_dir, "pretrained_weights/pose_encoder.pth"), map_location="cpu"))

audio_processor = load_audio_model(
    model_path=os.path.join(echomimic_dir, "pretrained_weights/audio_processor/tiny.pt"),
    device=device
)

scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

pipe = EchoMimicV2Pipeline(
    vae=vae,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    audio_guider=audio_processor,
    pose_encoder=pose_net,
    scheduler=scheduler,
).to(device, dtype=weight_dtype)

print("Models loaded.", flush=True)

# Build pose tensor
ref_img_pil = Image.open(ref_image_path).convert("RGB")
audio_clip = AudioFileClip(audio_path)
L = min(num_frames, int(audio_clip.duration * fps), len(os.listdir(pose_dir)))
print(f"Generating {L} frames...", flush=True)

pose_list = []
for index in range(L):
    tgt_musk = np.zeros((W, H, 3), dtype=np.uint8)
    npy_path = os.path.join(pose_dir, f"{index}.npy")
    detected_pose = np.load(npy_path, allow_pickle=True).tolist()
    imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))
    tgt_musk[rb:re, cb:ce, :] = im
    tgt_musk_pil = Image.fromarray(tgt_musk).convert("RGB")
    pose_list.append(torch.tensor(np.array(tgt_musk_pil)).to(device, dtype=weight_dtype).permute(2, 0, 1) / 255.0)

poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
audio_clip = audio_clip.set_duration(L / fps)

generator = torch.manual_seed(42)
video = pipe(
    ref_img_pil,
    audio_path,
    poses_tensor[:, :, :L, ...],
    W, H, L, steps, cfg_scale,
    generator=generator,
    audio_sample_rate=sample_rate,
    context_frames=12,
    fps=fps,
    context_overlap=3,
    start_idx=0,
).videos

# Save video + merge audio
tmp_path = output_path + ".nowav.mp4"
save_videos_grid(video[:, :, :L, :, :], tmp_path, n_rows=1, fps=fps)
video_clip = VideoFileClip(tmp_path)
final = video_clip.set_audio(AudioFileClip(audio_path).subclip(0, min(audio_clip.duration, video_clip.duration)))
final.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
os.remove(tmp_path)

print(f"DONE:{output_path}", flush=True)
'''


def _run_echomimic_impl(avatar_image_path: str, audio_path: str,
                         pose_dir: str, output_path: str) -> str:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
    except Exception:
        duration = 5.0

    pose_frame_count = len([f for f in os.listdir(pose_dir) if f.endswith(".npy")])
    num_frames = min(pose_frame_count, max(24, int(duration * 24)))

    wav_16k = audio_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_16k],
        check=True, capture_output=True
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(_SCRIPT)
        script_path = f.name

    from pipeline.gpu_env import gpu_env
    env = gpu_env()
    env["FFMPEG_PATH"] = "/usr/bin"

    log.info(f"EchoMimicV2: {num_frames} frames, pose_dir={pose_dir}, ref={avatar_image_path}")
    proc = subprocess.run(
        [VENV_PYTHON, script_path,
         ECHOMIMIC_DIR, avatar_image_path, wav_16k, pose_dir, output_path, str(num_frames)],
        capture_output=True, text=True, timeout=900, env=env, encoding="utf-8", errors="replace",
    )
    os.unlink(script_path)
    if os.path.exists(wav_16k):
        os.unlink(wav_16k)

    if proc.returncode != 0 or "DONE:" not in proc.stdout:
        log.error(f"EchoMimicV2 stderr: {proc.stderr[-2000:]}")
        log.error(f"EchoMimicV2 stdout: {proc.stdout[-500:]}")
        raise RuntimeError(f"EchoMimicV2 failed:\n{proc.stderr[-1500:]}")

    log.info(f"EchoMimicV2 done: {output_path}")
    return output_path


def run_echomimic_with_pose_dir(avatar_image_path: str, audio_path: str,
                                 pose_dir: str, output_path: str) -> str:
    """Animate avatar image with audio using EchoMimicV2, given a pre-extracted pose dir."""
    return _run_echomimic_impl(avatar_image_path, audio_path, pose_dir, output_path)


def run_echomimic(avatar_image_path: str, audio_path: str, output_path: str, pose_name: str = None) -> str:
    """Animate avatar image with audio using EchoMimicV2 using a bundled pose template."""
    if pose_name is None:
        pose_name = random.choice(POSE_TEMPLATES)
    pose_dir = os.path.join(ECHOMIMIC_DIR, "assets/halfbody_demo/pose", pose_name)
    return _run_echomimic_impl(avatar_image_path, audio_path, pose_dir, output_path)
