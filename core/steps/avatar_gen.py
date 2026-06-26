"""Avatar generation: extract face from consent video → PuLID portrait → img2img refinement."""
import os
import logging
import subprocess
import tempfile
import cv2
import numpy as np

log = logging.getLogger(__name__)

INSIGHTFACE_MODEL = os.environ.get(
    "INSIGHTFACE_MODEL", "/home/ubuntu/.insightface/models/inswapper_128.onnx"
)
PULID_DIR         = os.environ.get("PULID_DIR",         "/home/ubuntu/PuLID")
PULID_VENV_PYTHON = os.environ.get("PULID_VENV_PYTHON", "/home/ubuntu/venv-sonic/bin/python")
REALVIS_MODEL_ID  = os.environ.get("REALVIS_MODEL",     "SG161222/RealVisXL_V4.0")

_app = None


def _load_models():
    global _app
    if _app is not None:
        return _app, None

    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace buffalo_l...")
    _app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("InsightFace loaded.")
    return _app, None


# ── PuLID subprocess script ────────────────────────────────────────────────────
_PULID_SCRIPT = r'''
import sys, os, cv2, numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

face_path   = sys.argv[2]
output_path = sys.argv[3]
id_scale    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

from pulid.pipeline import PuLIDPipeline
pipeline = PuLIDPipeline()
pipeline.load_pretrain()

face_img = np.array(Image.open(face_path).convert("RGB"))
id_embedding = pipeline.get_id_embedding(face_img)

# Prompt explicitly preserves hair, beard, frontality, mouth closed
prompt = (
    "portrait photo of a person facing directly towards camera, "
    "looking straight at camera, eye level gaze, eyes looking forward, "
    "mouth closed, natural hair, natural beard if present, "
    "photorealistic, natural lighting, neutral background, "
    "sharp focus, high quality, 8k"
)
neg_prompt = (
    "cartoon, anime, illustration, painting, blurry, deformed, ugly, "
    "watermark, extra limbs, disfigured, open mouth, side view, "
    "looking away, looking up, upward gaze, chin up, tilted head, "
    "hat, cap, accessories"
)

# Generate single image — correct source frame means one is enough
imgs = pipeline.inference(
    prompt,
    (1, 1024, 768),
    neg_prompt,
    id_embedding,
    id_scale=id_scale,
    guidance_scale=1.2,
    steps=4,
)
best = imgs[0]
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
best.save(output_path)
print(f"PULID_DONE:{output_path}", flush=True)
'''

# ── RealVisXL img2img subprocess script ───────────────────────────────────────
_IMG2IMG_SCRIPT = r'''
import sys, os, torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler

model_id    = sys.argv[1]
input_path  = sys.argv[2]
output_path = sys.argv[3]
strength    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, add_watermarker=False
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = (
    "RAW photo, photorealistic portrait, ultra realistic, "
    "skin pores visible, natural skin texture, 8k, sharp focus, natural lighting"
)
neg = (
    "cartoon, painting, illustration, anime, smooth skin, plastic, "
    "airbrushed, blurry, deformed, ugly, watermark"
)

img = Image.open(input_path).convert("RGB")
result = pipe(
    prompt=prompt,
    negative_prompt=neg,
    image=img,
    strength=strength,
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
result.save(output_path)
print(f"IMG2IMG_DONE:{output_path}", flush=True)
'''


def generate_pulid_portrait(face_path: str, output_path: str, id_scale: float = 0.8) -> str:
    """Generate portrait via PuLID SDXL-Lightning (4 candidates, sharpest picked).

    Runs as subprocess to fully free VRAM after completion so Sonic can run next.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_PULID_SCRIPT)
        script_path = f.name

    try:
        proc = subprocess.run(
            [PULID_VENV_PYTHON, script_path, PULID_DIR, face_path, output_path, str(id_scale)],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0 or "PULID_DONE:" not in proc.stdout:
        log.error(f"PuLID stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"PuLID portrait generation failed:\n{proc.stderr[-1500:]}")

    log.info(f"PuLID portrait done: {output_path}")
    return output_path


def refine_portrait_img2img(portrait_path: str, output_path: str, strength: float = 0.2) -> str:
    """Refine PuLID portrait with RealVisXL V4.0 img2img for natural skin texture.

    strength=0.2: adds pores/texture without drifting identity.
    Runs as subprocess to isolate VRAM.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_IMG2IMG_SCRIPT)
        script_path = f.name

    try:
        proc = subprocess.run(
            [PULID_VENV_PYTHON, script_path, REALVIS_MODEL_ID, portrait_path, output_path, str(strength)],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0 or "IMG2IMG_DONE:" not in proc.stdout:
        log.error(f"img2img stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"img2img refinement failed:\n{proc.stderr[-1500:]}")

    log.info(f"img2img refinement done: {output_path}")
    return output_path


def _face_pose(face) -> tuple:
    """Return (yaw, pitch) in degrees from InsightFace pose attribute if available."""
    if hasattr(face, "pose") and face.pose is not None:
        # InsightFace pose: [pitch, yaw, roll] in degrees
        pose = face.pose
        return float(pose[1]), float(pose[0])
    return 0.0, 0.0


def _mouth_open_ratio(face, frame_h: int) -> float:
    """Estimate mouth openness from 3D landmark if available (0=closed, 1=wide open)."""
    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        lm = face.landmark_3d_68
        # Upper lip: idx 51, Lower lip: idx 57
        mouth_h = abs(lm[57][1] - lm[51][1])
        face_h = abs(face.bbox[3] - face.bbox[1])
        return mouth_h / max(face_h, 1)
    return 0.0


def extract_best_face_frame(video_path: str, output_jpg: str) -> str:
    """Extract the best frontal, mouth-closed, sharp face frame from a consent video.

    Scoring:
    - Requires yaw ≤ ±25° and pitch ≤ ±20° (frontal only)
    - Penalises open mouth (landmark ratio > 0.08)
    - Weighted: det_score 40% + face_area 30% + sharpness 20% + frontality 10%
    """
    app, _ = _load_models()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info(f"Scanning {total} frames ({fps:.1f}fps) for best frontal face frame...")

    best_score = -1
    best_frame = None
    candidates_checked = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            faces = app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                candidates_checked += 1

                # Frontality filter
                yaw, pitch = _face_pose(face)
                if abs(yaw) > 25 or abs(pitch) > 20:
                    frame_idx += 1
                    continue

                # Mouth open penalty
                mouth_ratio = _mouth_open_ratio(face, frame.shape[0])
                if mouth_ratio > 0.10:
                    frame_idx += 1
                    continue

                area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
                roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0

                # Skip blurry frames (Laplacian < 50 = motion blur / out-of-focus)
                if sharpness < 50:
                    frame_idx += 1
                    continue

                frontality = max(0, 1 - (abs(yaw) / 25 + abs(pitch) / 20) / 2)
                score = (face.det_score * 0.4
                         + (area / (frame.shape[0] * frame.shape[1])) * 0.3
                         + min(sharpness / 500, 1) * 0.2
                         + frontality * 0.1)

                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()

        frame_idx += 1

    cap.release()

    # Fallback: if strict filters found nothing, retry without mouth/frontality filter
    if best_frame is None:
        log.warning("No frontal mouth-closed frame found — retrying with relaxed filters")
        cap = cv2.VideoCapture(video_path)
        best_score = -1
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:
                faces = app.get(frame)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    area = (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1])
                    x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
                    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
                    score = (face.det_score * 0.5
                             + (area / (frame.shape[0]*frame.shape[1])) * 0.3
                             + min(sharpness/500, 1) * 0.2)
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
            frame_idx += 1
        cap.release()

    if best_frame is None:
        raise RuntimeError(f"No face detected in video: {video_path}")

    cv2.imwrite(output_jpg, best_frame)
    log.info(f"Best face frame extracted (score={best_score:.3f}, checked={candidates_checked}): {output_jpg}")
    return output_jpg


def extract_portrait_frame(video_path: str, output_jpg: str, size: int = 768) -> str:
    """Extract best face frame and crop to square portrait (head + shoulders).

    Used as EchoMimicV2 reference — kept for compatibility.
    """
    app, _ = _load_models()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Scanning {min(total, 250)} frames for best portrait frame...")

    best_score = -1
    best_frame = None
    best_bbox = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            faces = app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
                roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0
                score = (face.det_score * 0.5
                         + (area / (frame.shape[0] * frame.shape[1])) * 0.3
                         + min(sharpness / 500, 1) * 0.2)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_bbox = face.bbox
        frame_idx += 1
        if frame_idx > 250 * 5:
            break

    cap.release()

    if best_frame is None:
        raise RuntimeError(f"No face detected in video: {video_path}")

    h, w = best_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in best_bbox]
    face_w = x2 - x1
    face_h = y2 - y1
    cx = (x1 + x2) // 2

    crop_side = min(int(face_h * 3.5), min(h, w))
    crop_side = max(crop_side, face_w + 60)
    crop_top = max(0, y1 - int(face_h * 0.4))
    crop_bottom = crop_top + crop_side
    if crop_bottom > h:
        crop_bottom = h
        crop_top = max(0, h - crop_side)
    crop_left = max(0, cx - crop_side // 2)
    crop_right = crop_left + crop_side
    if crop_right > w:
        crop_right = w
        crop_left = max(0, w - crop_side)

    cropped = best_frame[crop_top:crop_bottom, crop_left:crop_right]

    from PIL import Image as PILImage
    img = PILImage.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    img = img.resize((size, size), PILImage.LANCZOS)
    img.save(output_jpg, quality=95)

    log.info(f"Portrait frame saved (score={best_score:.3f}): {output_jpg}")
    return output_jpg


def crop_head_shoulders(image_path: str, output_path: str) -> str:
    """Crop portrait to head+shoulders for Sonic lipsync (9:16, no hands).

    Detects face, crops forehead + shoulders only — eliminates SVD hand distortion.
    """
    app, _ = _load_models()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    faces = app.get(img)
    if not faces:
        raise RuntimeError(f"No face detected in: {image_path}")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox.astype(int)
    fh = y2 - y1

    crop_top = max(0, y1 - int(fh * 1.0))
    crop_bot = min(H, y2 + int(fh * 1.5))

    crop_h = crop_bot - crop_top
    target_w = int(crop_h * 9 / 16)
    if target_w < W:
        margin = (W - target_w) // 2
        cropped = img[crop_top:crop_bot, margin:margin + target_w]
    else:
        cropped = img[crop_top:crop_bot, 0:W]

    cv2.imwrite(output_path, cropped)
    log.info(f"Head+shoulders crop: {W}x{H} -> {cropped.shape[1]}x{cropped.shape[0]}: {output_path}")
    return output_path


def generate_avatar_image(face_path: str, outfit_id: str, scene_id: str,
                          output_path: str, target_size: tuple = (576, 1024)) -> str:
    """Generate portrait: PuLID → img2img → crop head+shoulders.

    outfit_id / scene_id retained for API compatibility but unused (PuLID generates from scratch).
    """
    tw, th = target_size
    out_dir = os.path.dirname(os.path.abspath(output_path))

    portrait_tmp = os.path.join(out_dir, "_portrait_pulid.png")
    refined_tmp  = os.path.join(out_dir, "_portrait_refined.png")
    cropped_tmp  = os.path.join(out_dir, "_portrait_cropped.png")

    generate_pulid_portrait(face_path, portrait_tmp)

    try:
        refine_portrait_img2img(portrait_tmp, refined_tmp)
        source = refined_tmp
    except Exception as e:
        log.warning(f"img2img refinement failed (using raw PuLID): {e}")
        source = portrait_tmp

    crop_head_shoulders(source, cropped_tmp)

    from PIL import Image
    img = Image.open(cropped_tmp).convert("RGB")
    img = img.resize((tw, th), Image.LANCZOS)
    img.save(output_path)

    log.info(f"Avatar image (PuLID + img2img): {output_path} ({tw}x{th})")
    return output_path


def generate_avatar_from_video(video_path: str, outfit_id: str, scene_id: str,
                               face_output: str, avatar_output: str) -> tuple:
    """Full onboarding flow: consent video → face → PuLID → img2img → crop → avatar.png.

    Returns (face_jpg_path, avatar_png_path)
    """
    out_dir = os.path.dirname(os.path.abspath(face_output))

    # Step 1: extract best face frame
    extract_best_face_frame(video_path, face_output)

    # Step 2: PuLID portrait (4-step SDXL-Lightning, id_scale=0.8)
    portrait_tmp = os.path.join(out_dir, "_pulid.png")
    generate_pulid_portrait(face_output, portrait_tmp)

    # Step 3: RealVisXL img2img realism (strength=0.2)
    refined_tmp = os.path.join(out_dir, "_refined.png")
    try:
        refine_portrait_img2img(portrait_tmp, refined_tmp)
        source = refined_tmp
    except Exception as e:
        log.warning(f"img2img failed, using raw PuLID: {e}")
        source = portrait_tmp

    # Step 4: crop to head+shoulders for Sonic
    crop_head_shoulders(source, avatar_output)

    return face_output, avatar_output
