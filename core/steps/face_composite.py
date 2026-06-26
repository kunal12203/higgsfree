"""Face-region compositing: animated face on sharp static body.

EchoMimicV2 animates the whole frame but blurs the body and creates
ghosting artifacts. This blends:
  - Background/body: taken pixel-perfect from the reference portrait
  - Face region: taken from EchoMimicV2 animated frames

Per-frame steps:
  1. Detect face bbox in animated frame
  2. Build an elliptical feathered mask around the face (includes forehead + chin + sides)
  3. Blend: output = mask*animated + (1-mask)*reference
"""
import os
import logging
import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _load_face_detector():
    import insightface
    from insightface.app import FaceAnalysis
    from pipeline.gpu_env import gpu_env
    env = gpu_env()
    # Ensure LD_LIBRARY_PATH is set in this process
    os.environ["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def composite_face_onto_reference(
    animated_video: str,
    reference_image: str,
    output_video: str,
    feather_ratio: float = 0.35,
) -> str:
    """For each frame in animated_video, keep face region animated, body from reference.

    feather_ratio: how much to expand the face mask beyond the detected bbox (0.35 = 35% padding)
    """
    app = _load_face_detector()

    ref_bgr = cv2.imread(reference_image)
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    cap = cv2.VideoCapture(animated_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize reference to match video frame size
    ref_bgr = cv2.resize(ref_bgr, (w, h))

    tmp_video = output_video + ".noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    # Detect face in reference once to get a stable fallback bbox
    ref_faces = app.get(ref_bgr)
    ref_bbox = None
    if ref_faces:
        f = max(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        ref_bbox = f.bbox

    log.info(f"Compositing {total} frames ({w}x{h})...")
    last_bbox = ref_bbox  # fallback to previous frame's bbox if detection fails

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face in animated frame
        faces = app.get(frame)
        if faces:
            bbox = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).bbox
            last_bbox = bbox
        else:
            bbox = last_bbox

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fw = x2 - x1
            fh = y2 - y1

            # Expand bbox by feather_ratio to include forehead, chin, neck
            pad_x = int(fw * feather_ratio)
            pad_y_top = int(fh * feather_ratio * 0.5)   # tight — hair from reference, avoids top ghost
            pad_y_bot = int(fh * (feather_ratio + 0.1))  # neck

            mx1 = max(0, x1 - pad_x)
            my1 = max(0, y1 - pad_y_top)
            mx2 = min(w, x2 + pad_x)
            my2 = min(h, y2 + pad_y_bot)

            # Elliptical feathered mask — smooth blend at edges
            mask = np.zeros((h, w), dtype=np.float32)
            cx_m = (mx1 + mx2) // 2
            cy_m = (my1 + my2) // 2
            axes = ((mx2 - mx1) // 2, (my2 - my1) // 2)
            cv2.ellipse(mask, (cx_m, cy_m), axes, 0, 0, 360, 1.0, -1)

            # Gaussian blur for feathered edge — larger kernel = softer blend
            blur_k = max(21, min(axes) // 2 * 2 + 1)
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
            mask3 = mask[:, :, np.newaxis]  # (H,W,1)

            composited = (frame.astype(np.float32) * mask3 +
                          ref_bgr.astype(np.float32) * (1 - mask3))
            out_frame = composited.astype(np.uint8)
        else:
            out_frame = ref_bgr.copy()

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Mux original audio back
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", animated_video,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        output_video,
    ], check=True, capture_output=True)
    os.remove(tmp_video)

    log.info(f"Face composite done: {output_video}")
    return output_video
